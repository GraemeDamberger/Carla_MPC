import copy
import os
import random
import time
from pathlib import Path

try:
    import cv2  # only needed when config['record'] is True (video output)
except ImportError:
    cv2 = None  # headless HPC runs don't record; keeps import working without opencv
import carla
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # only needed for per-rollout diagnostic plots
except ImportError:
    plt = None  # headless HPC sweeps skip plotting (config['save_plots'] = False)
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import NonlinearConstraint, minimize

from Experiments.Comparison.config import config, SimpleNN, ResidualNN
from Shared.funcs import bike, legendre, get_mpc_reference, global_to_local
from Shared.logging_utils import save_model, save_plot


class AdaptiveRBFController:
    def __init__(self, state_dim, control_dim, num_basis=50, gamma=80.0,
                 sigma=0.7, weight_clip=20.0, seed=0):
        rng = np.random.default_rng(seed)
        self.nu = control_dim
        self.nb = num_basis
        self.gamma = gamma
        self.sigma = sigma
        self.weight_clip = weight_clip
        self.centers = rng.uniform(-np.pi, np.pi, size=(num_basis, state_dim))
        self.W = np.zeros((num_basis, control_dim))

    def phi(self, x):
        diff = self.centers - x
        sq_norm = np.sum(diff ** 2, axis=1)
        phi = np.exp(-sq_norm / (2 * self.sigma ** 2))
        phi /= np.sum(phi) + 1e-8
        return phi

    def forward_control(self, x):
        # W is (num_basis, 1); take a scalar dot product. numpy 2.x rejects
        # float() on a 1-d size-1 array (W.T @ phi), so index the weight column.
        return float(self.W[:, 0] @ self.phi(x))

    def update(self, x, e, dt):
        dW = self.gamma * np.outer(self.phi(x), e)
        self.W = np.clip(self.W + dt * dW, -self.weight_clip, self.weight_clip)


def compute_speed_profile(path_xy, a_lat_max, a_acc_max, a_dec_max, v_min, v_max):
    """Curvature-aware target-speed profile along a reference path.

    path_xy : (2, M) global reference points.
    Returns (s, v): cumulative arc length and target speed at each point.

    Method (standard trajectory speed planning):
      1. curvature  kappa = |d(heading)| / d(arc length)
      2. curvature-limited speed  v = sqrt(a_lat_max / kappa), clamped [v_min, v_max]
      3. backward pass (decel limit) then forward pass (accel limit) so the
         profile is longitudinally feasible (brakes *before* a curve).
    """
    P   = np.asarray(path_xy, dtype=float).T          # (M, 2)
    d   = np.diff(P, axis=0)                           # (M-1, 2)
    seg = np.maximum(np.linalg.norm(d, axis=1), 1e-6)  # (M-1,)
    s   = np.concatenate([[0.0], np.cumsum(seg)])      # (M,)

    psi  = np.arctan2(d[:, 1], d[:, 0])                # (M-1,) segment headings
    dpsi = np.diff(psi)                                # (M-2,)
    dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi        # wrap to (-pi, pi]
    ds_mid = 0.5 * (seg[:-1] + seg[1:])                # (M-2,)

    kappa = np.zeros(len(P))
    kappa[1:-1] = np.abs(dpsi) / np.maximum(ds_mid, 1e-6)

    v = np.clip(np.sqrt(a_lat_max / np.maximum(kappa, 1e-3)), v_min, v_max)

    for k in range(len(v) - 2, -1, -1):               # backward: limit decel
        v[k] = min(v[k], np.sqrt(v[k + 1] ** 2 + 2 * a_dec_max * seg[k]))
    for k in range(1, len(v)):                         # forward: limit accel
        v[k] = min(v[k], np.sqrt(v[k - 1] ** 2 + 2 * a_acc_max * seg[k - 1]))
    return s, v


def cross_track_error(px, py, path_xy, i0, window=300):
    """Perpendicular distance from (px, py) to the reference polyline.

    This is the standard path-tracking metric and, unlike the distance to the
    MPC's time-indexed reference point, it is INDEPENDENT OF SPEED: a vehicle
    that lags behind the reference is not penalised, only one that leaves the
    path. That matters here because the MPC controls steering only — the
    longitudinal channel is a separate PID — so lateral deviation is the
    quantity the controller is actually responsible for.

    Searches segments within +/- `window` of index `i0` (the previous closest
    index) so the cost stays constant regardless of path length.

    Returns (distance [m], index of the closest segment).
    """
    n  = path_xy.shape[1]
    lo = max(0, i0 - window)
    hi = min(n - 1, i0 + window)
    if hi - lo < 1:
        d = float(np.hypot(px - path_xy[0, lo], py - path_xy[1, lo]))
        return d, lo

    A  = path_xy[:, lo:hi]        # segment starts  (2, m)
    B  = path_xy[:, lo + 1:hi + 1]  # segment ends   (2, m)
    AB = B - A
    AP = np.array([[px], [py]]) - A

    denom = np.sum(AB * AB, axis=0)
    t     = np.clip(np.sum(AP * AB, axis=0) / np.maximum(denom, 1e-12), 0.0, 1.0)
    C     = A + AB * t            # closest point on each segment
    d     = np.linalg.norm(np.array([[px], [py]]) - C, axis=0)

    k = int(np.argmin(d))
    return float(d[k]), lo + k


def apply_wheel_faults(vehicle, flat_tire=False, surface=None):
    """Apply plant faults by scaling the vehicle's OWN default wheel parameters.

    surface   : None | 'wet' | 'icy' — uniform peak-friction scaling on all wheels,
                from published tire-road peak friction coefficients.
    flat_tire : one wheel deflated. A deflation is not principally a peak-mu
                change: cornering stiffness collapses, the rolling radius drops
                and rolling resistance rises sharply, giving an asymmetric yaw
                moment. Modelled on config['flat_*_scale'].

    Returns a dict of the applied changes, for logging/provenance.
    """
    pc      = vehicle.get_physics_control()
    wheels  = pc.wheels
    applied = {}

    if surface is not None:
        scale = config[f"{surface}_friction_scale"]
        for w in wheels:
            w.tire_friction *= scale
        applied["surface"] = {"kind": surface, "friction_scale": scale,
                              "tire_friction": wheels[0].tire_friction}

    if flat_tire:
        idx = config['flat_tire_wheel']
        w   = wheels[idx]
        before = {"lat_stiff_value": w.lat_stiff_value, "radius": w.radius,
                  "damping_rate": w.damping_rate, "tire_friction": w.tire_friction}
        w.lat_stiff_value *= config['flat_lat_stiff_scale']
        w.radius          *= config['flat_radius_scale']
        w.damping_rate    *= config['flat_damping_scale']
        w.tire_friction   *= config['flat_friction_scale']
        applied["flat_tire"] = {
            "wheel": idx, "before": before,
            "after": {"lat_stiff_value": w.lat_stiff_value, "radius": w.radius,
                      "damping_rate": w.damping_rate, "tire_friction": w.tire_friction},
        }

    pc.wheels = wheels
    vehicle.apply_physics_control(pc)
    return applied


def crosswind_force(vehicle, wind_vec):
    """Steady-crosswind aerodynamic force on the vehicle, in world coordinates.

    Uses the relative air velocity (wind minus vehicle velocity), so the
    aerodynamic yaw angle — and hence the side force — changes as the vehicle
    turns, rather than being a fixed lateral push.

        F_side = 0.5 * rho * V_rel^2 * A * C_S,    C_S = C_S,max * sin(beta)

    Only the SIDE force is applied. CARLA already simulates longitudinal
    aerodynamic drag through the vehicle's own drag_coefficient, so adding a
    drag term here would double-count it; the side force is the component
    CARLA does not model.

    Returns (fx, fy) in Newtons, or None when the relative wind is negligible.
    """
    v      = vehicle.get_velocity()
    v_rel  = wind_vec - np.array([v.x, v.y])
    speed  = float(np.linalg.norm(v_rel))
    if speed < 1e-3:
        return None

    yaw = np.deg2rad(vehicle.get_transform().rotation.yaw)
    fwd = np.array([np.cos(yaw),  np.sin(yaw)])
    lat = np.array([-np.sin(yaw), np.cos(yaw)])

    # aerodynamic yaw angle between the relative wind and the vehicle axis
    beta = np.arctan2(float(v_rel @ lat), float(v_rel @ fwd))
    q    = 0.5 * config['air_density'] * speed ** 2 * config['frontal_area']

    f_lat = q * config['side_force_coeff'] * np.sin(beta)
    f     = f_lat * lat
    return float(f[0]), float(f[1])


def simulate_carla(trial_name, log_dir, method='normal', steering_force=0.0,
                   flat_tire=False, surface=None, wind=False,
                   spawn_index=None, model_path=None):
    """
    Run one CARLA simulation episode.

    method        : 'normal' | 'tube' | 'replay_buffer' | 'residual_dynamics' | 'tube_adaptive'
    steering_force: constant offset added to every steering command (actuator bias)
    flat_tire     : deflate one wheel (see apply_wheel_faults)
    surface       : None | 'wet' | 'icy' — uniform road-friction reduction
    wind          : apply the steady-crosswind aerodynamic model
    spawn_index   : spawn-point index selecting the route (default config route 0)
    """
    if model_path is None:
        model_path = config['model_path']

    # ------------------------------------------------------------------ config
    N            = config['N']
    dt           = config['dt']
    Np           = config['Np']
    scale_V      = config['scale_V']
    Steps        = config['steps']
    buffer_size  = config['buffer_size']
    batch_size   = config['batch_size']
    K_tube          = np.array(config['K_tube'])
    K_tube_adaptive = np.array(config['K_tube_adaptive'])
    seed            = config['seed']
    R_weight        = config['R']   # control-effort weight in the MPC cost

    if spawn_index is None:
        spawn_index = config['route_spawn_indices'][0]

    # MPC bicycle model
    sys = bike(config['l'], dt)
    leg = legendre(Np * dt, N, dt)
    P   = leg.P[:Np]
    Q   = P.T @ (config['Q'] * np.eye(Np)) @ P

    # LMU matrices used by replay_buffer and residual_dynamics
    theta_lmu = Np * dt
    A_lmu = np.zeros((N, N))
    B_lmu = np.zeros(N)
    for ii in range(N):
        B_lmu[ii] = (-1.) ** ii * (2 * ii + 1)
        for jj in range(N):
            A_lmu[ii, jj] = (2*ii+1) * (-1 if ii < jj else (-1.)**(ii-jj+1))
    A_lmu /= theta_lmu
    B_lmu /= theta_lmu

    # ------------------------------------------------------------------ models
    model_norm = SimpleNN(N, 2 * N)
    model_norm.load_state_dict(torch.load(model_path, weights_only=True))
    model_norm.eval()
    model_norm.to('cpu')

    model_online   = None
    model_residual = None
    optim_online   = None
    criterion      = nn.MSELoss()

    if method == 'replay_buffer':
        model_online = copy.deepcopy(model_norm)
        optim_online = torch.optim.Adam(
            model_online.parameters(),
            lr=config['online_lr_replay'], weight_decay=config['online_weight_decay'],
        )
    elif method == 'residual_dynamics':
        model_residual = ResidualNN(N, 2 * N)
        optim_online = torch.optim.Adam(
            model_residual.parameters(),
            lr=config['online_lr_residual'], weight_decay=config['online_weight_decay'],
        )

    adaptive = None
    if method == 'tube_adaptive':
        adaptive = AdaptiveRBFController(
            state_dim=1,
            control_dim=1,
            num_basis=config['rbf_num_basis'],
            gamma=config['rbf_gamma'],
            sigma=config['rbf_sigma'],
            weight_clip=config['rbf_weight_clip'],
        )

    # ------------------------------------------------------------------ helpers
    def get_Mx_direct(M_u):
        U = leg.decode(M_u)
        X = np.zeros((Np, 3))
        for k in range(1, Np):
            X[k] = sys.dynamics(X[k-1], scale_V, U[k-1])
        return leg.encode(X[:, 0]), leg.encode(X[:, 1])

    def get_Mx_neural(M_u, model):
        with torch.no_grad():
            pred = np.array(model(torch.tensor(M_u, dtype=torch.float32)))
        return pred[:N], pred[N:]

    def cost_fun(M_u, _X0, V, X_des, Y_des, Q_):
        if method == 'replay_buffer':
            # online-adapted copy of the network
            tx, ty = get_Mx_neural(M_u, model_online)
        elif method == 'residual_dynamics':
            # frozen base network + online-learned residual correction
            t = torch.tensor(M_u, dtype=torch.float32)
            with torch.no_grad():
                base  = np.array(model_norm(t))
                resid = np.array(model_residual(t))
            tx = base[:N] + resid[:N]
            ty = base[N:] + resid[N:]
        else:
            # normal / tube: frozen offline network
            tx, ty = get_Mx_neural(M_u, model_norm)

        tx = (V / scale_V) * tx
        ty = (V / scale_V) * ty
        ex = tx - leg.encode(X_des)
        ey = ty - leg.encode(Y_des)
        # tracking cost + control-effort penalty (R balances aggressive steering)
        return float(ex @ Q_ @ ex + ey @ Q_ @ ey) + R_weight * float(M_u @ M_u)

    def tube_control(X_prev, X_curr, V, U_nom):
        X_hat = sys.dynamics(X_prev, V, U_nom)
        return float(K_tube @ (X_curr - X_hat))

    def tube_adaptive_control(X_prev, X_curr, V, U_nom):
        X_hat   = sys.dynamics(X_prev, V, U_nom)
        e       = X_curr - X_hat
        u_tube  = float(K_tube_adaptive @ e)
        u_adapt = adaptive.forward_control(X_curr[2])
        adaptive.update(X_curr[2], -e[2], dt)
        return u_tube + u_adapt

    def constraint_decode_specific_points(M_u):
        return leg.P[sample_points] @ M_u

    def lmu_step(state, u_val):
        return state + (A_lmu @ state + B_lmu * u_val) * dt

    def to_leg_coeffs(lmu_state):
        v = lmu_state.copy()
        v[1::2] *= -1
        return leg.encode(leg.decode(v))

    def add_to_buffer(buf, item):
        buf.append(item)
        if len(buf) > buffer_size:
            buf.pop(0)
        return buf

    def process_image(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        out.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    def get_waypoints_from_vehicle(vehicle, distance=2.0, num_points=50):
        transform  = vehicle.get_transform()
        current_wp = vehicle.get_world().get_map().get_waypoint(
            transform.location, project_to_road=True, lane_type=carla.LaneType.Driving,
        )
        waypoints = []
        for _ in range(num_points):
            waypoints.append(current_wp.transform.location)
            nexts = current_wp.next(distance)
            if not nexts:
                break
            current_wp = nexts[0]
        return waypoints

    def draw_trajectory(world, X, Y, color=carla.Color(0, 255, 0), lifetime=0.1):
        for k in range(len(X) - 1):
            world.debug.draw_line(
                carla.Location(x=float(X[k]),   y=float(Y[k]),   z=0.2),
                carla.Location(x=float(X[k+1]), y=float(Y[k+1]), z=0.2),
                thickness=0.1, color=color, life_time=lifetime,
            )

    def update_spectator():
        t   = vehicle.get_transform()
        loc = t.location - t.get_forward_vector() * 8
        loc.z += 3
        spectator.set_transform(carla.Transform(loc, t.rotation))

    # ------------------------------------------------------------------ CARLA
    error_array = np.zeros((Steps, 2))   # legacy: offset to the MPC reference point
    xtrack      = np.zeros(Steps)        # cross-track: perpendicular distance to path
    ct_idx      = 0                      # running closest index on the reference

    port   = int(os.environ.get("CARLA_PORT", 2000))
    client = carla.Client("localhost", port)
    client.set_timeout(30.0)   # map loads can be slow
    world = client.get_world()

    # Load the evaluation map once per server; reuse it on later rollouts.
    target_map = config['map']
    if target_map not in world.get_map().name:
        world = client.load_world(target_map)

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = dt
    settings.random_seed         = seed
    settings.no_rendering_mode   = config['no_rendering_mode']
    world.apply_settings(settings)
    world.tick()

    vehicle_bp   = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()
    spawn_point  = spawn_points[spawn_index % len(spawn_points)]
    vehicle      = world.spawn_actor(vehicle_bp[3], spawn_point)
    time.sleep(1)

    # Plant-fault disturbances (scaled from this vehicle's own defaults).
    if flat_tire or surface is not None:
        applied = apply_wheel_faults(vehicle, flat_tire=flat_tire, surface=surface)
        print(f'  [{trial_name}] wheel faults: {applied}')
        world.tick()

    # Steady crosswind: constant world-frame wind vector, force recomputed each
    # step from the relative air velocity.
    wind_vec = None
    if wind:
        wind_speed = config['wind_speed_kmh'] / 3.6
        wind_dir   = np.deg2rad(config['wind_dir_deg'])
        wind_vec   = wind_speed * np.array([np.cos(wind_dir), np.sin(wind_dir)])
        print(f'  [{trial_name}] crosswind: {config["wind_speed_kmh"]} km/h '
              f'({wind_speed:.1f} m/s) bearing {config["wind_dir_deg"]}deg')

    # The RGB camera is only needed for video recording. Under -nullrhi
    # (headless HPC) there is no render device, so spawning a camera sensor
    # crashes the server the moment the client connects. Only create it when
    # actually recording.
    camera = None
    if config['record']:
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera    = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=-5.5, z=2.5)),
            attach_to=vehicle,
        )
        video_dir = Path(log_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(
            str(video_dir / f"simulation_{trial_name}_{method}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), 1.0 / dt, (800, 600),
        )
        camera.listen(process_image)

    spectator = world.get_spectator()
    world.tick()

    ref_points = config['ref_points']
    waypoints  = get_waypoints_from_vehicle(vehicle, num_points=ref_points)
    for wp in waypoints:
        world.debug.draw_point(wp, size=0.1, color=carla.Color(r=255, g=0, b=0), life_time=500.0)

    X_des = np.array([[wp.x, wp.y] for wp in waypoints]).T  # (2, ref_points)

    # Curvature-aware target-speed profile along the route (arc length -> speed).
    s_prof, v_prof = compute_speed_profile(
        X_des, config['a_lat_max'], config['a_acc_max'], config['a_dec_max'],
        config['v_min'], config['v_max'],
    )

    X_traj = np.zeros((3, Steps))
    X_curr = vehicle.get_location()
    theta  = np.deg2rad(vehicle.get_transform().rotation.yaw)
    X_traj[:, 0] = [X_curr.x, X_curr.y, theta]

    num_sample_points    = 5
    sample_points        = np.linspace(0, Np, num_sample_points).astype(int)
    U_sampled_constraint = NonlinearConstraint(
        constraint_decode_specific_points,
        -np.pi / 2.5 * np.ones(num_sample_points),
         np.pi / 2.5 * np.ones(num_sample_points),
    )

    options = {'eps': config['eps']}
    s0, error_i  = 0.0, 0.0
    V_log, U_mem = [], [0]
    F_wind_log   = []          # |crosswind force| per step, for reporting
    M_u          = np.zeros(N)

    # online-learning memory state
    u_lmu, x_lmu, y_lmu = np.zeros(N), np.zeros(N), np.zeros(N)
    buffer       = []
    window_steps = Np

    kp, kd, ki    = 0.5, 0.1, 0.2
    prev_speed    = 0.0
    U_prev_nom    = 0.0   # nominal steer applied at the previous step (for tube feedback)

    tag = f"{trial_name}_{method}".replace('.', '_')

    try:
        for i in range(1, Steps):
            if i % (Steps // 5) == 0:
                print(f'  [{tag}] {100 * i // Steps}%')

            # speed control — target speed from the curvature-aware profile,
            # sampled at the current arc length along the route.
            desired_speed = float(np.interp(s0, s_prof, v_prof))
            vel        = vehicle.get_velocity()
            cur_speed  = np.linalg.norm([vel.x, vel.y, vel.z])
            accel      = (cur_speed - prev_speed) / 0.05
            prev_speed = cur_speed
            error_i   += (desired_speed - cur_speed) * dt
            throttle   = np.clip(
                kp * (desired_speed - cur_speed) - kd * accel + ki * error_i,
                0.0, 1.0,
            )

            # MPC reference
            s0 += cur_speed * dt
            x_mpc_ref, y_mpc_ref = get_mpc_reference(
                X_des[0], X_des[1], cur_speed, s0, Np, dt,
            )
            draw_trajectory(world, x_mpc_ref, y_mpc_ref)
            x_ref_local, y_ref_local = global_to_local(
                x_mpc_ref, y_mpc_ref,
                X_traj[0, i-1], X_traj[1, i-1], X_traj[2, i-1],
            )

            res = minimize(
                cost_fun, M_u, method='SLSQP',
                args=(X_traj[:, i-1], cur_speed, x_ref_local, y_ref_local, Q),
                constraints=U_sampled_constraint,
                options=options,
            )
            M_u = res.x
            # If SLSQP diverges (e.g. an unstable adaptive/tube gain), it can
            # return non-finite coefficients. Reset the warm start so the solver
            # can recover next step instead of propagating NaN forever.
            if not np.all(np.isfinite(M_u)):
                M_u = np.zeros(N)
            U = leg.decode(M_u)
            U_mem.append(U[0])

            # steer: nominal + tube correction (if applicable) + constant bias disturbance
            # tube: predict X[i-1] using the control that was actually applied at step i-2,
            # then correct for the observed heading error (plant vs model)
            if method == 'tube' and i >= 2:
                U_tube  = tube_control(X_traj[:, i-2], X_traj[:, i-1], cur_speed, U_prev_nom)
                U_steer = U[0] + U_tube + steering_force
            elif method == 'tube_adaptive' and i >= 2:
                U_tube  = tube_adaptive_control(X_traj[:, i-2], X_traj[:, i-1], cur_speed, U_prev_nom)
                U_steer = U[0] + U_tube + steering_force
            else:
                U_steer = U[0] + steering_force
            U_prev_nom = U[0]

            # Sanitize the steering command before it reaches CARLA: a non-finite
            # or out-of-range steer crashes the server (this is what wiped every
            # tube_adaptive trial). Clamp to the valid [-1, 1] and neutralize NaN.
            if not np.isfinite(U_steer):
                U_steer = 0.0
            U_steer = float(np.clip(U_steer, -1.0, 1.0))

            brake = 0.0
            if cur_speed - desired_speed > 2.0:
                brake    = 0.3
                throttle = 0.0

            control          = carla.VehicleControl()
            control.throttle = float(throttle)
            control.brake    = float(brake)
            control.steer    = U_steer
            vehicle.apply_control(control)
            if wind_vec is not None:
                wf = crosswind_force(vehicle, wind_vec)
                if wf is not None:
                    vehicle.add_force(carla.Vector3D(x=wf[0], y=wf[1], z=0.0))
                    F_wind_log.append(float(np.hypot(*wf)))
            V_log.append(cur_speed)

            world.tick()
            update_spectator()

            X_curr = vehicle.get_location()
            world.debug.draw_point(X_curr, size=0.1, color=carla.Color(r=0, g=0, b=255), life_time=500.0)
            theta          = np.deg2rad(vehicle.get_transform().rotation.yaw)
            X_traj[:, i]   = [X_curr.x, X_curr.y, theta]
            error_array[i-1] = np.array([x_mpc_ref[0], y_mpc_ref[0]]) - X_traj[:2, i-1]
            xtrack[i-1], ct_idx = cross_track_error(
                X_traj[0, i], X_traj[1, i], X_des, ct_idx)

            # online learning (replay_buffer and residual_dynamics)
            if method in ('replay_buffer', 'residual_dynamics'):
                u_lmu = lmu_step(u_lmu, U[0])
                x_lmu = lmu_step(x_lmu, X_traj[0, i])
                y_lmu = lmu_step(y_lmu, X_traj[1, i])

                U_data = to_leg_coeffs(u_lmu)

                origin_step = max(0, i - Np)
                ox, oy, otheta = X_traj[:, origin_step]
                x_history = leg.decode(to_leg_coeffs(x_lmu))
                y_history = leg.decode(to_leg_coeffs(y_lmu))
                x_local, y_local = global_to_local(x_history, y_history, ox, oy, otheta)
                X_data = leg.encode(x_local)
                Y_data = leg.encode(y_local)

                M_u_t = torch.tensor(U_data, dtype=torch.float32)
                M_x_t = torch.tensor(np.hstack((X_data, Y_data)), dtype=torch.float32)

                if i > window_steps:
                    buffer = add_to_buffer(buffer, (M_u_t, M_x_t))

                if i > window_steps + buffer_size:
                    if i == window_steps + buffer_size + 1:
                        print(f'  [{tag}] online training started')
                    batch    = random.sample(buffer, min(len(buffer), batch_size))
                    inp, tgt = zip(*batch)
                    inp      = torch.stack(inp)
                    tgt      = torch.stack(tgt)

                    if method == 'replay_buffer':
                        pred = model_online(inp)
                        loss = criterion(pred, tgt)
                        optim_online.zero_grad()
                        loss.backward()
                        optim_online.step()
                    else:  # residual_dynamics
                        with torch.no_grad():
                            base = model_norm(inp)
                        res_pred = model_residual(inp)
                        loss     = criterion(res_pred, tgt - base)
                        optim_online.zero_grad()
                        loss.backward()
                        optim_online.step()

    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        if config['record']:
            out.release()
        vehicle.destroy()

    # ------------------------------------------------------------------ metrics
    # Cross-track RMSE is the reported metric: perpendicular deviation from the
    # path, independent of speed. The legacy reference-point RMSE is kept for
    # comparison — it mixes lateral error with longitudinal lag, so a slower
    # vehicle scores better on it regardless of path-following quality.
    n_valid     = max(len(V_log), 1)
    rmse_xtrack = float(np.sqrt(np.mean(xtrack[:n_valid] ** 2)))
    rmse_refpt  = float(np.sqrt(np.mean(error_array[:n_valid] ** 2)))
    print(f'  [{tag}] cross-track RMSE = {rmse_xtrack:.4f} m  '
          f'(ref-point RMSE = {rmse_refpt:.4f} m)')

    # ------------------------------------------------------------------ plots
    # Skipped during hyperparameter sweeps (config['save_plots'] = False) and
    # when matplotlib is unavailable — the sweep only needs the returned RMSE.
    if config.get('save_plots', True) and plt is not None:
        # speed: commanded profile vs achieved
        steps_run = len(V_log)
        s_walk    = np.cumsum(np.array(V_log) * dt)
        v_target  = np.interp(s_walk, s_prof, v_prof)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(V_log, label='achieved', linewidth=1.2)
        ax.plot(v_target, 'r--', label='profile target', linewidth=1.2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Speed [m/s]')
        ax.set_title(f'Speed — {tag}')
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        save_plot(log_dir, fig, f'velocity_{tag}')
        plt.close(fig)

        # Top-down trajectory, coloured by CROSS-TRACK error. Only the driven
        # portion of the reference is drawn — the full path is ~4x longer than
        # a 50 s episode covers, and plotting all of it dwarfs the trajectory.
        xt  = xtrack[:steps_run]
        err = np.linalg.norm(error_array[:steps_run], axis=1)
        ref_end = min(X_des.shape[1], ct_idx + 50)
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.plot(X_des[0, :ref_end], X_des[1, :ref_end], '-', color='0.75',
                linewidth=3, label='Reference path (driven)', zorder=1)
        sc = ax.scatter(X_traj[0, :steps_run], X_traj[1, :steps_run],
                        c=xt, cmap='viridis', s=3, zorder=2)
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label('Cross-track error [m]')
        ax.plot(X_traj[0, 0], X_traj[1, 0], 'o', color='tab:green',
                markersize=10, label='Start', zorder=3)
        ax.plot(X_traj[0, steps_run-1], X_traj[1, steps_run-1], 's',
                color='tab:red', markersize=9, label='End', zorder=3)
        subtitle = (f'cross-track RMSE {rmse_xtrack:.3f} m   '
                    f'(ref-point RMSE {rmse_refpt:.3f} m)')
        if F_wind_log:
            subtitle += f'  ·  mean |F_wind| {np.mean(F_wind_log):.0f} N'
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Trajectory — {tag}\n{subtitle}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.4)
        ax.axis('equal')
        plt.tight_layout()
        save_plot(log_dir, fig, f'trajectory_{tag}')
        plt.close(fig)

        # both error definitions against distance travelled, so the difference
        # between lateral deviation and longitudinal lag is visible directly
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(s_walk, xt, linewidth=1.2, label='cross-track (lateral)')
        ax.plot(s_walk, err, linewidth=1.0, alpha=0.7,
                label='to MPC reference point (incl. lag)')
        ax.set_xlabel('Distance along route [m]')
        ax.set_ylabel('Error [m]')
        ax.set_title(f'Error vs distance — {tag}')
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        save_plot(log_dir, fig, f'error_{tag}')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(U_mem)
        ax.set_xlabel('Step')
        ax.set_ylabel('Steer [rad]')
        ax.set_title(f'Control — {tag}')
        ax.grid(True)
        plt.tight_layout()
        save_plot(log_dir, fig, f'control_{tag}')
        plt.close(fig)

    if method == 'replay_buffer':
        save_model(log_dir, model_online, f'model_online_{tag}')
    elif method == 'residual_dynamics':
        save_model(log_dir, model_residual, f'model_residual_{tag}')

    return rmse_xtrack
