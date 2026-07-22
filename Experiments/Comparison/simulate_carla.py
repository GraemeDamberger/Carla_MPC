import copy
import os
import random
import time
from pathlib import Path

import cv2
import carla
import matplotlib.pyplot as plt
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
        return float(self.W.T @ self.phi(x))

    def update(self, x, e, dt):
        dW = self.gamma * np.outer(self.phi(x), e)
        self.W = np.clip(self.W + dt * dW, -self.weight_clip, self.weight_clip)


def simulate_carla(trial_name, log_dir, method='normal', steering_force=0.0, wind_force=0.0, model_path=None):
    """
    Run one CARLA simulation episode.

    method        : 'normal' | 'tube' | 'replay_buffer' | 'residual_dynamics'
    steering_force: constant offset added to every steering command (actuator bias)
    wind_force    : lateral force [N] applied to the vehicle each step (unmodeled disturbance)
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
        return float(ex @ Q_ @ ex + ey @ Q_ @ ey)

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
    error_array = np.zeros((Steps, 2))

    port   = int(os.environ.get("CARLA_PORT", 2000))
    client = carla.Client("localhost", port)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = dt
    settings.random_seed         = seed
    settings.no_rendering_mode   = config['no_rendering_mode']
    world.apply_settings(settings)
    world.tick()

    vehicle_bp  = world.get_blueprint_library().filter('*vehicle*')
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle     = world.spawn_actor(vehicle_bp[3], spawn_point)
    time.sleep(1)

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera    = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=-5.5, z=2.5)),
        attach_to=vehicle,
    )

    if config['record']:
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
    M_u          = np.zeros(N)

    # online-learning memory state
    u_lmu, x_lmu, y_lmu = np.zeros(N), np.zeros(N), np.zeros(N)
    buffer       = []
    window_steps = Np

    kp, kd, ki    = 0.5, 0.1, 0.2
    desired_speed = 15.0
    prev_speed    = 0.0
    U_prev_nom    = 0.0   # nominal steer applied at the previous step (for tube feedback)

    tag = f"{trial_name}_{method}".replace('.', '_')

    try:
        for i in range(1, Steps):
            if i % (Steps // 5) == 0:
                print(f'  [{tag}] {100 * i // Steps}%')

            # speed control
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

            brake = 0.0
            if cur_speed - desired_speed > 2.0:
                brake    = 0.3
                throttle = 0.0

            control          = carla.VehicleControl()
            control.throttle = float(throttle)
            control.brake    = float(brake)
            control.steer    = float(U_steer)
            vehicle.apply_control(control)
            if wind_force != 0.0:
                vehicle.add_force(carla.Vector3D(x=0.0, y=wind_force, z=0.0))
            V_log.append(cur_speed)

            world.tick()
            update_spectator()

            X_curr = vehicle.get_location()
            world.debug.draw_point(X_curr, size=0.1, color=carla.Color(r=0, g=0, b=255), life_time=500.0)
            theta          = np.deg2rad(vehicle.get_transform().rotation.yaw)
            X_traj[:, i]   = [X_curr.x, X_curr.y, theta]
            error_array[i-1] = np.array([x_mpc_ref[0], y_mpc_ref[0]]) - X_traj[:2, i-1]

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
        camera.stop()
        camera.destroy()
        if config['record']:
            out.release()
        vehicle.destroy()

    # ------------------------------------------------------------------ plots
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(V_log, label='V')
    ax.axhline(desired_speed, color='r', linestyle='--', label='V_des')
    ax.set_xlabel('Step')
    ax.set_ylabel('Speed [m/s]')
    ax.set_title(f'Speed — {tag}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save_plot(log_dir, fig, f'velocity_{tag}')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(X_traj[0], X_traj[1], label='Tracked', linewidth=2)
    ax.plot(X_des[0, ::5], X_des[1, ::5], 'o', markersize=4, label='Reference')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Trajectory — {tag}')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.tight_layout()
    save_plot(log_dir, fig, f'trajectory_{tag}')
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

    return float(np.sqrt(np.mean(error_array ** 2)))
