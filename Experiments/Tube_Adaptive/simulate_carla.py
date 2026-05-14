import carla
from config import config
from config import SimpleNN
from Shared.funcs import bike, legendre, get_mpc_reference, global_to_local, local_to_global
import torch
import numpy as np
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from Shared.logging_utils import save_plot
from pathlib import Path
import cv2



def simulate_carla(trial_num,log_dir):
    def get_Mx(M_u, leg,sys, Np,N ,model_norm,option='nodirect', data='X'):

        if option == 'direct':
            U = leg.decode(M_u)
            X = np.zeros((Np, 3))
            for i in range(1, Np):
                X[i, :] = sys.dynamics(X[i - 1, :], scale_V, U[i - 1])
            temp_X = leg.encode(X[:, 0])
            temp_Y = leg.encode(X[:, 1])
        else:
            input_vector = M_u
            input_leg = torch.tensor(input_vector, dtype=torch.float32)  # .to('cuda')
            with torch.no_grad():
                X_pred = np.array(model_norm(input_leg))  # .to('cpu'))
            temp_X = X_pred[:N]
            temp_Y = X_pred[N:]
        return temp_X, temp_Y


    def cost_fun(M_u, X0, V, X_des, Y_des, leg,sys, Np, N, model_norm, Q):
        temp_X, temp_Y = get_Mx(M_u,leg,sys,Np,N, model_norm, option='direct', data='X')

        # X_global = np.zeros(temp_X.shape)
        # Y_global = np.zeros(temp_Y.shape)
        # for i in range(len(temp_X)):
        #    X_global[i],Y_global[i] =  local_to_global(temp_X[i],temp_Y[i],X0[2])

        temp_X =  1 / scale_V * V * temp_X
        # temp_X+=X0[0]
        temp_Y =  1 / scale_V * V * temp_Y
        # temp_Y+=X0[1]
        temp_Ex = temp_X - leg.encode(X_des) #Do this outside the optimization loop for quicker results
        temp_Ey = temp_Y - leg.encode(Y_des)
        # temp_Ey = leg.encode(X[:,1]-Y_des)
        J_x = np.matmul(temp_Ex, np.matmul(Q, temp_Ex))
        J_y = np.matmul(temp_Ey, np.matmul(Q, temp_Ey))

        cost = J_x + J_y  # +J_theta
        return cost


    def get_waypoints_from_vehicle(vehicle, distance=2.0, num_points=50):
        world = vehicle.get_world()
        map = world.get_map()

        transform = vehicle.get_transform()
        current_wp = map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        waypoints = []
        for _ in range(num_points):
            waypoints.append(current_wp.transform.location)
            next_wps = current_wp.next(distance)
            if not next_wps:
                break
            current_wp = next_wps[0]

        return waypoints

    def constraint_decode_specific_points(M_u):
        #sample_points = np.sort(np.random.uniform(0,int(Np/dt),num_sample_points)).astype(int)
        return np.matmul(leg.P[sample_points,:],M_u)

    def process_image(image):
        # Convert CARLA raw image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # drop alpha
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        out.write(array)  # write frame to video
    def feedback_mix(M_u, model_norm, leg, X_measure, Y_measure, alpha):
        input_leg = torch.tensor(M_u, dtype=torch.float32)
        MX_pred = np.array(model_norm(input_leg))
        MX_pred = 1 / scale_V * V * MX_pred
        model_pred = leg.decode(MX_pred)
        X_model = model_pred[:N]
        Y_model = model_pred[N:]
        X_mix = X_model * alpha + X_measure * (1 - alpha)
        Y_mix = Y_model * alpha + Y_measure * (1 - alpha)
        return X_mix, Y_mix

    import carla

    import carla

    def draw_trajectory(world, X, Y, color=carla.Color(0, 255, 0), thickness=0.1, lifetime=0.1, z_offset=0.2):
        """
        Draws a temporary trajectory in CARLA given X and Y coordinates.

        Parameters:
        - world: carla.World object
        - X, Y: arrays or lists of same length
        - color: carla.Color
        - thickness: line thickness
        - lifetime: seconds the line stays visible
        - z_offset: height above ground for visibility
        """
        n = len(X)
        if n < 2:
            return

        for i in range(n - 1):
            start = carla.Location(x=X[i], y=Y[i], z=z_offset)
            end = carla.Location(x=X[i + 1], y=Y[i + 1], z=z_offset)
            world.debug.draw_line(start, end, thickness=thickness, color=color, life_time=lifetime)
        # Example usage in your simulation loop
        # X = np.linspace(0, 10, 50)
        # Y = np.sin(X)
        # draw_trajectory_xy(world, X, Y, color=carla.Color(255,0,0), lifetime=0.1)
    def tube_control(sys, X_prev, X_current, V, U_nom, K, adaptive):
        X_hat = sys.dynamics(X_prev, V, U_nom)
        e = X_current - X_hat
        U_adapt = float(adaptive.forward_control(X_current[2]))
        U_tube = float(K @ e)
        adaptive.update(X_current[2], -e[2], dt)
        return U_tube + U_adapt

    class AdaptiveRBFController:
        def __init__(
                self,
                state_dim,
                control_dim,
                num_basis=30,
                gamma=500.0,
                sigma=1.0,
                weight_clip=20.0,
                seed=0
        ):
            """
            state_dim   : dimension of x
            control_dim : dimension of u
            num_basis   : number of RBF neurons
            gamma       : adaptation gain
            sigma       : RBF width
            """

            rng = np.random.default_rng(seed)

            self.nx = state_dim
            self.nu = control_dim
            self.nb = num_basis

            self.gamma = gamma
            self.sigma = sigma
            self.weight_clip = weight_clip

            # RBF centers in normalized state-space
            self.centers = rng.uniform(-np.pi, np.pi, size=(num_basis, state_dim))

            # Decoder weights:
            # each basis contributes to each control input
            # shape = (num_basis, control_dim)
            self.W = np.zeros((num_basis, control_dim))

        # --------------------------------------------------------
        # RADIAL BASIS FUNCTIONS
        # --------------------------------------------------------
        def phi(self, x):
            """
            x shape: (nx,)
            returns phi shape: (nb,)
            """
#            diff = self.centers - x[None, :]
            diff = self.centers - x
            sq_norm = np.sum(diff ** 2, axis=1)

            phi = np.exp(-sq_norm / (2 * self.sigma ** 2))

            # optional normalize
            s = np.sum(phi) + 1e-8
            phi /= s

            return phi

        # --------------------------------------------------------
        # COMPUTE ADAPTIVE CONTROL TERM
        # --------------------------------------------------------
        def forward_control(self, x):
            """
            u_adapt = W^T phi
            """
            basis = self.phi(x)  # (nb,)
            u = self.W.T @ basis  # (nu,)
            return u

        # --------------------------------------------------------
        # ONLINE LEARNING
        # --------------------------------------------------------
        def update(self, x, e, dt):
            """
            x : actual state
            e : tube error = x - z
            dt: timestep

            W_dot = gamma * phi(x) * e_u^T

            If control_dim != state_dim, use first nu components of error
            or map externally.
            """

            basis = self.phi(x)  # (nb,)

            # use first nu dimensions of error
#            err = e[:self.nu]
            err = e

            #print(x)
            # outer product => (nb, nu)
            dW = self.gamma * np.outer(basis, err)

            self.W += dt * dW

            # projection / clipping for robustness
            self.W = np.clip(
                self.W,
                -self.weight_clip,
                self.weight_clip
            )
    N = config['N']
    L = config['l']
    dt = config['dt']
    Np = config['Np']
    scale_V = config['scale_V']
    kpV = config['kpV']
    kdV = config['kdV']
    alpha = config['alpha']
    Q = config["Q"] * np.eye(Np)
    R = config["R"] * np.eye(Np)
    sys = bike(L,dt)

    K_tube = np.array(config['K_tube'])
    Steps = config['steps']

    model_norm = SimpleNN(N,2*N)
    if config['hyp_opt'] == True:
        model_norm.load_state_dict(state_dict=torch.load(config['opt_model_path'], weights_only=True))
    else:
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]  # Carla_MPC/
        #model_path = project_root / "Experiments" / "Normal" / "logs" / "run_2025-08-01_14-51-18" / "models" / "model_trial_0"
        model_path = project_root / "Experiments" / "Normal" / "logs" / "run_2026-03-19_11-00-55" / "models" / "model_trial_0"

        model_norm.load_state_dict(state_dict=torch.load(model_path, weights_only=True))
        #model_norm.load_state_dict(state_dict=torch.load(config['model_path'], weights_only=True))
    model_norm.eval()
    model_norm = model_norm.to('cpu')

    options = {
            'eps': config["eps"], #0.001
            #    'maxiter': 5
        }
    error_array = np.zeros((Steps, 2))
    # setup carla
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    settings.random_seed = config["seed"]
    world.apply_settings(settings)

    world.tick()

    vehicle_bp = world.get_blueprint_library().filter('*vehicle*')
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp[3], spawn_point)
    time.sleep(1)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5))

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    adaptive = AdaptiveRBFController(
        state_dim=1,
        control_dim=1,
        num_basis=50,
        gamma=80.0,
        sigma=0.7
    )
    if config['record'] == True:
        frame_width = 800  # set your camera width
        frame_height = 600  # set your camera height
        fps = 1.0 / dt  # match your control loop timestep
        video_dir = Path(log_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"simulation_{trial_num}.mp4"
        out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        camera.listen(process_image)
        #camera.listen(lambda image: image.save_to_disk('images/%06d.png' % image.frame))

    spectator = world.get_spectator()

    # Function to keep spectator in third-person view
    def update_spectator():
        transform = vehicle.get_transform()
        # Position camera 8 meters behind and 3 meters above the car
        spectator_location = transform.location - transform.get_forward_vector() * 8
        spectator_location.z += 3
        spectator_transform = carla.Transform(spectator_location, transform.rotation)
        spectator.set_transform(spectator_transform)


    world.tick()
    ref_points = config["ref_points"]  # Made big, for full sim
    waypoints = get_waypoints_from_vehicle(vehicle, num_points=ref_points)
    for wp in waypoints:
        world.debug.draw_point(wp, size=0.1, color=carla.Color(r=255, g=0, b=0), life_time=500.0)
    X_des = np.zeros((2, ref_points))
    i = 0
    for wp in waypoints:
        X_des[:, i] = np.array([wp.x, wp.y])
        i += 1

    V = []
    X = np.zeros((3, Steps))
    X_curr = vehicle.get_location()
    # theta = np.deg2rad(vehicle.get_transform().rotation.yaw)-np.pi
    theta = np.deg2rad(vehicle.get_transform().rotation.yaw)
    X[:, 0] = np.array([X_curr.x, X_curr.y, theta])

    num_sample_points = 5
    sample_points = np.linspace(0,Np,num_sample_points).astype(int)
    U_lb = -np.pi/2.5 * np.ones(num_sample_points)
    U_ub = np.pi/2.5 * np.ones(num_sample_points)

    U_sampled_constraint = NonlinearConstraint(constraint_decode_specific_points,U_lb,U_ub)

    s0 = 0
    error_i = 0
    U_steer = np.zeros(Np)
    U_mem = [0]

    M_u = np.zeros(N)
    U_prev_nom = 0.0
    leg = legendre(Np * dt, N, dt)
    P = leg.P[0:Np, :]
    Q = np.matmul(np.transpose(P), np.matmul(Q, P))
    R = np.matmul(np.transpose(P), np.matmul(R, P))

    kp = 0.5
    kd = 0.1
    ki = 0.2

    desired_speed = 15.0
    previous_speed = 0.0

    try:
        for i in range(1, Steps):
            if i % (Steps / 5) == 0:
                print(f'{100 * i / Steps}%')
            # Speed control stuff
            velocity = vehicle.get_velocity()
            current_speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])  # m/s

            acceleration = (current_speed - previous_speed) / 0.05  # assuming 20 Hz
            previous_speed = current_speed
            error_i += (desired_speed - current_speed) * dt
            error = desired_speed - current_speed
            throttle_cmd = kp * error - kd * acceleration + ki * error_i
            throttle = np.clip(throttle_cmd, 0.0, 1.0)

            # MPC
            # s0 = s0 + np.sqrt( (X[0,i-1] - X[0,i])**2 + (X[0,i-1] - X[0,i])**2 )
            s0 = s0 + current_speed * dt
            x_mpc_ref, y_mpc_ref = get_mpc_reference(X_des[0, :], X_des[1, :], current_speed, s0, Np, dt)
            temp_x = leg.decode(leg.encode(x_mpc_ref))
            temp_Y = leg.decode(leg.encode(x_mpc_ref))
            draw_trajectory(world, x_mpc_ref, y_mpc_ref, color=carla.Color(0, 255, 0), lifetime=0.1)

            x_ref_local, y_ref_local = global_to_local(x_mpc_ref, y_mpc_ref, X[0,i - 1], X[1,i - 1], X[2,i - 1])
            res = minimize(cost_fun, M_u, method='SLSQP', args=(X[:,i - 1], current_speed, x_ref_local, y_ref_local, leg,sys, Np, N, model_norm,Q),
                           constraints=U_sampled_constraint, options=options)  # Run optimization
            M_u = res.x
            U = leg.decode(M_u)
            U_mem.append(U[0])
            U_mpc = U[0]

            if i >= 2:
                U_tube = tube_control(sys, X[:, i-2], X[:, i-1], current_speed, U_prev_nom, K_tube, adaptive)
            else:
                U_tube = 0.0
            U_prev_nom = U_mpc

            U_steer = U_mpc + U_tube

            brake = 0.0
            if current_speed - desired_speed > 2.0:  # Do i need this??
                brake = 0.3
                throttle = 0.0

            # Apply control
            control = carla.VehicleControl()
            control.throttle = throttle
            control.brake = brake
            control.steer = U_steer #res.x[0]
            vehicle.apply_control(control)
            V.append(current_speed)

            world.tick()
            update_spectator()
            X_curr = vehicle.get_location()
            world.debug.draw_point(X_curr, size=0.1, color=carla.Color(r=0, g=0, b=255), life_time=500.0)

            theta = np.deg2rad(vehicle.get_transform().rotation.yaw)
            #X_mix, Y_mix = feedback_mix(M_u, model_norm, leg, X_curr.x, X_curr.y, alpha)
            X[:, i] = np.array([X_curr.x, X_curr.y, theta])
            error_array[i - 1,:] = np.array([x_mpc_ref[0], y_mpc_ref[0]]) - X[:2,i - 1]

    finally:
        W = adaptive.W
        print("||W||_F =", np.linalg.norm(W))
        print("max|W| =", np.max(np.abs(W)))
        print("mean|W| =", np.mean(np.abs(W)))
        if (vehicle is not None):
            if camera is not None:
                camera.stop()
                camera.destroy()
                if config['record'] == True:
                    out.release()
        vehicle.destroy()
    if config['hyp_opt'] != True:
        #plot_save_path = config['tracking_plot_location']
        fig = plt.figure()
        plt.plot(V)
        plt.plot(desired_speed * np.ones(len(V)))
        plt.legend(['V', 'V_des'])
        save_plot(log_dir,fig,f"velocity_plot_trial_{trial_num}")
        #plt.savefig(os.path.join(plot_save_path, f"velocity_plot_trial{trial_num}.png"))
        #plt.show()

    # --- Trajectory plot ---
    fig_traj, ax_traj = plt.subplots(figsize=(8, 6))
    ax_traj.plot(X[0, :], X[1, :], label='Tracked Path', color='blue', linewidth=2)
    ax_traj.plot(X_des[0, ::5], X_des[1, ::5], 'o', label='Desired Path (sparse)', color='orange', markersize=6)
    #ax_traj.plot(x_mpc_ref, y_mpc_ref, '.', label='MPC Reference', color='green', markersize=6)

    ax_traj.set_xlabel('X [m]', fontsize=12)
    ax_traj.set_ylabel('Y [m]', fontsize=12)
    ax_traj.set_title(f'Trajectory Tracking Trial {trial_num}', fontsize=14)
    ax_traj.grid(True, linestyle='--', alpha=0.5)
    ax_traj.legend(fontsize=10)
    ax_traj.axis('equal')
    plt.tight_layout()

    # Save trajectory plot
    save_plot(log_dir, fig_traj, f"tracking_plot_trial_{trial_num}")
    plt.close(fig_traj)

    # --- Control input plot (only if hyp_opt != True) ---
    if not config.get('hyp_opt', False):
        fig_ctrl, ax_ctrl = plt.subplots(figsize=(8, 4))
        ax_ctrl.plot(U_mem)  # assumes shape (n_inputs, timesteps)
        ax_ctrl.set_title(f'Control Inputs Trial {trial_num}', fontsize=14)
        ax_ctrl.set_xlabel('Time step', fontsize=12)
        ax_ctrl.set_ylabel('Control Value', fontsize=12)
        ax_ctrl.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Save control plot
        save_plot(log_dir, fig_ctrl, f"control_plot_trial_{trial_num}")
        plt.close(fig_ctrl)

    '''
    fig = plt.figure()
    plt.plot(X[0, :], X[1, :])
    plt.plot(X_des[0, ::5], X_des[1, ::5], '.')
    plt.plot(x_mpc_ref, y_mpc_ref, '.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['Actual Path', 'Reference Path'])
    #plt.savefig(os.path.join(plot_save_path, f"tracking_plot_trial{trial_num}.png"))
    save_plot(log_dir,fig,f"tracking_plot_trial_{trial_num}")
    #plt.show()
    if config['hyp_opt'] != True:
        fig = plt.figure()
        plt.plot(U_mem)
        plt.title('Control')
        save_plot(log_dir,fig,f"control_plot_trial_{trial_num}")
        #plt.savefig(os.path.join(plot_save_path, f"control_plot_trial{trial_num}.png"))
        #plt.show()
    '''



    return np.sqrt(np.mean(error_array ** 2))
