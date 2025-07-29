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



def simulate_carla(trial_num,log_dir):
    def get_Mx(M_u, leg,sys, Np,N ,model_norm,option='direct', data='X'):

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
        temp_X, temp_Y = get_Mx(M_u,leg,sys,Np,N, model_norm, option='nondirect', data='X')

        # X_global = np.zeros(temp_X.shape)
        # Y_global = np.zeros(temp_Y.shape)
        # for i in range(len(temp_X)):
        #    X_global[i],Y_global[i] =  local_to_global(temp_X[i],temp_Y[i],X0[2])

        temp_X =  1 / scale_V * V * temp_X
        # temp_X+=X0[0]
        temp_Y =  1 / scale_V * V * temp_Y
        # temp_Y+=X0[1]
        temp_Ex = temp_X - leg.encode(X_des)
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


    N = config['N']
    L = config['l']
    dt = config['dt']
    Np = config['Np']
    scale_V = config['scale_V']
    kpV = config['kpV']
    kdV = config['kdV']

    Q = config["Q"] * np.eye(Np)
    R = config["R"] * np.eye(Np)

    sys = bike(L,dt)

    Steps = config['steps']

    model_norm = SimpleNN(N,2*N)
    model_norm.load_state_dict(state_dict=torch.load(config['model_path'], weights_only=True))
    model_norm.eval()
    model_norm = model_norm.to('cpu')

    options = {
            'eps': 0.001, #0.001
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
    world.apply_settings(settings)

    world.tick()

    vehicle_bp = world.get_blueprint_library().filter('*vehicle*')
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp[3], spawn_point)
    time.sleep(1)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5))

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    #camera.listen(lambda image: image.save_to_disk('images/%06d.png' % image.frame))

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
            if i % (Steps / 100) == 0:
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
            x_ref_local, y_ref_local = global_to_local(x_mpc_ref, y_mpc_ref, X[0,i - 1], X[1,i - 1], X[2,i - 1])
            res = minimize(cost_fun, M_u, method='SLSQP', args=(X[:,i - 1], current_speed, x_ref_local, y_ref_local, leg,sys, Np, N, model_norm,Q),
                           constraints=U_sampled_constraint, options=options)  # Run optimization
            M_u = res.x
            U = leg.decode(M_u)
            U_mem.append(U[0])
            U_steer = U[0]

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

            X_curr = vehicle.get_location()
            world.debug.draw_point(X_curr, size=0.1, color=carla.Color(r=0, g=0, b=255), life_time=500.0)

            theta = np.deg2rad(vehicle.get_transform().rotation.yaw)
            X[:, i] = np.array([X_curr.x, X_curr.y, theta])
            error_array[i - 1,:] = np.array([x_mpc_ref[0], y_mpc_ref[0]]) - X[:2,i - 1]
    finally:
        if (vehicle is not None):
            if camera is not None:
                camera.stop()
                camera.destroy()
        vehicle.destroy()

    #plot_save_path = config['tracking_plot_location']
    fig = plt.figure()
    plt.plot(V)
    plt.plot(desired_speed * np.ones(len(V)))
    plt.legend(['V', 'V_des'])
    save_plot(log_dir,fig,f"velocity_plot_trial_{trial_num}")
    #plt.savefig(os.path.join(plot_save_path, f"velocity_plot_trial{trial_num}.png"))
    #plt.show()

    fig = plt.figure()
    plt.plot(X[0, :], X[1, :])
    plt.plot(X_des[0, ::5], X_des[1, ::5], '.')
    plt.plot(x_mpc_ref, y_mpc_ref, '.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['XY', 'XY des'])
    #plt.savefig(os.path.join(plot_save_path, f"tracking_plot_trial{trial_num}.png"))
    save_plot(log_dir,fig,f"tracking_plot_trial_{trial_num}")
    #plt.show()

    fig = plt.figure()
    plt.plot(U_mem)
    plt.title('Control')
    save_plot(log_dir,fig,f"control_plot_trial_{trial_num}")
    #plt.savefig(os.path.join(plot_save_path, f"control_plot_trial{trial_num}.png"))
    #plt.show()




    return np.sqrt(np.mean(error ** 2))
