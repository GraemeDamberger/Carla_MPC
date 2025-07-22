from scipy.optimize import Bounds
from config import config
from config import SimpleNN
from Shared.funcs import bike, legendre, get_mpc_reference, global_to_local, local_to_global
import torch
import numpy as np
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os


#def constraint_decode_specific_points(M_u):
#    # sample_points = np.sort(np.random.uniform(0,int(Np/dt),num_sample_points)).astype(int)
#    return np.matmul(leg.P[sample_points, :], M_u)


def get_Mx(M_u, leg,sys, Np,N ,model_norm,option='direct', data='X'):
    V = 50
    if option == 'direct':
        U = leg.decode(M_u)
        X = np.zeros((Np, 3))
        for i in range(1, Np):
            X[i, :] = sys.dynamics(X[i - 1, :], V, U[i - 1])
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
    temp_X, temp_Y = get_Mx(M_u,leg,sys,Np,N, model_norm, option='nodirect', data='X')

    # X_global = np.zeros(temp_X.shape)
    # Y_global = np.zeros(temp_Y.shape)
    # for i in range(len(temp_X)):
    #    X_global[i],Y_global[i] =  local_to_global(temp_X[i],temp_Y[i],X0[2])

    temp_X = 0.02 * V * temp_X
    # temp_X+=X0[0]
    temp_Y = 0.02 * V * temp_Y
    # temp_Y+=X0[1]
    temp_Ex = temp_X - leg.encode(X_des)
    temp_Ey = temp_Y - leg.encode(Y_des)
    # temp_Ey = leg.encode(X[:,1]-Y_des)
    J_x = np.matmul(temp_Ex, np.matmul(Q, temp_Ex))
    J_y = np.matmul(temp_Ey, np.matmul(Q, temp_Ey))

    cost = J_x + J_y  # +J_theta
    return cost

def run_sim(trial_num):
    def constraint_decode_specific_points(M_u):
        # sample_points = np.sort(np.random.uniform(0,int(Np/dt),num_sample_points)).astype(int)
        return np.matmul(leg.P[sample_points, :], M_u)

    N = config['N']
    L = config['l']
    dt = config['dt']
    Np = config['Np']

    kpV = config['kpV']
    kdV = config['kdV']

    model_norm = SimpleNN(N,2*N)
    model_norm.load_state_dict(state_dict=torch.load(config['model_path'], weights_only=True))
    model_norm.eval()
    model_norm = model_norm.to('cpu')

    sim_T = config['sim_T']
    sim_steps = int(sim_T / dt)
    sim_time = np.linspace(0, sim_T, sim_steps)

    ref_T = sim_T
    ref_steps = config['ref_steps']
    ref_time = np.linspace(0, ref_T + Np * dt, ref_steps)

    Q = config["Q"] * np.eye(Np)
    R = config["R"] * np.eye(Np)
    sys = bike(L,dt)

    options = {
        'eps': 0.001,
        #    'maxiter': 5
    }

    U = np.zeros(Np)

    X0 = np.array([0, 0, 0])
    V = 0
    V_des = 1
    v_mem = [V]
    U_mem = [U[0]]

    X = np.zeros((sim_steps, X0.shape[0]))
    X[0, :] = X0

    U = np.zeros(Np)
    M_u = np.zeros(N)

    error = np.zeros((sim_steps,2))

    X_des = np.transpose(np.array([ref_time, -np.sin(0.2 * np.pi * ref_time)]))
    # X_des = np.zeros((ref_steps,2))
    # X_des[20:,1] = 1
    # X_des[:,0] = ref_time
    s0 = 0
    # U_bounds = Bounds(-np.pi/2.3, np.pi/2.3)
    U_bounds = Bounds(-np.pi / 2.5, np.pi / 2.5)

    num_sample_points = 5
    sample_points = np.linspace(0, Np, num_sample_points).astype(int)
    U_lb = -np.pi / 2.5 * np.ones(num_sample_points)
    U_ub = np.pi / 2.5 * np.ones(num_sample_points)

    U_sampled_constraint = NonlinearConstraint(constraint_decode_specific_points, U_lb, U_ub)

    leg = legendre(Np * dt, N, dt)
    P = leg.P[0:Np, :]
    Q = np.matmul(np.transpose(P), np.matmul(Q, P))
    R = np.matmul(np.transpose(P), np.matmul(R, P))

    for i in range(1, sim_steps):
        # start = time.time()

        #if i % (sim_steps / 10) == 0:
        #    print(f'{100 * i / sim_steps}%')
        # x_mpc_ref, y_mpc_ref = get_mpc_reference(X_des[i:i+Np,0], X_des[i:i+Np,1], V, 0, Np, dt)
        # s0 = s0 + V*dt
        x_mpc_ref, y_mpc_ref = get_mpc_reference(X_des[:, 0], X_des[:, 1], V, s0, Np, dt)
        x_ref_local, y_ref_local = global_to_local(x_mpc_ref, y_mpc_ref, X[i - 1, 0], X[i - 1, 1], X[i - 1, 2])

        # res = minimize(cost_fun, M_u ,method='SLSQP', args = (X[i-1,:],V,x_mpc_ref,y_mpc_ref),bounds = U_bounds)#,options = options) #Run optimization
        res = minimize(cost_fun, M_u, method='SLSQP', args=(X[i - 1, :], V, x_ref_local, y_ref_local, leg,sys, Np, N, model_norm,Q),
                       constraints=U_sampled_constraint, options=options)  # Run optimization
        # res = minimize(cost_fun, M_u ,method='Powell', args = (X[i-1,:],V,x_ref_local,y_ref_local),bounds = U_bounds,options = options) #Run optimization
        # res = minimize(cost_fun, M_u ,method='trust-constr', args = (X[i-1,:],V,x_ref_local,y_ref_local),bounds = U_bounds)#,options = options) #Run optimization

        M_u = res.x
        U = leg.decode(M_u)
        # U = U%np.pi
        X[i, :] = sys.dynamics(X[i - 1, :], V, U[0])
        s0 = s0 + np.sqrt((X[i - 1, 0] - X[i, 0]) ** 2 + (X[i - 1, 1] - X[i, 1]) ** 2)

        U_v = kpV * (V_des - V) + kdV * (0 - (V - v_mem[i - 1]) * dt)
        V = V + U_v * dt
        v_mem.append(V)
        U_mem.append(U[0])
        error[i-1] = np.array([x_mpc_ref[0],y_mpc_ref[0]]) - X[i-1, :2]
    plt.figure()
    plt.plot(X_des[:, 0], X_des[:, 1], '.')
    plt.plot(X[:, 0], X[:, 1])

    plt.legend(['xdes', 'x'])
    plot_save_path = config['tracking_plot_location']
    plt.savefig(os.path.join(plot_save_path, f"tracking_plot_trial{trial_num}.png"))


    return np.sqrt(np.mean(error ** 2))


