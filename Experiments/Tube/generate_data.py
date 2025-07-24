from Shared.funcs import bike, legendre
import numpy as np
from config import config
def generate_data(trial_num,log_dir):
    def get_Mx(M_u,M_x_des,leg,Np,sys,option = 'direct', data = 'X'):

        U = leg.decode(M_u)
        X = np.zeros((Np, 3))
        X_des = leg_des.decode(M_x_des[:N_des])
        Y_des = leg_des.decode(M_x_des[N_des:])
        for i in range(1, Np):
            theta_des = np.arctan2((Y_des[i - 1] - X[i - 1, 1]), (X_des[i - 1] - X[i - 1, 0]))
            U_p = kp * (theta_des - X[i - 1, 2])
            X[i, :] = sys.dynamics(X[i - 1, :], scale_V, U[i - 1] + U_p)
        temp_X = leg.encode(X[:, 0])
        temp_Y = leg.encode(X[:, 1])


        return temp_X,temp_Y

    # Configure parameters
    samples = config['samples']
    M_u_lb = config['M_u_lb']
    M_u_ub = config['M_u_ub']
    dt = config['dt']
    Np = config['Np']
    scale_V = config['scale_V']
    l = config['l']
    N = config['N']
    N_des = config['N_des']
    kp = config['kp_tube']
    sys = bike(l,dt)
    leg = legendre(Np*dt,N,dt)
    leg_des = legendre(Np * dt, N_des, dt)

    M_theta_des_lb = -2
    M_theta_des_ub = 2

    #Generate data
    data = []
    for i in range(samples):
        mu = np.random.uniform(M_u_lb,M_u_ub,N) #Randomly sample control trajectories
        xthetades = np.random.uniform(M_u_lb, M_u_ub, 2 * N_des)
        x,y = get_Mx(mu,xthetades ,leg,Np,sys, option='direct')
        data.append(np.hstack((mu,xthetades, x, y)))

    #Save data for training
    data = np.array(data)
    save_path = config['data_path']
    np.save(save_path, data)
