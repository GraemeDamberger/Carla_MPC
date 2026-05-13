from Shared.funcs import bike, legendre
import numpy as np
from config import config
def generate_data(trial_num,log_dir):
    def get_Mx(M_u,leg,Np,sys,option = 'direct', data = 'X'):
        U = leg.decode(M_u)
        X = np.zeros((Np,3))
        for i in range(1,Np):
            X[i,:] = sys.dynamics(X[i-1,:],scale_V,U[i-1])
        temp_X = leg.encode(X[:,0])
        temp_Y = leg.encode(X[:,1])


        return temp_X,temp_Y
    def constraint_decode_specific_points(M_u):
        return np.matmul(leg.P[sample_points,:],M_u)

    # Configure parameters
    samples = config['samples']
    M_u_lb = config['M_u_lb']
    M_u_ub = config['M_u_ub']
    dt = config['dt']
    Np = config['Np']
    scale_V = config['scale_V']
    l = config['l']
    N = config['N']


    num_sample_points = 10
    sample_points = np.linspace(0, Np, num_sample_points).astype(int)
    U_lb = -1 * np.ones(num_sample_points)
    U_ub = 1 * np.ones(num_sample_points)
    sys = bike(l,dt)
    leg = legendre(Np*dt,N,dt)


    #Generate data
    data = []
    for i in range(samples):

        while(True):
            mu = np.random.uniform(M_u_lb,M_u_ub,N) #Randomly sample control trajectories
            U_temp = constraint_decode_specific_points(mu)
            if np.all((U_temp >= U_lb) & (U_temp <= U_ub)):
                break
        #mu = np.random.uniform(M_u_lb,M_u_ub,N) #Randomly sample control trajectories
        x,y = get_Mx(mu, leg,Np,sys, option='direct')
        data.append(np.hstack((mu, x, y)))

    #Save data for training
    data = np.array(data)
    save_path = config['data_path']
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data)
