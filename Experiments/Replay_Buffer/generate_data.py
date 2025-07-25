from Shared.funcs import bike, legendre
import numpy as np
from Experiments.Replay_Buffer.config import config
def generate_data(trial_num,log_dir):
    def get_Mx(M_u,leg,Np,sys,option = 'direct', data = 'X'):
        U = leg.decode(M_u)
        X = np.zeros((Np,3))
        for i in range(1,Np):
            X[i,:] = sys.dynamics(X[i-1,:],scale_V,U[i-1])
        temp_X = leg.encode(X[:,0])
        temp_Y = leg.encode(X[:,1])


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

    sys = bike(l,dt)
    leg = legendre(Np*dt,N,dt)


    #Generate data
    data = []
    for i in range(samples):
        mu = np.random.uniform(M_u_lb,M_u_ub,N) #Randomly sample control trajectories
        x,y = get_Mx(mu, leg,Np,sys, option='direct')
        data.append(np.hstack((mu, x, y)))

    #Save data for training
    data = np.array(data)
    save_path = config['data_path']
    np.save(save_path, data)
