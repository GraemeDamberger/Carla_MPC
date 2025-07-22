from Shared.funcs import bike, legendre
import numpy as np
import os
from config import config

def get_Mx(M_u,leg,Np,sys,option = 'direct', data = 'X'):
    V = 50
    U = leg.decode(M_u)
    X = np.zeros((Np,3))
    for i in range(1,Np):
        X[i,:] = sys.dynamics(X[i-1,:],V,U[i-1])
    temp_X = leg.encode(X[:,0])
    temp_Y = leg.encode(X[:,1])
    
        
    return temp_X,temp_Y


samples = config['samples']
M_u_lb = config['M_u_lb']
M_u_ub = config['M_u_ub']
dt = config['dt']
Np = config['Np']

l = config['l']
N = config['N']

sys = bike(l,dt)
leg = legendre(Np*dt,N,dt)


# Step 1: Generate data
#M_u = np.linspace(-np.pi/2.5, np.pi/2.5, 250)
#Theta, M_u = np.meshgrid(theta_vals, M_u_vals)
data = []
for i in range(samples):
    mu = np.random.uniform(M_u_lb,M_u_ub,N)
    x,y = get_Mx(mu, leg,Np,sys, option='direct')
    data.append(np.hstack((mu, x, y)))

data = np.array(data)
print(data.shape)
save_path = config['data_path']
np.save(save_path, data)
#np.save("my_array.npy", data)
