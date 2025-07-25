from config import config
from config import SimpleNN
import numpy as np
import matplotlib.pyplot as plt
from Shared.funcs import legendre, bike
import torch


def get_Mx(M_u, M_x_des, option='direct', data='X'):
    if option == 'direct':
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
    else:
        input_vector = np.hstack((M_u, M_x_des))
        input_leg = torch.tensor(input_vector, dtype=torch.float32)  # .to('cuda')
        with torch.no_grad():
            X_pred = np.array(model(input_leg))  # .to('cpu'))
        temp_X = X_pred[:N]
        temp_Y = X_pred[N:]

    return temp_X, temp_Y
v_scale = config['scale_V']
l = config['l']
dt = config['dt']
Np = config['Np']
N = config['N']
sys = bike(l,dt)

N_des = config['N_des']
kp = config['kp_tube']
scale_V = config['scale_V']
in_dim = N + 2 * N_des  # X_data.shape[1]
out_dim = 2 * N
leg = legendre(Np*dt,N,dt)
leg_des = legendre(Np * dt, N_des, dt)

model = SimpleNN(in_dim,out_dim)
model.load_state_dict(state_dict=torch.load(config['model_path'], weights_only=True))
model.eval()
model = model.to('cpu')
# model_norm = model_norm.to('cpu')

out_index = 0
num_samples = 1000
mu_samples = np.linspace(-np.pi / 2.5, np.pi / 2.5, num_samples)
M_u = np.zeros(5)
out_data_direct = np.zeros((5, num_samples))
out_data_neural = np.zeros((5, num_samples))
M_x_des = np.zeros(2)
for mu_i in range(5):
    for sample_i in range(num_samples):
        M_u = np.zeros(5)
        M_u[mu_i] = mu_samples[sample_i]
        out_data_direct[mu_i, sample_i] = get_Mx(M_u, M_x_des,'direct')[0][out_index]
        out_data_neural[mu_i, sample_i] = get_Mx(M_u, M_x_des,'npdirect')[0][out_index]

for i in range(5):
    plt.plot(mu_samples, out_data_direct[i, :])
    plt.plot(mu_samples, out_data_neural[i, :])
    plt.legend(["Direct", "Neural"])
    plt.show()