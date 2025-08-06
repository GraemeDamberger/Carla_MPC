from config import config
from config import SimpleNN
import numpy as np
import matplotlib.pyplot as plt
from Shared.funcs import legendre, bike
import torch
def get_Mx(M_u, option='direct', data='X'):
    V = v_scale
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
            # X_pred = np.array(model_norm(input_leg))#.to('cpu'))
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
leg = legendre(Np*dt,N,dt)

model = SimpleNN(N,2*N)
model.load_state_dict(state_dict=torch.load(config['model_path'], weights_only=True))
model.eval()
model = model.to('cpu')
# model_norm = model_norm.to('cpu')

out_index = 0
num_samples = 1000
mu_samples = np.linspace(-np.pi / 2.5, np.pi / 2.5, num_samples)
M_u = np.zeros(N)
out_data_direct = np.zeros((5, num_samples))
out_data_neural = np.zeros((5, num_samples))

for mu_i in range(N):
    for sample_i in range(num_samples):

        M_u = -np.ones(N)
        M_u[mu_i] = mu_samples[sample_i]
        out_data_direct[mu_i, sample_i] = get_Mx(M_u, 'direct')[0][out_index]
        out_data_neural[mu_i, sample_i] = get_Mx(M_u, 'nodirect')[0][out_index]

for i in range(N):
    plt.plot(mu_samples, out_data_direct[i, :])
    plt.plot(mu_samples, out_data_neural[i, :])
    plt.legend(["Direct", "Neural"])
    plt.show()