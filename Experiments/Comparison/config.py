import numpy as np
import torch.nn as nn

config = {
# Model and training
    "samples":500000,
    "batch_size": 64,
    "epochs": 1000,
    "data_path":"Data/Training_Data/training_set.npy",
    "model_path":"Data/model.pth",
    "weight_decay": 5e-4,
    "learning_rate": 0.5e-3,
    "lr_factor": 0.8,
    "lr_patience": 20,
    "scale_V": 50,

# Shared Controller
    "Np": 50,
    "N": 5,
    "M_u_lb":-np.pi/10,
    "M_u_ub":np.pi/10,
    "Q": 10,
    "R": 0,
    "kpV":100,
    "kdV": 2,
    "eps": 0.001,

# Disturbance
    "steering_force": [0.1,0.2,0.3],
    "wind_force": [500, 1000, 2000],
# Plant Model
    "l": 2.5,
    "dt": 0.005,

# Online Learning (replay_buffer and residual_dynamics)
    "buffer_size": 1000,
    "online_lr": 1e-8,
    "online_weight_decay": 1e-5,

# Tube
    "K_tube": [0.0, 0.0, -15.0],

# Simulation
    "sim_T": 10000,
    "ref_steps":100,
    "num_trials":1,
    "steps": 10000,
    "ref_points":1500,
    "seed":26,
    "record": False,
}

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, output_size)
        )
    def forward(self, x):
        return self.net(x)

class ResidualNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, output_size)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

