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
    "wind_force": [5000, 10000, 15000],
# Plant Model
    "l": 2.5,
    "dt": 0.005,

# Online Learning (replay_buffer and residual_dynamics)
    "buffer_size": 139,#1000,
    "online_lr_replay": 4.5463954951931305e-07,#4e-7,
    "online_lr_residual": 4.5463954951931305e-07,#2e-7,
    "online_weight_decay": 0.0004272274816226789,#1e-5,

# Tube
    "K_tube": [0,0,-44.96820062851194],#[0.0, 0.0, -15.0],

# Tube Adaptive
    "K_tube_adaptive": [0,0, -39.61905791892411],#[0.0, 0.0, -15.0],
    "rbf_num_basis": 50,
    "rbf_gamma": 26.30158762851422,#80.0,
    "rbf_sigma": 0.254658253138533,#0.7,
    "rbf_weight_clip": 20.0,

# Simulation
    "sim_T": 10000,
    "ref_steps":100,
    "num_trials":1,
    "steps": 10000,
    "ref_points":1500,
    "seed":26,
    "record": False,
    "no_rendering_mode": False,
    "save_plots": True,   # per-rollout diagnostic plots; tuning sets this False
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

