import numpy as np
import torch.nn as nn

config = {
# Model and training
    "samples": 100000, #100000
    "batch_size": 64,
    "epochs": 500, #500
    "data_path":"Data/Training_Data/training_set.npy",
    "model_path":"Data/model.pth",
    "weight_decay": 5e-4, #5e-2
    "learning_rate": 0.5e-4, #1e-6
    "scale_V": 50, #50
    "buffer_size": 1000,
    "online_lr": 1e-9, #1e-8
    "online_weight_decay": 1e-2, #1e-2

# Controller
    "Np": 50, #50
    "N": 5,
    "M_u_lb":-np.pi/10,
    "M_u_ub":np.pi/10,
    "Q": 1e1,
    "R": 0,
    "kpV":100,
    "kdV": 2,

# Plant
    "l": 1,
    "dt": 0.005, #0.001

# Simulation
    "sim_T": 1000,
    "ref_steps":100,
    "num_trials":1,
    "steps": 10000,
    "ref_points":1500
}
'''
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, output_size)
        )

    def forward(self, x):
        return self.net(x)
'''
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