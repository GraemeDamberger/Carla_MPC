import numpy as np
import torch.nn as nn

config = {
# Model and training
    "samples":500000,
    "batch_size": 64,
    "epochs": 2000,
    "data_path":"Data/Training_Data/training_set.npy",
    "model_path":"Data/model.pth",
    "weight_decay": 5e-4, #5e-2
    "learning_rate": 0.5e-3, #1e-6
    "scale_V": 5, #50

# Controller
    "Np": 50, #50
    "N": 5,
    "M_u_lb":-np.pi/10,#-np.pi/2.5,
    "M_u_ub":np.pi/10,#np.pi/2.5,
    "Q": 70.6388,#1e1,
    "R": 0,
    "kpV":100,
    "kdV": 2,
    "eps": 0.0007,#0.001,

# Plant
    "l": 1,
    "dt": 0.01, #0.001

# Simulation
    "sim_T": 10,
    "ref_steps":100,
    "num_trials":1,
    "steps": 10000,
    "ref_points":1500,
    "seed":26,
    "record": False,

# Hyperparameter Tuning
    "hyp_opt":False,
    "opt_model_path":"logs/run_2025-08-01_14-51-18/models/model_trial_0.pth",
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

'''
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            #nn.Linear(128, 128),
            #nn.Tanh(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        return self.net(x)
'''