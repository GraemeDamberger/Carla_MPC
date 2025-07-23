import numpy as np
import torch.nn as nn

config = {
# Model and training
    "samples": 100000,
    "batch_size": 32,
    "epochs": 1000,
    "data_path":"Data/Training_Data/training_set.npy",
    "model_path":"Data/model.pth",
    "train_plot_path":"Data/Training_Data/",
    "weight_decay": 5e-2,
    "learning_rate": 1e-6,

# Controller
    "Np": 50,
    "N": 5,
    "M_u_lb":-np.pi/2.5,
    "M_u_ub":np.pi/2.5,
    "Q": 1e1,
    "R": 0,
    "kpV":100,
    "kdV": 2,

# Plant
    "l": 1,
    "dt": 0.005,

# Simulation
    "sim_T": 10,
    "ref_steps":100,
    "tracking_plot_location":"Data/Sim_Data/",
    "num_trials":10,
    "rmse_data_location":"Data/Sim_Data/rmse_data.text",
    "steps": 20000,
    "ref_points":1500
}

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