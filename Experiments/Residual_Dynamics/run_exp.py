#import numpy as np

from Experiments.Residual_Dynamics.generate_data import generate_data
from Experiments.Residual_Dynamics.train2 import train
from Experiments.Residual_Dynamics.simulate_carla import simulate_carla
from Experiments.Residual_Dynamics.config import config
from Shared.logging_utils import (
    create_log_dir, save_config, save_git_info, save_metrics,
)
log_dir = create_log_dir()
num_trials = config['num_trials']
rmse = []
for trial in range(1):
    #generate_data(trial,log_dir)
    #train(trial,log_dir)
    rmse.append(simulate_carla(trial,log_dir))
save_config(log_dir, config)
save_git_info(log_dir)
save_metrics(log_dir, rmse)


