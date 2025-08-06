import numpy as np

from generate_data import generate_data
from train3 import train
from simulate_carla import simulate_carla
from config import config
from Shared.logging_utils import (
    create_log_dir, save_config, save_git_info, save_metrics,
)
from hyp_opt import hyp_opt

test_switch = 0

if test_switch == 0:
    log_dir = create_log_dir()
    num_trials = config['num_trials']
    rmse = []
    for trial in range(5):
        generate_data(trial,log_dir)
        train(trial,log_dir)
        rmse.append(simulate_carla(trial,log_dir))
    save_config(log_dir, config)
    save_git_info(log_dir)
    save_metrics(log_dir, rmse)
else:
    # int, float, categorical,  loguniform
    #model_path = "Experiments/Normal/logs/run_2025-07-31_17-00-43/models/model_trial_0.pth"
    model_path = "logs/run_2025-08-01_14-51-18/models/model_trial_0"
    param_space = {
        "Q": ("uniform", 1e0, 1e3),
        'eps': ("uniform", 1e-4, 1e-1),
    }
    study = hyp_opt( param_space, model_path, n_trials = 100, steps = 1000)
    print(study)