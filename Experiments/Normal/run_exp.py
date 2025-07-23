import numpy as np

from generate_data import generate_data
from train import train
from simulate import simulate
from simulate_carla import simulate_carla
from config import config

num_trials = config['num_trials']
rmse_save_loc = config['rmse_data_location']
rmse = []
for trial in range(1):
    generate_data(trial)
    train(trial)
    rmse.append(simulate_carla(trial))
np.savetxt(rmse_save_loc,np.array(rmse))