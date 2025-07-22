import numpy as np

#import generate_data
#import train
#import simulate
from simulate import run_sim
from config import config

num_trials = config['num_trials']
rmse_save_loc = config['rmse_data_location']
rmse = []
for trial in range(num_trials):
    rmse.append(run_sim(trial))
np.savetxt(rmse_save_loc,np.array(rmse))