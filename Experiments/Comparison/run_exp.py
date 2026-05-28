import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

from Experiments.Comparison.config import config
from Experiments.Comparison.generate_data import generate_data
from Experiments.Comparison.simulate_carla import simulate_carla
from Experiments.Comparison.train import train
from Shared.logging_utils import create_log_dir, save_config, save_git_info, save_metrics, save_plot

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,
                    help='Path to a trained model (.pth) from a previous run. '
                         'Omit to use config[\'model_path\'] (default trained location).')
args = parser.parse_args()
# ------------------ This is for manually putting in model path
args.model = "Experiments/Comparison/logs/run_2026-05-12_12-40-58/models/model_trial_0"

model_path = args.model or config['model_path']




#METHODS = ['normal','replay_buffer', 'residual_dynamics']
METHODS = ['normal', 'tube', 'replay_buffer', 'residual_dynamics','tube_adaptive']
COLORS  = ['steelblue', 'darkorange', 'green', 'crimson', 'mediumpurple']
#COLORS  = ['steelblue', 'darkorange','green']

# 7 trials: 1 base + 3 steering disturbances + 3 wind-force disturbances
trials = [{'name': 'base', 'steering_force': 0.0, 'wind_force': 0.0}]
for sf in config['steering_force']:
    trials.append({'name': f'steer_{sf}', 'steering_force': sf, 'wind_force': 0.0})
for wf in config['wind_force']:
    trials.append({'name': f'wind_{wf}', 'steering_force': 0.0, 'wind_force': float(wf)})

t_start = time.time()

log_dir = create_log_dir(base="Experiments/Comparison/logs")
print(f"Log dir:    {log_dir}")
print(f"Model path: {model_path}")
print(f"Trials: {[t['name'] for t in trials]}")
print(f"Methods: {METHODS}")
print(f"Total simulations: {len(trials) * len(METHODS)}")

if args.model is None:
    print("\n=== Generating training data ===")
    generate_data(0, log_dir)

    print("\n=== Training base model ===")
    train(0, log_dir)
else:
    print("\n=== Skipping data generation and training (using provided model) ===")

# results[trial_name][method] = rmse
results = {t['name']: {} for t in trials}

for trial in trials:
    print(f"\n=== Trial: {trial['name']}  "
          f"(steering_force={trial['steering_force']}, wind_force={trial['wind_force']}) ===")
    for method in METHODS:
        rmse = simulate_carla(
            trial['name'], log_dir,
            method=method,
            steering_force=trial['steering_force'],
            wind_force=trial['wind_force'],
            model_path=model_path,
        )
        results[trial['name']][method] = rmse
        print(f"  {method:<25}: RMSE = {rmse:.4f} m")

# ------------------------------------------------------------------ summary
print("\n=== Summary (RMSE [m]) ===")
col_w = 20
header = f"{'trial':<20}" + "".join(f"{m:>{col_w}}" for m in METHODS)
print(header)
print("-" * len(header))
for t in trials:
    row = f"{t['name']:<20}" + "".join(f"{results[t['name']][m]:>{col_w}.4f}" for m in METHODS)
    print(row)

# ------------------------------------------------------------------ grouped bar chart
trial_names = [t['name'] for t in trials]
x     = np.arange(len(trial_names))
n     = len(METHODS)
width = 0.7 / n

fig, ax = plt.subplots(figsize=(14, 5))
for j, (method, color) in enumerate(zip(METHODS, COLORS)):
    vals   = [results[name][method] for name in trial_names]
    offset = (j - (n - 1) / 2) * width
    ax.bar(x + offset, vals, width, label=method, color=color)

ax.set_xticks(x)
ax.set_xticklabels(trial_names, rotation=20, ha='right')
ax.set_ylabel('RMSE [m]')
ax.set_title('Method Comparison — Tracking RMSE Across Disturbances')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
save_plot(log_dir, fig, 'comparison_rmse')
plt.close(fig)

save_config(log_dir, config)
save_git_info(log_dir)
save_metrics(log_dir, results)

elapsed = time.time() - t_start
h, rem  = divmod(int(elapsed), 3600)
m, s    = divmod(rem, 60)
print(f"\nTotal time: {h:02d}:{m:02d}:{s:02d}")
print(f"All results saved to {log_dir}")
