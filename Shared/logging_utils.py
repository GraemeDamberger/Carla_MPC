import os
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
import platform
import torch
import matplotlib.pyplot as plt


def create_log_dir(base="logs"):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path(base) / f"run_{timestamp}"
    (log_dir / "plots").mkdir(parents=True, exist_ok=True)
    (log_dir / "models").mkdir(parents=True, exist_ok=True)
    return log_dir


def save_config(log_dir, config_dict):
    config_path = Path(log_dir) / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)


def save_metrics(log_dir, metrics_dict):
    metrics_path = Path(log_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)


def save_git_info(log_dir):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        commit = "Not a git repository or git not installed"
    git_info = {
        "git_commit": commit,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__ if torch else "n/a"
    }
    with open(Path(log_dir) / "environment.json", 'w') as f:
        json.dump(git_info, f, indent=2)


def save_plot(log_dir, fig, name="plot.png"):
    fig_path = Path(log_dir) / "plots" / name
    fig.savefig(fig_path)


def save_model(log_dir, model, name="model.pth"):
    model_path = Path(log_dir) / "models" / name
    torch.save(model.state_dict(), model_path)
