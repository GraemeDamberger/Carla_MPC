import optuna
import config
import copy
import simulate_carla
import importlib
from Shared.logging_utils import (
    create_log_dir, save_config
)

def hyp_opt(param_space:dict, model_path, n_trials = 50, steps = 5000):
    log_dir = create_log_dir(base="logs/optuna_trials")
    print(log_dir)
    original_config = copy.deepcopy(config.config)

    def objective(trial):
        config.config["hyp_opt"] = True
        config.config["opt_model_path"] = model_path
        config.config["steps"] = steps
        for name, (ptype, *params) in param_space.items():
            if ptype == "loguniform":
                config.config[name] = trial.suggest_float(name, *params, log=True)
            elif ptype == "uniform":
                config.config[name] = trial.suggest_float(name, *params)
            elif ptype == "int":
                config.config[name] = trial.suggest_int(name, *params)
            elif ptype == "categorical":
                config.config[name] = trial.suggest_categorical(name, params[0])
            else:
                raise ValueError(f"Unsupported type '{ptype}' for parameter '{name}'")
        save_config(log_dir, config.config)
        print(f"🔍 Trial {trial.number} parameters: {trial.params}")
        try:
            # Run your simulation
            score = simulate_carla.simulate_carla(trial.number, log_dir)
            return score
        finally:
            config.config = copy.deepcopy(original_config)
            importlib.reload(simulate_carla)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("✅ Best Parameters:", study.best_trial.params)
    return study