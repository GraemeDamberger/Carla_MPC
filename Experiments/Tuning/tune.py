"""
Optuna hyperparameter tuning for the online MPC methods:
    replay_buffer | residual_dynamics | tube

The offline base network is never retrained — pass its path via --model.

Tuning uses short 3 000-step runs, averaged across multiple seeds to smooth
out CARLA's physics noise and the replay-buffer's sampling randomness.

Usage (from project root):
    python -m Experiments.Tuning.tune --method replay_buffer
    python -m Experiments.Tuning.tune --method residual_dynamics --n_trials 60 --n_seeds 3
    python -m Experiments.Tuning.tune --method tube --n_trials 40

Resume a crashed study:
    python -m Experiments.Tuning.tune --method replay_buffer \\
        --resume Experiments/Tuning/logs/study_replay_buffer_2026-...

Validate best params at full 10 000 steps on all disturbance scenarios:
    python -m Experiments.Tuning.tune --method replay_buffer --validate \\
        --params_file Experiments/Tuning/logs/study_.../best_params.json
"""

import argparse
import json
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import torch

from Experiments.Comparison.config import config
from Experiments.Comparison.simulate_carla import simulate_carla

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = (
    "Experiments/Comparison/logs/run_2026-05-12_12-40-58/models/model_trial_0"
)
TUNE_STEPS = 10_000
FULL_STEPS = 10_000
BASE_SEED  = 26

# All scenarios used during final validation (mirrors Comparison/run_exp.py)
ALL_SCENARIOS = (
    [{"name": "base",         "steering_force": 0.0,  "wind_force": 0.0}]
    + [{"name": f"steer_{sf}", "steering_force": sf,   "wind_force": 0.0}
       for sf in config["steering_force"]]
    + [{"name": f"wind_{wf}",  "steering_force": 0.0,  "wind_force": float(wf)}
       for wf in config["wind_force"]]
)

# Buffer size cap: ensure online training fires at least 200 times within TUNE_STEPS.
# Training starts at step (Np + buffer_size + 1), so cap = TUNE_STEPS - Np - 200.
_MAX_BUFFER = TUNE_STEPS/2 - config["Np"]# 2 750

# Search spaces per method.  Each entry: name → (kind, low, high)
#   kind: "log_float" | "float" | "int"
SEARCH_SPACES: dict[str, dict] = {
    "replay_buffer": {
        "online_lr_replay":    ("log_float", 1e-8, 1e-6),
        "buffer_size":         ("int",        100,  _MAX_BUFFER),
        "online_weight_decay": ("log_float", 1e-7, 1e-3),
    },
    "residual_dynamics": {
        "online_lr_residual":  ("log_float", 1e-8, 1e-6),
        "buffer_size":         ("int",        100,  _MAX_BUFFER),
        "online_weight_decay": ("log_float", 1e-7, 1e-3),
    },
    "tube": {
        "K_tube_heading": ("float", -50.0, 0.0),
    },
    "tube_adaptive": {
        "K_tube_adaptive_heading": ("float",     -50.0,  0.0),
        "rbf_gamma":               ("log_float",   1.0, 1000.0),
        "rbf_sigma":               ("log_float",   0.05,  5.0),
    },
}

# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

@contextmanager
def patched_config(**overrides):
    """Temporarily override global config keys; always restores on exit."""
    originals = {k: config[k] for k in overrides if k in config}
    config.update(overrides)
    try:
        yield
    finally:
        for k in overrides:
            if k in originals:
                config[k] = originals[k]
            elif k in config:
                del config[k]


def params_to_overrides(method: str, params: dict, steps: int) -> dict:
    """Translate Optuna params dict into config key/value overrides."""
    overrides: dict = {"steps": steps, "no_rendering_mode": True}
    if method == "replay_buffer":
        overrides["online_lr_replay"]    = params["online_lr_replay"]
        overrides["buffer_size"]         = params["buffer_size"]
        overrides["online_weight_decay"] = params["online_weight_decay"]
    elif method == "residual_dynamics":
        overrides["online_lr_residual"]  = params["online_lr_residual"]
        overrides["buffer_size"]         = params["buffer_size"]
        overrides["online_weight_decay"] = params["online_weight_decay"]
    elif method == "tube":
        overrides["K_tube"] = [0.0, 0.0, params["K_tube_heading"]]
    elif method == "tube_adaptive":
        overrides["K_tube_adaptive"] = [0.0, 0.0, params["K_tube_adaptive_heading"]]
        overrides["rbf_gamma"]       = params["rbf_gamma"]
        overrides["rbf_sigma"]       = params["rbf_sigma"]
    return overrides

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial, method: str) -> dict:
    params = {}
    for name, spec in SEARCH_SPACES[method].items():
        kind = spec[0]
        if kind == "log_float":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
    return params


def make_objective(method, temp_dir, n_seeds, steps, scenarios, model_path):
    """Return an Optuna objective closure over the given settings."""

    def objective(trial: optuna.Trial) -> float:
        params   = suggest_params(trial, method)
        overrides = params_to_overrides(method, params, steps)

        print(f"\n[Trial {trial.number}] {method}")
        for k, v in params.items():
            print(f"  {k}: {v:.3g}" if isinstance(v, float) else f"  {k}: {v}")

        rmse_values: list[float] = []

        with patched_config(**overrides):
            for seed_idx in range(n_seeds):
                seed = BASE_SEED + seed_idx
                # Seed all RNG sources that affect replay-buffer sampling
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                for scen in scenarios:
                    with patched_config(seed=seed):
                        rmse = simulate_carla(
                            "tune_temp",
                            temp_dir,
                            method=method,
                            steering_force=scen["steering_force"],
                            wind_force=scen.get("wind_force", 0.0),
                            model_path=model_path,
                        )
                    rmse_values.append(rmse)
                    print(
                        f"  seed={seed_idx}  scen={scen.get('name', scen)}"
                        f"  rmse={rmse:.4f} m"
                    )

        mean_rmse = float(np.mean(rmse_values))
        print(f"[Trial {trial.number}] → mean RMSE = {mean_rmse:.4f} m")
        return mean_rmse

    return objective

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(method: str, params: dict, log_dir: Path,
                   model_path: str, steps: int = FULL_STEPS) -> dict:
    """Run the best params on all 7 disturbance scenarios at full step count."""
    print(f"\n{'='*55}")
    print(f"Validation — method: {method}  ({steps} steps)")
    print(f"Params: {params}")

    overrides = params_to_overrides(method, params, steps)
    val_dir   = log_dir / "validation"
    (val_dir / "plots").mkdir(parents=True, exist_ok=True)
    (val_dir / "models").mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}
    with patched_config(**overrides):
        for scen in ALL_SCENARIOS:
            name = scen["name"]
            print(f"\n  Scenario: {name}")
            rmse = simulate_carla(
                name, val_dir,
                method=method,
                steering_force=scen["steering_force"],
                wind_force=scen.get("wind_force", 0.0),
                model_path=model_path,
            )
            results[name] = rmse
            print(f"  {name:<20}: RMSE = {rmse:.4f} m")

    print(f"\n{'--- Validation Summary ':->55}")
    for name, rmse in results.items():
        print(f"  {name:<20}: {rmse:.4f} m")
    print(f"  {'Mean':<20}: {np.mean(list(results.values())):.4f} m")

    out = {"method": method, "params": params, "steps": steps, "results": results}
    with open(log_dir / "validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {log_dir}/validation_results.json")
    return results

# ---------------------------------------------------------------------------
# Optuna visualisation
# ---------------------------------------------------------------------------

def save_optuna_plots(study: optuna.Study, plot_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as oviz

        for plot_fn, fname in [
            (oviz.plot_optimization_history,  "optimization_history.png"),
            (oviz.plot_param_importances,      "param_importances.png"),
            (oviz.plot_parallel_coordinate,    "parallel_coordinate.png"),
        ]:
            try:
                ax = plot_fn(study)
                ax.figure.savefig(str(plot_dir / fname), bbox_inches="tight")
                plt.close(ax.figure)
            except Exception as e:
                print(f"  Warning: {fname} skipped — {e}")

        print("Optuna plots saved.")
    except ImportError:
        print("Warning: optuna matplotlib visualisation not available.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna tuning for online MPC methods (Carla_MPC)"
    )
    parser.add_argument("--method", required=True,
                        choices=list(SEARCH_SPACES.keys()),
                        nargs='+',
                        help="One or more methods to tune sequentially")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Optuna trials to run (default 50)")
    parser.add_argument("--n_seeds", type=int, default=2,
                        help="CARLA runs per scenario per trial for noise averaging (default 3)")
    parser.add_argument("--steps", type=int, default=TUNE_STEPS,
                        help=f"Sim steps per run (default {TUNE_STEPS})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the pre-trained base model")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name (auto-generated if omitted; single method only)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an existing study log dir to resume (single method only)")
    # Validation mode
    parser.add_argument("--validate", action="store_true",
                        help="Validation mode: run best params at full steps on all scenarios")
    parser.add_argument("--params_file", type=str, default=None,
                        help="Path to best_params.json (required with --validate)")
    args = parser.parse_args()

    # ---- Validation mode ----
    if args.validate:
        if not args.params_file:
            parser.error("--validate requires --params_file")
        with open(args.params_file) as f:
            best_info = json.load(f)
        log_dir = Path(args.params_file).parent
        run_validation(
            best_info["method"], best_info["params"], log_dir,
            model_path=args.model,
            steps=args.steps if args.steps != TUNE_STEPS else FULL_STEPS,
        )
        return

    # ---- Tuning mode ----
    if len(args.method) > 1 and args.resume:
        parser.error("--resume can only be used with a single method")
    if len(args.method) > 1 and args.study_name:
        parser.error("--study_name can only be used with a single method")

    batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tune_scenarios = ALL_SCENARIOS

    for method in args.method:
        print(f"\n{'='*55}")
        print(f"Tuning method: {method}  ({args.method.index(method)+1}/{len(args.method)})")

        study_name = args.study_name or f"{method}_{batch_timestamp}"

        if args.resume:
            log_dir = Path(args.resume)
            if not (log_dir / "study.log").exists():
                raise FileNotFoundError(f"No study.log found in {log_dir}")
            study_name = log_dir.name.removeprefix("study_")
        else:
            log_dir = Path("Experiments/Tuning/logs") / f"study_{study_name}"

        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "plots").mkdir(exist_ok=True)

        temp_dir = log_dir / "temp"
        (temp_dir / "plots").mkdir(parents=True, exist_ok=True)
        (temp_dir / "models").mkdir(parents=True, exist_ok=True)

        # JournalStorage (file-backed) instead of SQLite: safe for many
        # concurrent Slurm array workers writing to one study on Lustre.
        storage = JournalStorage(JournalFileBackend(str(log_dir / "study.log")))

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(seed=None),
        )

        already_done = len(study.trials)
        print(f"Study:      {study_name}")
        print(f"Log dir:    {log_dir}")
        n_runs = args.n_seeds * len(tune_scenarios)
        print(f"Steps/run:  {args.steps}  ×  {args.n_seeds} seeds  ×  {len(tune_scenarios)} scenarios"
              f"  = {args.steps * n_runs} sim-steps / Optuna trial")
        print(f"Scenarios:  {[s['name'] for s in tune_scenarios]}")
        print(f"n_trials:   {args.n_trials}  (+{already_done} already completed)")
        print(f"Search space:")
        for k, v in SEARCH_SPACES[method].items():
            print(f"  {k:<25} {v}")

        t0        = time.time()
        objective = make_objective(
            method, temp_dir, args.n_seeds,
            args.steps, tune_scenarios, args.model,
        )
        study.optimize(objective, n_trials=args.n_trials)

        elapsed    = time.time() - t0
        h, rem     = divmod(int(elapsed), 3600)
        m, s       = divmod(rem, 60)

        best = study.best_trial
        print(f"\n{'='*55}")
        print(f"Best trial #{best.number}  —  mean RMSE = {best.value:.4f} m")
        for k, v in best.params.items():
            print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"Elapsed: {h:02d}:{m:02d}:{s:02d}")

        best_info = {
            "method":           method,
            "study_name":       study_name,
            "best_trial":       best.number,
            "mean_rmse":        best.value,
            "params":           best.params,
            "tune_steps":       args.steps,
            "n_seeds":          args.n_seeds,
            "tuning_scenario":  tune_scenarios[0],
        }
        with open(log_dir / "best_params.json", "w") as f:
            json.dump(best_info, f, indent=2)
        print(f"Best params → {log_dir}/best_params.json")

        save_optuna_plots(study, log_dir / "plots")
        print(f"All outputs → {log_dir}")


if __name__ == "__main__":
    main()
