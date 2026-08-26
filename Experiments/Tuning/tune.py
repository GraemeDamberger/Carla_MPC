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

from Experiments.Comparison.config import config, CONDITIONS
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

# Study-name suffix used by the HPC array script (tune_array.sh names each study
# "<method>_<STUDY_SUFFIX>"). --report reconstructs study names with the same rule.
STUDY_SUFFIX = "sweep_v4"

# Evaluation grid: each route (spawn index) x each disturbance condition. The
# objective normalizes per-route by that route's nominal RMSE so no single route
# dominates. Routes come from config['route_spawn_indices']; CONDITIONS from config.

# Buffer size cap: ensure online training fires at least 200 times within TUNE_STEPS.
# Training starts at step (Np + buffer_size + 1), so cap = TUNE_STEPS - Np - 200.
_MAX_BUFFER = TUNE_STEPS/2 - config["Np"]# 2 750

# Search spaces per method.  Each entry: name → (kind, low, high)
#   kind: "log_float" | "float" | "int"
# "R" (MPC control-effort weight) is tuned for every method so the sweep finds
# each method's tracking-vs-smoothness balance.
SEARCH_SPACES: dict[str, dict] = {
    "replay_buffer": {
        "online_lr_replay":    ("log_float", 1e-8, 1e-6),
        "buffer_size":         ("int",        100,  _MAX_BUFFER),
        "online_weight_decay": ("log_float", 1e-7, 1e-3),
        "R":                   ("log_float", 1e-2, 1e2),
    },
    "residual_dynamics": {
        "online_lr_residual":  ("log_float", 1e-8, 1e-6),
        "buffer_size":         ("int",        100,  _MAX_BUFFER),
        "online_weight_decay": ("log_float", 1e-7, 1e-3),
        "R":                   ("log_float", 1e-2, 1e2),
    },
    "tube": {
        "K_tube_heading": ("float",     -50.0, 0.0),
        "R":              ("log_float",  1e-2, 1e2),
    },
    "tube_adaptive": {
        "K_tube_adaptive_heading": ("float",     -50.0,  0.0),
        "rbf_gamma":               ("log_float",   1.0, 1000.0),
        "rbf_sigma":               ("log_float",   0.05,  5.0),
        "R":                       ("log_float",   1e-2,  1e2),
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


def params_to_overrides(method: str, params: dict, steps: int,
                        save_plots: bool = False) -> dict:
    """Translate Optuna params dict into config key/value overrides.

    save_plots stays False for tuning (hundreds of throwaway rollouts) and is
    turned on for validation, where the trajectory plots are the deliverable.
    """
    overrides: dict = {"steps": steps, "no_rendering_mode": True,
                       "save_plots": save_plots}
    if "R" in params:
        overrides["R"] = params["R"]
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


def make_objective(method, temp_dir, n_seeds, steps, routes, model_path,
                   rmse_cap=None):
    """Return an Optuna objective closure over the given settings.

    Objective = mean over (route x condition x seed) of the per-route-normalized
    RMSE: each condition's RMSE is divided by that route's nominal RMSE, so route
    difficulty does not skew the comparison. `rmse_cap` bounds that ratio.
    """

    def objective(trial: optuna.Trial) -> float:
        params    = suggest_params(trial, method)
        overrides = params_to_overrides(method, params, steps)

        print(f"\n[Trial {trial.number}] {method}")
        for k, v in params.items():
            print(f"  {k}: {v:.3g}" if isinstance(v, float) else f"  {k}: {v}")

        norm_floor = config['rmse_norm_floor']
        ratios: list[float] = []

        with patched_config(**overrides):
            for seed_idx in range(n_seeds):
                seed = BASE_SEED + seed_idx
                # Seed all RNG sources that affect replay-buffer sampling
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                for spawn_index in routes:
                    # Run every condition on this route; normalize by nominal.
                    route_rmse: dict[str, float] = {}
                    for cond in CONDITIONS:
                        with patched_config(seed=seed):
                            route_rmse[cond["name"]] = simulate_carla(
                                "tune_temp", temp_dir, method=method,
                                steering_force=cond["steering_force"],
                                flat_tire=cond["flat_tire"],
                                surface=cond["surface"], wind=cond["wind"],
                                spawn_index=spawn_index, model_path=model_path,
                            )

                    base = max(route_rmse["nominal"], norm_floor)
                    for cname, rmse in route_rmse.items():
                        r = rmse / base
                        if rmse_cap is not None:
                            r = min(r, rmse_cap)
                        ratios.append(r)
                        print(f"  seed={seed_idx}  route={spawn_index}  "
                              f"cond={cname}  rmse={rmse:.4f} m  norm={r:.3f}")

        score = float(np.mean(ratios))
        print(f"[Trial {trial.number}] → mean normalized RMSE = {score:.4f}")
        return score

    return objective

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(method: str, params: dict, log_dir: Path,
                   model_path: str, steps: int = FULL_STEPS) -> dict:
    """Run the best params over every route x condition at full step count.

    Records the raw RMSE for each (route, condition) plus the per-route
    normalized ratio, so the writeup can show both absolute tracking and
    disturbance-rejection. Result keys are "route<idx>/<condition>".
    """
    print(f"\n{'='*55}")
    print(f"Validation — method: {method}  ({steps} steps)")
    print(f"Params: {params}")

    # Validation keeps its plots: the trajectory figures are the point.
    overrides   = params_to_overrides(method, params, steps, save_plots=True)
    routes      = config['route_spawn_indices']
    norm_floor  = config['rmse_norm_floor']
    val_dir     = log_dir / "validation"
    (val_dir / "plots").mkdir(parents=True, exist_ok=True)
    (val_dir / "models").mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}
    normalized: dict[str, float] = {}
    with patched_config(**overrides):
        for spawn_index in routes:
            route_rmse: dict[str, float] = {}
            for cond in CONDITIONS:
                name = cond["name"]
                print(f"\n  Route {spawn_index} / {name}")
                rmse = simulate_carla(
                    f"route{spawn_index}_{name}", val_dir, method=method,
                    steering_force=cond["steering_force"],
                    flat_tire=cond["flat_tire"],
                    surface=cond["surface"], wind=cond["wind"],
                    spawn_index=spawn_index, model_path=model_path,
                )
                route_rmse[name] = rmse
                print(f"  route{spawn_index}/{name:<12}: RMSE = {rmse:.4f} m")

            base = max(route_rmse["nominal"], norm_floor)
            for name, rmse in route_rmse.items():
                key = f"route{spawn_index}/{name}"
                results[key]    = rmse
                normalized[key] = rmse / base

    print(f"\n{'--- Validation Summary ':->55}")
    for key, rmse in results.items():
        print(f"  {key:<22}: {rmse:.4f} m   (norm {normalized[key]:.3f})")
    print(f"  {'Mean normalized':<22}: {np.mean(list(normalized.values())):.4f}")

    out = {"method": method, "params": params, "steps": steps,
           "results": results, "normalized": normalized}
    with open(log_dir / "validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {log_dir}/validation_results.json")
    return results

# ---------------------------------------------------------------------------
# Reporting (recover results straight from the shared journal)
# ---------------------------------------------------------------------------

def run_report(methods: list[str], suffix: str = STUDY_SUFFIX,
               study_name: str | None = None) -> None:
    """Load each method's journal study and print + rewrite its best_params.json.

    Safe to run against a study whose workers were killed at walltime: the
    journal holds every completed trial even when best_params.json was never
    written. Reads only — never launches CARLA.
    """
    log_root = Path("Experiments/Tuning/logs")
    for method in methods:
        name    = study_name or f"{method}_{suffix}"
        log_dir = log_root / f"study_{name}"
        journal = log_dir / "study.log"

        print(f"\n{'='*55}")
        print(f"{method}  (study: {name})")
        if not journal.exists():
            print(f"  no study.log found at {journal} — skipped")
            continue

        storage = JournalStorage(JournalFileBackend(str(journal)))
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except Exception as e:
            print(f"  could not load study: {e}")
            continue

        n_done = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  trials completed: {n_done}  (total records: {len(study.trials)})")

        try:
            best = study.best_trial
        except ValueError:
            print("  no completed trial yet — nothing to report")
            continue

        # v3+ objective is a dimensionless per-route-normalized ratio, not metres.
        print(f"  best trial #{best.number}: mean normalized RMSE = {best.value:.4f}")
        for k, v in best.params.items():
            print(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")

        best_info = {
            "method":     method,
            "study_name": name,
            "best_trial": best.number,
            "mean_rmse":  best.value,
            "params":     best.params,
            "n_trials":   n_done,
        }
        with open(log_dir / "best_params.json", "w") as f:
            json.dump(best_info, f, indent=2)
        print(f"  → wrote {log_dir}/best_params.json")

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
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Seeds per route/condition for noise averaging "
                             "(default 1; route diversity provides the averaging)")
    parser.add_argument("--rmse_cap", type=float, default=None,
                        help="Upper-bound the per-route NORMALIZED RMSE ratio at this value. "
                             "Stops any single blown-up condition from dominating the objective. "
                             "Changes objective semantics — use a fresh study.")
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
    # Report mode
    parser.add_argument("--report", action="store_true",
                        help="Report mode: read each method's journal study and "
                             "print + rewrite best_params.json (no CARLA runs)")
    parser.add_argument("--study_suffix", type=str, default=STUDY_SUFFIX,
                        help=f"Study-name suffix for --report (default '{STUDY_SUFFIX}', "
                             "matches tune_array.sh)")
    args = parser.parse_args()

    # ---- Report mode ----
    if args.report:
        run_report(
            args.method,
            suffix=args.study_suffix,
            study_name=args.study_name if len(args.method) == 1 else None,
        )
        return

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

    routes = config['route_spawn_indices']

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

        # Record the objective's cap so the study is self-documenting. Warn if a
        # resumed study was built with a different cap (values would be mixed).
        prior_cap = study.user_attrs.get("rmse_cap", "unset")
        if prior_cap not in ("unset", args.rmse_cap):
            print(f"WARNING: study was created with rmse_cap={prior_cap} but this "
                  f"run uses {args.rmse_cap}; objective values will be inconsistent.")
        study.set_user_attr("rmse_cap", args.rmse_cap)

        already_done = len(study.trials)
        print(f"Study:      {study_name}")
        print(f"Log dir:    {log_dir}")
        print(f"RMSE cap:   {args.rmse_cap}")
        n_runs = args.n_seeds * len(routes) * len(CONDITIONS)
        print(f"Grid:       {args.n_seeds} seeds × {len(routes)} routes × {len(CONDITIONS)} conditions"
              f"  = {n_runs} rollouts / trial  ({args.steps * n_runs} sim-steps)")
        print(f"Routes:     {routes}")
        print(f"Conditions: {[c['name'] for c in CONDITIONS]}")
        print(f"n_trials:   {args.n_trials}  (+{already_done} already completed)")
        print(f"Search space:")
        for k, v in SEARCH_SPACES[method].items():
            print(f"  {k:<25} {v}")

        t0        = time.time()
        objective = make_objective(
            method, temp_dir, args.n_seeds,
            args.steps, routes, args.model,
            rmse_cap=args.rmse_cap,
        )
        study.optimize(objective, n_trials=args.n_trials)

        elapsed    = time.time() - t0
        h, rem     = divmod(int(elapsed), 3600)
        m, s       = divmod(rem, 60)

        best = study.best_trial
        print(f"\n{'='*55}")
        print(f"Best trial #{best.number}  —  mean normalized RMSE = {best.value:.4f}")
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
            "routes":           routes,
        }
        with open(log_dir / "best_params.json", "w") as f:
            json.dump(best_info, f, indent=2)
        print(f"Best params → {log_dir}/best_params.json")

        save_optuna_plots(study, log_dir / "plots")
        print(f"All outputs → {log_dir}")


if __name__ == "__main__":
    main()
