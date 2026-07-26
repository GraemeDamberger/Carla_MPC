"""
Analyse the HPC tuning studies: export CSVs and plots for review.

Reads the Optuna journal studies written by tune_array.sh (no CARLA, no torch)
and, if the Slurm worker logs are available, parses them for the per-scenario
RMSE breakdown of each method's best trial — which the study itself does not
store (it only keeps the scalar mean objective).

Usage (from repo root):
    python -m Experiments.Tuning.analyze
    python -m Experiments.Tuning.analyze \
        --logs "$SCRATCH/carla/logs" \
        --methods tube replay_buffer residual_dynamics tube_adaptive

Outputs land in Experiments/Tuning/logs/analysis/:
    summary.csv                 one row per method (best trial + params + counts)
    <method>_trials.csv         every trial: number, state, mean RMSE, params
    per_scenario_best.csv       best-trial RMSE per scenario, methods as columns
    *.png                       plots (only if matplotlib is importable)

Plots need matplotlib. If it is missing on the cluster env, install it with:
    pip install --no-index matplotlib
CSV export works without it.
"""

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

STUDY_SUFFIX = "sweep_v1"
LOG_ROOT     = Path("Experiments/Tuning/logs")
DEFAULT_METHODS = ["tube", "replay_buffer", "residual_dynamics", "tube_adaptive"]

# Canonical scenario order (mirrors tune.py ALL_SCENARIOS) for stable plotting.
SCENARIO_ORDER = [
    "base", "steer_0.1", "steer_0.2", "steer_0.3",
    "wind_5000", "wind_10000", "wind_15000",
]

SEED_RE  = re.compile(r"seed=(\d+)\s+scen=(\S+)\s+rmse=([\d.]+)")
START_RE = re.compile(r"^\[Trial (\d+)\]\s+(\w+)\s*$")
STUDY_RE = re.compile(r"^Study:\s+(\S+)")

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_study(method: str, suffix: str):
    name    = f"{method}_{suffix}"
    journal = LOG_ROOT / f"study_{name}" / "study.log"
    if not journal.exists():
        return name, None
    storage = JournalStorage(JournalFileBackend(str(journal)))
    try:
        return name, optuna.load_study(study_name=name, storage=storage)
    except Exception as e:
        print(f"  {method}: could not load study — {e}")
        return name, None


def parse_logs(log_glob: str) -> dict:
    """Parse worker logs into {method: {trial: {scenario: [rmse, ...]}}}."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in glob.glob(log_glob):
        method, cur_trial = None, None
        try:
            with open(path, errors="replace") as f:
                for raw in f:
                    line = raw.strip()
                    sm = STUDY_RE.match(line)
                    if sm:
                        study  = sm.group(1)
                        method = (study[:-len(f"_{STUDY_SUFFIX}")]
                                  if study.endswith(f"_{STUDY_SUFFIX}") else study)
                        continue
                    tm = START_RE.match(line)
                    if tm:
                        cur_trial = int(tm.group(1))
                        continue
                    em = SEED_RE.search(line)
                    if em and method is not None and cur_trial is not None:
                        _seed, scen, rmse = em.groups()
                        data[method][cur_trial][scen].append(float(rmse))
        except OSError as e:
            print(f"  warning: could not read {path} — {e}")
    return data


def scenario_means(trial_scens: dict) -> dict:
    """{scenario: [rmse per seed]} -> {scenario: mean over seeds}."""
    return {s: sum(v) / len(v) for s, v in trial_scens.items() if v}

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_trials_csv(method: str, study, out_dir: Path) -> None:
    param_keys = sorted({k for t in study.trials for k in t.params})
    path = out_dir / f"{method}_trials.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_number", "state", "mean_rmse", *param_keys])
        for t in sorted(study.trials, key=lambda t: t.number):
            w.writerow([
                t.number, t.state.name,
                "" if t.value is None else f"{t.value:.6f}",
                *[t.params.get(k, "") for k in param_keys],
            ])
    print(f"  wrote {path}")


def build_summary(methods, studies, parsed) -> list:
    rows = []
    for method in methods:
        study = studies[method]
        if study is None:
            rows.append({"method": method, "n_completed": 0, "n_failed": 0,
                         "n_total": 0, "best_trial": "", "best_mean_rmse": "",
                         "best_params": ""})
            continue
        states   = [t.state for t in study.trials]
        n_done   = states.count(TrialState.COMPLETE)
        n_fail   = states.count(TrialState.FAIL)
        try:
            best = study.best_trial
            rows.append({
                "method": method, "n_completed": n_done, "n_failed": n_fail,
                "n_total": len(study.trials), "best_trial": best.number,
                "best_mean_rmse": f"{best.value:.6f}",
                "best_params": json.dumps(best.params),
            })
        except ValueError:
            rows.append({"method": method, "n_completed": n_done,
                         "n_failed": n_fail, "n_total": len(study.trials),
                         "best_trial": "", "best_mean_rmse": "",
                         "best_params": ""})
    return rows


def write_summary_csv(rows, out_dir: Path) -> None:
    path = out_dir / "summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "n_completed", "n_failed", "n_total",
            "best_trial", "best_mean_rmse", "best_params"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def build_per_scenario(methods, studies, parsed) -> dict:
    """{method: {scenario: mean rmse of best trial}} from parsed worker logs."""
    out = {}
    for method in methods:
        study = studies[method]
        if study is None:
            continue
        try:
            best_n = study.best_trial.number
        except ValueError:
            continue
        trial_scens = parsed.get(method, {}).get(best_n)
        if trial_scens:
            out[method] = scenario_means(trial_scens)
    return out


def write_per_scenario_csv(per_scen, out_dir: Path) -> None:
    if not per_scen:
        print("  per_scenario: no worker-log data found — skipped "
              "(pass --logs pointing at the Slurm .out files)")
        return
    methods   = list(per_scen)
    scenarios = [s for s in SCENARIO_ORDER
                 if any(s in per_scen[m] for m in methods)]
    path = out_dir / "per_scenario_best.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", *methods])
        for s in scenarios:
            w.writerow([s, *[f"{per_scen[m].get(s, ''):.6f}"
                             if s in per_scen[m] else "" for m in methods]])
    print(f"  wrote {path}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(rows, out_dir: Path) -> None:
    pairs = [(r["method"], float(r["best_mean_rmse"]), r["n_completed"])
             for r in rows if r["best_mean_rmse"] != ""]
    if not pairs:
        return
    names, vals, counts = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, vals, color="#4C72B0")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Best mean RMSE [m]")
    ax.set_title("Best objective per method (n = completed trials)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(out_dir / "comparison_best_rmse.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out_dir / 'comparison_best_rmse.png'}")


def plot_history(method: str, study, out_dir: Path) -> None:
    done = [t for t in sorted(study.trials, key=lambda t: t.number)
            if t.state == TrialState.COMPLETE and t.value is not None]
    if not done:
        return
    xs   = [t.number for t in done]
    ys   = [t.value for t in done]
    best, run = float("inf"), []
    for y in ys:
        best = min(best, y)
        run.append(best)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, s=20, alpha=0.5, label="trial")
    ax.plot(xs, run, color="#C44E52", label="running best")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Mean RMSE [m]")
    ax.set_title(f"Optimization history — {method}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{method}_history.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out_dir / f'{method}_history.png'}")


def plot_per_scenario(per_scen, out_dir: Path) -> None:
    if not per_scen:
        return
    import numpy as np
    methods   = list(per_scen)
    scenarios = [s for s in SCENARIO_ORDER
                 if any(s in per_scen[m] for m in methods)]
    x     = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(methods):
        vals = [per_scen[m].get(s, 0.0) for s in scenarios]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=20)
    ax.set_ylabel("RMSE [m]  (best trial, seed-averaged)")
    ax.set_title("Per-scenario RMSE of each method's best configuration")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "per_scenario_best.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out_dir / 'per_scenario_best.png'}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Analyse Carla_MPC tuning studies")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--suffix", default=STUDY_SUFFIX,
                   help=f"study-name suffix (default '{STUDY_SUFFIX}')")
    p.add_argument("--logs", default=None,
                   help="dir holding Slurm worker .out files "
                        "(for per-scenario breakdown); e.g. $SCRATCH/carla/logs")
    p.add_argument("--out", default=str(LOG_ROOT / "analysis"),
                   help="output directory for CSVs and plots")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading studies:")
    studies = {}
    for m in args.methods:
        _, studies[m] = load_study(m, args.suffix)

    parsed = {}
    if args.logs:
        parsed = parse_logs(str(Path(args.logs) / "carla_hpo_tune-*.out"))

    print("\nExporting CSVs:")
    rows = build_summary(args.methods, studies, parsed)
    write_summary_csv(rows, out_dir)
    for m in args.methods:
        if studies[m] is not None and studies[m].trials:
            write_trials_csv(m, studies[m], out_dir)
    per_scen = build_per_scenario(args.methods, studies, parsed)
    write_per_scenario_csv(per_scen, out_dir)

    if plt is None:
        print("\nmatplotlib not available — skipping plots "
              "(pip install --no-index matplotlib to enable).")
    else:
        print("\nWriting plots:")
        plot_comparison(rows, out_dir)
        for m in args.methods:
            if studies[m] is not None:
                plot_history(m, studies[m], out_dir)
        plot_per_scenario(per_scen, out_dir)

    print("\n" + "=" * 55)
    print(f"{'method':<20}{'n_done':>8}{'n_fail':>8}{'best RMSE':>12}")
    for r in rows:
        best = r["best_mean_rmse"] or "—"
        print(f"{r['method']:<20}{r['n_completed']:>8}{r['n_failed']:>8}{best:>12}")
    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
