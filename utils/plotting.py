import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.metrics import MAIN_CONDITIONS as CONDITIONS, MAIN_GAPS as GAPS

CHANCE = 0.5


def _tick_labels(names):
    return [name.replace("_", " ") for name in names]


def _chance_line(ax):
    ax.axhline(CHANCE, linestyle="--", color="gray", label="chance (balanced classes)")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_paired_effects(summary, path):
    """Plot each subject's balanced accuracy before and after ICA."""
    experiments = summary["experiment"].unique()
    fig, axes = plt.subplots(1, len(experiments), figsize=(6 * len(experiments), 5),
                             squeeze=False)
    for ax, experiment in zip(axes[0], experiments):
        rows = summary[summary["experiment"] == experiment]
        for _, row in rows.iterrows():
            color = "tab:green" if row["delta_ica"] > 0 else "tab:red"
            ax.plot([0, 1], [row["baseline"], row["ica"]], color=color, alpha=.35)
        ax.scatter(np.zeros(len(rows)), rows["baseline"], s=18)
        ax.scatter(np.ones(len(rows)), rows["ica"], s=18)
        _chance_line(ax)
        ax.set_xticks([0, 1], ["Baseline", "ICA"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(experiment.replace("_", " "))
    _save(fig, path)


def _direction_tag(out_dir):
    config_path = out_dir / "config.json"
    if not config_path.exists():
        return ""
    direction = json.loads(config_path.read_text(encoding="utf-8")).get("direction", "both")
    return {"1to2": " [session 1 → 2]", "2to1": " [session 2 → 1]"}.get(direction, " [both directions]")


def plot_conditions(subject_summary, path, n_seeds, tag=""):
    data = [subject_summary[c].to_numpy() for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, showfliers=False)
    for i, values in enumerate(data, 1):
        ax.scatter(np.full(len(values), i), values, s=14, alpha=.7)
    ax.set_xticks(range(1, len(CONDITIONS) + 1), _tick_labels(CONDITIONS))
    _chance_line(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(f"Per-subject balanced accuracy (mean over {n_seeds} seeds){tag}")
    ax.legend()
    _save(fig, path)


def plot_gaps(subject_summary, path, n_seeds, tag=""):
    x = np.arange(len(subject_summary))
    fig, ax = plt.subplots(figsize=(max(9, 0.3 * len(x)), 5))
    width = 0.8 / len(GAPS)
    for i, name in enumerate(GAPS):
        offset = (i - (len(GAPS) - 1) / 2) * width
        err = subject_summary[f"{name}_seed_sd"].fillna(0)
        ax.bar(x + offset, subject_summary[name], width=width, yerr=err, capsize=3, label=name)
    ax.axhline(0, linewidth=1, color="gray")
    ax.set_xticks(x, subject_summary["test_subject"].astype(str))
    ax.set_xlabel("Test subject")
    ax.set_ylabel("Within-session BA minus held-out BA")
    ax.set_title(f"Generalization gaps (mean ± SD over {n_seeds} seeds){tag}")
    ax.legend()
    _save(fig, path)


def plot_session_directions(session_gaps, path):
    per_subject = (session_gaps
                   .groupby(["test_subject", "train_session", "test_session"])["G_session"]
                   .agg(["mean", "std"])
                   .reset_index())
    directions = per_subject[["train_session", "test_session"]].drop_duplicates().to_numpy()
    subjects = sorted(per_subject["test_subject"].unique())
    x = np.arange(len(subjects))
    width = 0.8 / len(directions)
    fig, ax = plt.subplots(figsize=(max(9, 0.3 * len(x)), 5))
    for i, (src, tgt) in enumerate(directions):
        rows = (per_subject[(per_subject["train_session"] == src) & (per_subject["test_session"] == tgt)]
                .set_index("test_subject").loc[subjects])
        offset = (i - (len(directions) - 1) / 2) * width
        ax.bar(x + offset, rows["mean"], width=width, yerr=rows["std"].fillna(0), capsize=3,
               label=f"session {src} → {tgt}")
    ax.axhline(0, linewidth=1, color="gray")
    ax.set_xticks(x, [str(s) for s in subjects])
    ax.set_xlabel("Test subject")
    ax.set_ylabel("G_session (within BA minus cross-session BA)")
    ax.set_title("Cross-session gap by direction (mean ± SD over seeds)")
    ax.legend()
    _save(fig, path)


def plot_seed_variability(seed_summary, path, tag=""):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot([seed_summary[c].to_numpy() for c in CONDITIONS], showfliers=False)
    for i, c in enumerate(CONDITIONS, 1):
        jitter = rng.uniform(-0.08, 0.08, len(seed_summary))
        ax.scatter(i + jitter, seed_summary[c], s=14, alpha=.7)
    ax.set_xticks(range(1, len(CONDITIONS) + 1), _tick_labels(CONDITIONS))
    _chance_line(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean balanced accuracy over subjects")
    ax.set_title(f"Cohort mean for each of {len(seed_summary)} seeds{tag}")
    ax.legend()
    _save(fig, path)


def save_figures(out_dir):
    out_dir = Path(out_dir)
    subject_summary = pd.read_csv(out_dir / "subject_summary.csv")
    seed_summary = pd.read_csv(out_dir / "seed_summary.csv")
    session_gaps = pd.read_csv(out_dir / "session_gaps.csv")
    n_seeds = len(seed_summary)
    tag = _direction_tag(out_dir)

    plot_conditions(subject_summary, out_dir / "conditions_boxplot.png", n_seeds, tag)
    plot_gaps(subject_summary, out_dir / "generalization_gaps.png", n_seeds, tag)
    plot_session_directions(session_gaps, out_dir / "session_directions.png")
    if n_seeds > 1:
        plot_seed_variability(seed_summary, out_dir / "seed_variability.png", tag)
