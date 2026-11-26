import argparse
import json
import os
from functools import partial
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from data.loader import DEFAULT_CACHE, DIRECTIONS, load_lee2019, parse_subjects
from experiments.sensitivity import DEFAULT, config_name, grid
from experiments.within_session import SPLITS, split_seeds
from models.csp_lda import build_csp_lda
from run_baselines import DEFAULT_SEEDS, new_run_dir, parse_seeds, run_all
from utils.metrics import MAIN_CONDITIONS, MAIN_GAPS, summarize, within_chance_tests
from utils.plotting import plot_sensitivity

METRICS = MAIN_CONDITIONS + MAIN_GAPS
TABLES = ("summary", "statistics", "subject_summary", "direction_summary")


def parse_args():
    parser = argparse.ArgumentParser(description="CSP+LDA hyperparameter sensitivity on Lee2019 MI")
    parser.add_argument("--subjects", default="all", help="comma-separated IDs or 'all'")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--direction", choices=list(DIRECTIONS), default="both")
    parser.add_argument("--split", choices=SPLITS, default="blocks")
    parser.add_argument("--out", default="results")
    parser.add_argument("--tag", help="run folder name; defaults to a timestamp, never overwrites")
    parser.add_argument("--n-jobs", type=int, default=3,
                        help="configurations evaluated in parallel; each job holds its own copy of the data")
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=32.0)
    parser.add_argument("--resample", type=int, default=128)
    parser.add_argument("--plot-only", action="store_true", help="redraw the figure of --tag")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run in --tag, reusing configurations that already finished")
    return parser.parse_args()


def config_done(config_dir):
    return all((config_dir / f"{name}.csv").exists() for name in ("results", *TABLES))


def collect_config(config, config_dir):
    statistics = pd.read_csv(config_dir / "statistics.csv")
    subject_summary = pd.read_csv(config_dir / "subject_summary.csv")
    chance_path = config_dir / "within_chance_tests.csv"
    chance = pd.read_csv(chance_path) if chance_path.exists() else None
    stats = statistics.set_index("metric").loc[METRICS].reset_index()
    stats = stats.assign(config=config_name(config), **{k: str(v) for k, v in config.items()},
                         is_default=config == DEFAULT,
                         within_above_chance=None if chance is None else int(chance["above_chance"].sum()))
    subjects_table = subject_summary[["test_subject", *METRICS]].assign(config=config_name(config))
    return stats, subjects_table


def check_finished(config_dir, subjects, seeds):
    results = pd.read_csv(config_dir / "results.csv")
    found_subjects = sorted(results["test_subject"].unique().tolist())
    found_seeds = sorted(results["seed"].dropna().astype(int).unique().tolist())
    if found_subjects != sorted(subjects) or found_seeds != sorted(seeds):
        raise ValueError(f"{config_dir} was run with subjects {found_subjects} and seeds {found_seeds}; "
                         "it does not match this run's settings")


def evaluate_config(config, args, subjects, seeds, out_dir, threads):
    config_dir = out_dir / "configs" / config_name(config)
    if config_done(config_dir):
        return collect_config(config, config_dir)
    with threadpool_limits(limits=threads):
        data = load_lee2019(subjects, fmin=args.fmin, fmax=args.fmax, resample=args.resample,
                            cache_dir=DEFAULT_CACHE)
        results, predictions = run_all(data, subjects, seeds, partial(build_csp_lda, **config),
                                       args.direction, args.split)
    tables = summarize(results)
    chance = within_chance_tests(predictions)

    config_dir.mkdir(parents=True)
    if chance is not None:
        chance.to_csv(config_dir / "within_chance_tests.csv", index=False)
    for name in TABLES:
        tables[name].to_csv(config_dir / f"{name}.csv", index=False)
    results.to_csv(config_dir / "results.csv", index=False)
    return collect_config(config, config_dir)


def robustness(summary, per_subject):
    default_name = config_name(DEFAULT)
    default = per_subject[per_subject["config"] == default_name].set_index("test_subject")
    rows = []
    for metric in METRICS:
        means = summary[summary["metric"] == metric].set_index("config")
        is_gap = metric in MAIN_GAPS
        significant = (means["wilcoxon_p"] < 0.05) & (means["mean"] > (0.0 if is_gap else 0.5))
        best = means["mean"].idxmax()
        rows.append({
            "metric": metric,
            "default": means.loc[default_name, "mean"],
            "min_over_configs": means["mean"].min(),
            "max_over_configs": means["mean"].max(),
            "range": means["mean"].max() - means["mean"].min(),
            "highest_config": best,
            "highest_minus_default": means.loc[best, "mean"] - means.loc[default_name, "mean"],
            "default_rank_from_highest": int(means["mean"].rank(ascending=False, method="min").loc[default_name]),
            "n_configs": len(means),
            "n_configs_significantly_positive": int(significant.sum()),
            "test": "Wilcoxon two-sided vs 0" if is_gap else "Wilcoxon one-sided greater vs 0.5",
        })
    corr = (per_subject.pivot(index="test_subject", columns="config", values="within_session")
            .corrwith(default["within_session"]))
    return pd.DataFrame(rows), corr.rename("within_session_corr_with_default").reset_index()


def main():
    args = parse_args()
    run_root = Path(args.out) / "csp_lda_sensitivity" / args.direction

    if args.plot_only:
        out_dir = run_root / args.tag
        plot_sensitivity(pd.read_csv(out_dir / "sensitivity_summary.csv"), out_dir / "sensitivity.png")
        print("Figure saved to", out_dir.resolve())
        return

    subjects = parse_subjects(args.subjects)
    seeds = split_seeds(args.split, parse_seeds(args.seeds))
    configs = grid()
    if args.resume:
        if not args.tag:
            raise ValueError("--resume needs the --tag of the run to continue")
        out_dir = run_root / args.tag
        if (out_dir / "config.json").exists():
            raise FileExistsError(f"{out_dir} is already complete")
        finished = [c for c in configs if config_done(out_dir / "configs" / config_name(c))]
        for config in finished:
            check_finished(out_dir / "configs" / config_name(config), subjects, seeds)
        print(f"Resuming {out_dir}: {len(finished)} of {len(configs)} configurations already finished")
    else:
        out_dir = new_run_dir(run_root, args.tag, args.split)
    threads = max(1, (os.cpu_count() or 1) // args.n_jobs)
    print(f"{len(configs)} configurations, split {args.split}, seeds/folds {seeds}, "
          f"{args.n_jobs} parallel jobs x {threads} threads")

    outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(evaluate_config)(config, args, subjects, seeds, out_dir, threads) for config in configs
    )
    summary = pd.concat([stats for stats, _ in outputs], ignore_index=True)
    per_subject = pd.concat([table for _, table in outputs], ignore_index=True)
    robust, corr = robustness(summary, per_subject)

    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    per_subject.to_csv(out_dir / "sensitivity_per_subject.csv", index=False)
    robust.to_csv(out_dir / "robustness.csv", index=False)
    corr.to_csv(out_dir / "within_correlation_with_default.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps({
        "run": out_dir.name, "model": "csp_lda", "direction": args.direction, "split": args.split,
        "subjects": subjects, "seeds": seeds, "fmin": args.fmin, "fmax": args.fmax,
        "resample_hz": args.resample, "default": DEFAULT, "grid": configs,
        "selection": "no configuration is selected; every configuration is evaluated on the same splits",
    }, indent=2), encoding="utf-8")
    plot_sensitivity(summary, out_dir / "sensitivity.png")

    wide = summary.pivot(index="config", columns="metric", values="mean")[METRICS]
    wide["within_above_chance"] = summary.groupby("config")["within_above_chance"].first()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nMEAN BALANCED ACCURACY / GAP BY CONFIGURATION\n", wide.round(3).to_string())
        print("\nROBUSTNESS\n", robust.round(4).to_string(index=False))
    print("\nResults written to", out_dir.resolve())


if __name__ == "__main__":
    main()
