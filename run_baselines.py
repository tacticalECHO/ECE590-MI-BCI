import argparse
import json
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

from data.loader import DEFAULT_CACHE, DIRECTIONS, load_lee2019, parse_subjects
from experiments import cross_session, cross_subject, within_session
from experiments.within_session import SPLITS, split_seeds
from models.csp_lda import build_csp_lda
from utils.metrics import confusion_counts, main_view, summarize, within_chance_tests
from utils.plotting import save_figures

MODELS = {"csp_lda": build_csp_lda}
DEFAULT_SEEDS = ",".join(str(s) for s in range(10))


def parse_seeds(spec):
    seeds = [int(s) for s in spec.split(",") if s.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use unique integer seeds, e.g. 0,1,2,3,4")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Lee2019 MI baselines")
    parser.add_argument("--model", choices=sorted(MODELS), default="csp_lda")
    parser.add_argument("--subjects", default="1,2,3,4,5", help="comma-separated IDs or 'all'")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="comma-separated split seeds")
    parser.add_argument("--out", default="results")
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=32.0)
    parser.add_argument("--resample", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--direction", choices=list(DIRECTIONS), default="both",
                        help="session direction for within/cross-session: 1to2, 2to1 or both")
    parser.add_argument("--split", choices=SPLITS, default="random",
                        help="within-session split: random 75/25 per seed, chrono (last 25%% held out), "
                             "or blocks (4 contiguous time blocks, seed = held-out block)")
    parser.add_argument("--no-cache", action="store_true",
                        help="re-read raw .mat files instead of the preprocessed per-subject cache")
    parser.add_argument("--tag", help="run folder name; defaults to a timestamp, never overwrites")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw figures from saved CSVs of --tag, or of the latest run")
    return parser.parse_args()


def latest_run(direction_dir):
    runs = [p for p in direction_dir.glob("*") if (p / "config.json").exists()]
    if not runs:
        raise FileNotFoundError(f"No saved runs in {direction_dir}")
    return max(runs, key=lambda p: (p / "config.json").stat().st_mtime)


def new_run_dir(direction_dir, tag, split):
    out_dir = direction_dir / (tag or f"{datetime.now():%Y%m%d-%H%M%S}_{split}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already has results; pick another --tag")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_all(data, subjects, seeds, build_model, direction, split):
    records, predictions = [], []
    for name, module in [
        ("within_session", within_session),
        ("cross_session", cross_session),
        ("cross_subject", cross_subject),
    ]:
        recs, preds = module.run(data, subjects, seeds, build_model, direction, split)
        records += recs
        predictions += preds
        print(f"{name}: {len(recs)} evaluations")

    results = pd.DataFrame(records)
    predictions = pd.concat(predictions, ignore_index=True)
    for frame in (results, predictions):
        frame["seed"] = frame["seed"].astype("Int64")
    return results, predictions


def save_tables(out_dir, results, predictions, tables, config):
    results.to_csv(out_dir / "results.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    confusion_counts(predictions).to_csv(out_dir / "confusion_counts.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    direction_dir = Path(args.out) / args.model / args.direction

    if args.plot_only:
        out_dir = direction_dir / args.tag if args.tag else latest_run(direction_dir)
        save_figures(out_dir)
        print("Figures saved to", out_dir.resolve())
        return

    subjects = parse_subjects(args.subjects)
    seeds = split_seeds(args.split, parse_seeds(args.seeds))
    if len(subjects) < 3:
        raise ValueError("Use at least 3 subjects for a meaningful cross-subject evaluation")
    out_dir = new_run_dir(direction_dir, args.tag, args.split)
    print(f"Split: {args.split}, seeds/folds: {seeds}")

    print(f"Loading Lee2019_MI subjects {subjects} (first run downloads data)")
    data = load_lee2019(subjects, fmin=args.fmin, fmax=args.fmax, resample=args.resample,
                        cache_dir=None if args.no_cache else DEFAULT_CACHE)
    print("X shape (trials, channels, samples):", data.X.shape)
    print("Class counts:", pd.Series(data.y).value_counts().to_dict())

    build_model = partial(MODELS[args.model], n_components=args.n_components)
    results, predictions = run_all(data, subjects, seeds, build_model, args.direction, args.split)
    tables = summarize(results)
    chance = within_chance_tests(predictions)
    if chance is not None:
        tables["within_chance_tests"] = chance

    split_text = {
        "random": "stratified random 75/25 split per seed",
        "chrono": "first 75% of trials in time order for training, last 25% for testing",
        "blocks": "4 contiguous time blocks; seed = index of the held-out block",
    }[args.split]
    config = {
        "run": out_dir.name, "model": args.model, "direction": args.direction, "split": args.split,
        "subjects": subjects, "seeds": seeds,
        "fmin": args.fmin, "fmax": args.fmax, "resample_hz": args.resample,
        "n_components": args.n_components,
        "dataset": "Lee2019_MI offline labeled trials",
        "within_session": f"source session(s) of the chosen direction, {split_text}",
        "cross_session": "train on the same 75% of the source session, test on all of the target session",
        "cross_subject": "leave-one-subject-out: train on all trials of the other selected subjects, test on the same held-out trials as within_session",
        "cross_subject_matched": "as cross_subject, but train on a class- and subject-stratified sample of other subjects' trials matching the within_session training size",
        "cross_subject_all_trials": "original definition: same LOSO model as cross_subject, tested on all trials of the held-out subject, seed-independent",
    }
    save_tables(out_dir, results, predictions, tables, config)
    save_figures(out_dir)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for title, name, digits in [("SUMMARY", "summary", 3), ("BY SESSION DIRECTION", "direction_summary", 3),
                                    ("PER SUBJECT", "subject_summary", 3), ("STATISTICS", "statistics", 4)]:
            print(f"\n{title}\n", main_view(tables[name]).round(digits).to_string(index=False))
        print("\nSupplementary pooled matched control (cross_subject_matched) is kept in the CSV files only")
        if chance is not None:
            print(f"\nWithin-session above chance (binomial p < 0.05): "
                  f"{int(chance['above_chance'].sum())}/{len(chance)} subjects")
    print("\nResults written to", out_dir.resolve())


if __name__ == "__main__":
    main()
