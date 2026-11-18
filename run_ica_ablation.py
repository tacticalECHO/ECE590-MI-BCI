"""Command-line entry point for the paired cross-subject ICA ablation."""

import argparse
from functools import partial
from pathlib import Path

import pandas as pd

from data.loader import DEFAULT_CACHE, DIRECTIONS, load_lee2019, parse_subjects
from experiments import ica_ablation
from experiments.within_session import SPLITS, split_seeds
from models.csp_lda import build_csp_lda
from run_baselines import new_run_dir, parse_seeds, save_tables
from utils.metrics import paired_summary
from utils.plotting import plot_paired_effects


def _ica_components(value):
    number = float(value)
    if number >= 1 and number.is_integer():
        return int(number)
    if 0 < number <= 1:
        return number
    raise argparse.ArgumentTypeError("use an integer count or a variance fraction in (0, 1]")


def parse_args():
    parser = argparse.ArgumentParser(description="Lee2019 cross-subject ICA ablation")
    parser.add_argument("--subjects", default="1,2,3,4,5",
                        help="comma-separated IDs or 'all'")
    parser.add_argument("--seeds", default="0,1,2,3",
                        help="random split seeds; blocks always uses folds 0,1,2,3")
    parser.add_argument("--direction", choices=list(DIRECTIONS), default="both")
    parser.add_argument("--split", choices=SPLITS, default="blocks")
    parser.add_argument("--out", default="results/ica_ablation/csp_lda")
    parser.add_argument("--tag", help="run folder; defaults to a timestamp")
    parser.add_argument("--fit-fmin", type=float, default=1.0,
                        help="broad-band lower edge used to fit ICA")
    parser.add_argument("--fit-fmax", type=float, default=40.0,
                        help="broad-band upper edge used to fit ICA")
    parser.add_argument("--fmin", type=float, default=8.0,
                        help="classification-band lower edge")
    parser.add_argument("--fmax", type=float, default=32.0,
                        help="classification-band upper edge")
    parser.add_argument("--resample", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=6,
                        help="number of CSP components")
    parser.add_argument("--ica-components", type=_ica_components, default=.99,
                        help="ICA component count or explained-variance fraction")
    parser.add_argument("--max-exclude", type=int, default=2,
                        help="maximum ocular components removed per recording")
    parser.add_argument("--eog-correlation-threshold", type=float, default=.3,
                        help="absolute component/Fp1-Fp2 correlation threshold")
    parser.add_argument("--ica-seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.fit_fmin < args.fmin < args.fmax <= args.fit_fmax:
        raise ValueError("Require fit_fmin < fmin < fmax <= fit_fmax")
    subjects = parse_subjects(args.subjects)
    if len(subjects) < 3:
        raise ValueError("Use at least 3 subjects for cross-subject evaluation")
    seeds = split_seeds(args.split, parse_seeds(args.seeds))

    out_dir = new_run_dir(Path(args.out) / args.direction, args.tag, args.split)

    print(f"Loading broad-band Lee2019_MI data for subjects {subjects}")
    data = load_lee2019(
        subjects,
        fmin=args.fit_fmin,
        fmax=args.fit_fmax,
        resample=args.resample,
        cache_dir=None if args.no_cache else DEFAULT_CACHE,
    )
    print("X shape (trials, channels, samples):", data.X.shape)

    build_model = partial(build_csp_lda, n_components=args.n_components)
    records, prediction_frames, component_rows = ica_ablation.run(
        data,
        subjects,
        seeds,
        build_model,
        direction=args.direction,
        split=args.split,
        analysis_fmin=args.fmin,
        analysis_fmax=args.fmax,
        random_state=args.ica_seed,
        ica_n_components=args.ica_components,
        eog_threshold=args.eog_correlation_threshold,
        max_exclude=args.max_exclude,
    )
    results = pd.DataFrame(records)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    for frame in (results, predictions):
        frame["seed"] = frame["seed"].astype("Int64")
    subject_summary, statistics = paired_summary(results)
    tables = {"subject_summary": subject_summary, "statistics": statistics,
              "ica_components": pd.DataFrame(component_rows)}

    config = {**vars(args),
        "run": out_dir.name,
        "subjects": subjects,
        "seeds": seeds,
        "ica_scope": "fitted independently to each subject/session without labels",
    }
    save_tables(out_dir, results, predictions, tables, config)
    plot_paired_effects(subject_summary, out_dir / "ica_effects.png")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nICA ABLATION STATISTICS\n")
        print(tables["statistics"].round(4).to_string(index=False))
    print("\nResults written to", out_dir.resolve())


if __name__ == "__main__":
    main()
