"""Evaluate EEGNet on the Lee2019 MI time-block protocol."""

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from data.loader import load_lee2019, parse_subjects
from experiments.cross_subject import sample_matched
from experiments.within_session import split_seeds, split_session
from models.eegnet import fit_eegnet
from run_baselines import new_run_dir, save_tables
from utils.metrics import main_view, score_model, summarize, within_chance_tests
from utils.plotting import save_figures


SUBJECTS = "all"
DIRECTION = "both"
OUT = Path("results")
RUN_TAG = None
FMIN = 8.0
FMAX = 32.0

TRAINING_SEED = 0
VALIDATION_FRACTION = 0.2
SMALL_POOL_BATCH_SIZE = 8
SMALL_POOL_DROPOUT = 0.25
SMALL_POOL_LEARNING_RATE = 3e-3
LOSO_BATCH_SIZE = 64
LOSO_DROPOUT = 0.5
LOSO_LEARNING_RATE = 1e-3
MAX_EPOCHS = 60
PATIENCE = 10
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fit_training_pool(data, pool_ids, *, group_by_subject=False):
    if group_by_subject:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=VALIDATION_FRACTION, random_state=TRAINING_SEED
        )
        fit_positions, val_positions = next(splitter.split(pool_ids, groups=data.subject[pool_ids]))
        fit_ids, val_ids = pool_ids[fit_positions], pool_ids[val_positions]
        batch_size = LOSO_BATCH_SIZE
        dropout = LOSO_DROPOUT
        learning_rate = LOSO_LEARNING_RATE
    else:
        fit_ids, val_ids = train_test_split(
            pool_ids, test_size=VALIDATION_FRACTION, random_state=TRAINING_SEED,
            stratify=data.y[pool_ids],
        )
        batch_size = SMALL_POOL_BATCH_SIZE
        dropout = SMALL_POOL_DROPOUT
        learning_rate = SMALL_POOL_LEARNING_RATE
    model = fit_eegnet(
        data.X[fit_ids], data.y[fit_ids], data.X[val_ids], data.y[val_ids],
        device=DEVICE, seed=TRAINING_SEED, dropout=dropout, batch_size=batch_size,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    return model, {"n_fit": len(fit_ids), "n_val": len(val_ids)}


def run_evaluations(data, subjects, seeds, direction, split):
    records, predictions = [], []
    for index, subject in enumerate(subjects, 1):
        print(f"Evaluating subject {subject} ({index}/{len(subjects)})", flush=True)

        full_ids = data.ids(exclude_subject=subject)
        full_model, full_counts = fit_training_pool(data, full_ids, group_by_subject=True)
        record, preds = score_model(
            full_model, data, full_ids, data.ids(subject=subject),
            experiment="cross_subject_all_trials", seed=None, test_subject=subject,
            train_session="others", test_session="both",
        )
        record.update(full_counts)
        records.append(record)
        predictions.append(preds)

        for seed in seeds:
            for source, target in data.session_pairs(subject, direction):
                own_ids, test_ids = split_session(data, subject, source, seed, split)
                own_model, own_counts = fit_training_pool(data, own_ids)
                matched_ids = sample_matched(data, subject, len(own_ids), seed)
                matched_model, matched_counts = fit_training_pool(data, matched_ids)

                target_ids = data.ids(subject=subject, session=target)
                for experiment, model, pool, test, counts, train_session, test_session in (
                    ("within_session", own_model, own_ids, test_ids, own_counts, source, source),
                    ("cross_session", own_model, own_ids, target_ids, own_counts, source, target),
                    ("cross_subject", full_model, full_ids, test_ids, full_counts, "others", source),
                    ("cross_subject_matched", matched_model, matched_ids, test_ids,
                     matched_counts, "others", source),
                ):
                    record, preds = score_model(
                        model, data, pool, test, experiment=experiment, seed=seed,
                        test_subject=subject, train_session=train_session, test_session=test_session,
                    )
                    record.update(counts)
                    records.append(record)
                    predictions.append(preds)

    results = pd.DataFrame(records)
    predictions = pd.concat(predictions, ignore_index=True)
    for frame in (results, predictions):
        frame["seed"] = frame["seed"].astype("Int64")
    return results, predictions


def main():
    subjects = parse_subjects(SUBJECTS)
    split = "blocks"
    seeds = split_seeds(split, ())
    out_dir = new_run_dir(OUT / "eegnet" / DIRECTION, RUN_TAG, split)
    print(f"Split: {split}, folds: {seeds}")

    print(f"Loading subjects...{subjects}")
    data = load_lee2019(subjects, fmin=FMIN, fmax=FMAX, resample=128)
    print("Class counts:", pd.Series(data.y).value_counts().to_dict())

    results, predictions = run_evaluations(data, subjects, seeds, DIRECTION, split)
    tables = summarize(results)
    chance = within_chance_tests(predictions)
    if chance is not None:
        tables["within_chance_tests"] = chance

    config = {
        "run": out_dir.name, "model": "eegnet", "direction": DIRECTION, "split": split,
        "subjects": subjects, "seeds": seeds, "fmin": FMIN, "fmax": FMAX,
        "resample_hz": data.sfreq,
        "validation_fraction": VALIDATION_FRACTION,
        "training_seed": TRAINING_SEED, "device": DEVICE,
        "small_pool_batch_size": SMALL_POOL_BATCH_SIZE, "small_pool_dropout": SMALL_POOL_DROPOUT,
        "small_pool_learning_rate": SMALL_POOL_LEARNING_RATE,
        "loso_batch_size": LOSO_BATCH_SIZE, "loso_dropout": LOSO_DROPOUT,
        "loso_learning_rate": LOSO_LEARNING_RATE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
    }
    save_tables(out_dir, results, predictions, tables, config)
    save_figures(out_dir)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for title, name, digits in [("SUMMARY", "summary", 3), ("BY SESSION DIRECTION", "direction_summary", 3),
                                    ("PER SUBJECT", "subject_summary", 3), ("STATISTICS", "statistics", 4)]:
            print(f"\n{title}\n", main_view(tables[name]).round(digits).to_string(index=False))
    print("\nResults written to", out_dir.resolve())


if __name__ == "__main__":
    main()
