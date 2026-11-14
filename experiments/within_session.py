import numpy as np
from sklearn.model_selection import train_test_split

from utils.metrics import evaluate_split

TEST_SIZE = 0.25
N_BLOCKS = round(1 / TEST_SIZE)
SPLITS = ("random", "chrono", "blocks")


def split_seeds(split, seeds):
    if split == "chrono":
        return [0]
    if split == "blocks":
        return list(range(N_BLOCKS))
    return seeds


def split_session(data, subject, session, seed, split="random"):
    ids = data.ids(subject=subject, session=session)
    if split == "random":
        return train_test_split(ids, test_size=TEST_SIZE, random_state=seed, stratify=data.y[ids])
    if split == "chrono":
        n_test = round(len(ids) * TEST_SIZE)
        return ids[:-n_test], ids[-n_test:]
    if split == "blocks":
        blocks = np.array_split(ids, N_BLOCKS)
        train_ids = np.concatenate([b for i, b in enumerate(blocks) if i != seed])
        return train_ids, blocks[seed]
    raise ValueError(f"Unknown split {split!r}; use one of {SPLITS}")


def run(data, subjects, seeds, build_model, direction="both", split="random"):
    records, predictions = [], []
    for seed in seeds:
        for subject in subjects:
            for session, _ in data.session_pairs(subject, direction):
                train_ids, test_ids = split_session(data, subject, session, seed, split)
                record, preds = evaluate_split(
                    build_model, data, train_ids, test_ids,
                    experiment="within_session", seed=seed, test_subject=subject,
                    train_session=session, test_session=session,
                )
                records.append(record)
                predictions.append(preds)
    return records, predictions
