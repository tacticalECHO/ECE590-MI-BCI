import numpy as np
from sklearn.model_selection import train_test_split

from experiments.within_session import split_session
from utils.metrics import fit_model, score_model


def sample_matched(data, held_out, n_train, seed):
    candidates = data.ids(exclude_subject=held_out)
    labels = data.y[candidates].astype(str)
    strata = np.char.add(np.char.add(data.subject[candidates].astype(str), "_"), labels)
    if n_train < len(np.unique(strata)):
        strata = labels
    train_ids, _ = train_test_split(candidates, train_size=n_train, random_state=seed, stratify=strata)
    return np.sort(train_ids)


def run(data, subjects, seeds, build_model, direction="both", split="random"):
    records, predictions = [], []
    for held_out in subjects:
        full_ids = data.ids(exclude_subject=held_out)
        full_model = fit_model(build_model, data, full_ids)
        record, preds = score_model(
            full_model, data, full_ids, data.ids(subject=held_out),
            experiment="cross_subject_all_trials", seed=None, test_subject=held_out,
            train_session="others", test_session="both",
        )
        records.append(record)
        predictions.append(preds)
        for seed in seeds:
            for session, _ in data.session_pairs(held_out, direction):
                own_train_ids, test_ids = split_session(data, held_out, session, seed, split)
                matched_ids = sample_matched(data, held_out, len(own_train_ids), seed)
                for experiment, model, train_ids in [
                    ("cross_subject", full_model, full_ids),
                    ("cross_subject_matched", fit_model(build_model, data, matched_ids), matched_ids),
                ]:
                    record, preds = score_model(
                        model, data, train_ids, test_ids,
                        experiment=experiment, seed=seed, test_subject=held_out,
                        train_session="others", test_session=session,
                    )
                    records.append(record)
                    predictions.append(preds)
    return records, predictions
