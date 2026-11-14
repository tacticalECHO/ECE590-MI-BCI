from experiments.within_session import split_session
from utils.metrics import evaluate_split


def run(data, subjects, seeds, build_model, direction="both", split="random"):
    records, predictions = [], []
    for seed in seeds:
        for subject in subjects:
            for source, target in data.session_pairs(subject, direction):
                train_ids, _ = split_session(data, subject, source, seed, split)
                test_ids = data.ids(subject=subject, session=target)
                record, preds = evaluate_split(
                    build_model, data, train_ids, test_ids,
                    experiment="cross_session", seed=seed, test_subject=subject,
                    train_session=source, test_session=target,
                )
                records.append(record)
                predictions.append(preds)
    return records, predictions
