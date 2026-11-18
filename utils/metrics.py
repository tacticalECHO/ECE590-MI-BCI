import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

CHANCE = 0.5
N_BOOTSTRAP = 10000

SESSION_CONDITIONS = ["within_session", "cross_session", "cross_subject", "cross_subject_matched"]
SESSION_GAPS = ["G_session", "G_subject", "G_subject_matched"]
CONDITIONS = SESSION_CONDITIONS + ["cross_subject_all_trials"]
GAPS = SESSION_GAPS + ["G_subject_all_trials"]
SUPPLEMENTARY = ["cross_subject_matched", "G_subject_matched"]
MAIN_CONDITIONS = [c for c in CONDITIONS if c not in SUPPLEMENTARY]
MAIN_GAPS = [g for g in GAPS if g not in SUPPLEMENTARY]


def main_view(table):
    if "metric" in table.columns:
        return table[~table["metric"].isin(SUPPLEMENTARY)]
    return table[[c for c in table.columns if not any(c.startswith(s) for s in SUPPLEMENTARY)]]


def scores(y_true, y_pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def fit_model(build_model, data, train_ids):
    model = build_model()
    model.fit(data.X[train_ids], data.y[train_ids])
    return model


def score_model(model, data, train_ids, test_ids, **info):
    y_true = data.y[test_ids]
    y_pred = model.predict(data.X[test_ids])
    record = {**info, "n_train": len(train_ids), "n_test": len(test_ids), **scores(y_true, y_pred)}
    predictions = pd.DataFrame({
        **info,
        "trial_index": test_ids,
        "true_label": y_true.astype(str),
        "predicted_label": y_pred.astype(str),
    })
    return record, predictions


def evaluate_split(build_model, data, train_ids, test_ids, **info):
    return score_model(fit_model(build_model, data, train_ids), data, train_ids, test_ids, **info)


def confusion_counts(predictions):
    keys = ["experiment", "test_subject", "true_label", "predicted_label"]
    if "preprocessing" in predictions:
        keys.insert(0, "preprocessing")
    return (predictions
            .groupby(keys)
            .size()
            .rename("count")
            .reset_index())


def _pick(results, experiment, session_column, extra=()):
    columns = ["seed", "test_subject", session_column, *extra, "balanced_accuracy"]
    rows = results.loc[results["experiment"] == experiment, columns]
    return rows.rename(columns={session_column: "train_session", "balanced_accuracy": experiment})


def _session_gaps(results):
    keys = ["seed", "test_subject", "train_session"]
    gaps = _pick(results, "within_session", "train_session")
    gaps = gaps.merge(_pick(results, "cross_session", "train_session", ["test_session"]),
                      on=keys, validate="one_to_one")
    for experiment in ["cross_subject", "cross_subject_matched"]:
        gaps = gaps.merge(_pick(results, experiment, "test_session"), on=keys, validate="one_to_one")
    gaps["G_session"] = gaps["within_session"] - gaps["cross_session"]
    gaps["G_subject"] = gaps["within_session"] - gaps["cross_subject"]
    gaps["G_subject_matched"] = gaps["within_session"] - gaps["cross_subject_matched"]
    return gaps[keys + ["test_session"] + SESSION_CONDITIONS + SESSION_GAPS]


def _seed_gaps(session_gaps, results):
    gaps = session_gaps.groupby(["seed", "test_subject"])[SESSION_CONDITIONS + SESSION_GAPS].mean()
    all_trials = (results[results["experiment"] == "cross_subject_all_trials"]
                  .set_index("test_subject")["balanced_accuracy"])
    gaps["cross_subject_all_trials"] = gaps.index.get_level_values("test_subject").map(all_trials).to_numpy()
    gaps["G_subject_all_trials"] = gaps["within_session"] - gaps["cross_subject_all_trials"]
    return gaps[CONDITIONS + GAPS]


def _direction_summary(session_gaps):
    columns = SESSION_CONDITIONS + SESSION_GAPS
    per_subject = (session_gaps
                   .groupby(["train_session", "test_session", "test_subject"])[columns]
                   .mean())
    summary = per_subject.groupby(level=["train_session", "test_session"]).agg(["mean", "std"])
    summary.columns = [f"{name}_{'mean' if stat == 'mean' else 'sd_across_subjects'}"
                       for name, stat in summary.columns]
    return summary.reset_index()


def _bootstrap_ci(values, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def paired_summary(results):
    """Summarize paired baseline/ICA balanced-accuracy differences."""
    paired = (results
              .groupby(["experiment", "test_subject", "preprocessing"])
              ["balanced_accuracy"].mean()
              .unstack("preprocessing")[["baseline", "ica"]]
              .reset_index())
    paired["delta_ica"] = paired["ica"] - paired["baseline"]

    rows = []
    for experiment, group in paired.groupby("experiment"):
        delta = group["delta_ica"].to_numpy()
        low, high = _bootstrap_ci(delta)
        p_value = (wilcoxon(delta).pvalue
                   if len(delta) > 1 and np.any(delta != 0) else np.nan)
        rows.append({
            "experiment": experiment,
            "n_subjects": len(delta),
            "baseline_mean": group["baseline"].mean(),
            "ica_mean": group["ica"].mean(),
            "delta_ica_mean": delta.mean(),
            "delta_ica_median": np.median(delta),
            "delta_ci95_low": low,
            "delta_ci95_high": high,
            "n_improved": int((delta > 0).sum()),
            "n_worsened": int((delta < 0).sum()),
            "n_unchanged": int((delta == 0).sum()),
            "wilcoxon_p": p_value,
        })
    return paired, pd.DataFrame(rows)


def group_statistics(subject_summary):
    rows = []
    for metric in CONDITIONS + GAPS:
        values = subject_summary[metric].dropna().to_numpy()
        reference = 0.0 if metric in GAPS else CHANCE
        alternative = "two-sided" if metric in GAPS else "greater"
        low, high = _bootstrap_ci(values)
        differences = values - reference
        p_value = (wilcoxon(differences, alternative=alternative).pvalue
                   if np.any(differences != 0) and len(values) > 1 else np.nan)
        rows.append({
            "metric": metric, "n_subjects": len(values), "reference": reference,
            "mean": values.mean(), "median": np.median(values),
            "ci95_low": low, "ci95_high": high,
            "n_above_reference": int((values > reference).sum()),
            "wilcoxon_alternative": alternative, "wilcoxon_p": p_value,
        })
    return pd.DataFrame(rows)


def within_chance_tests(predictions, alpha=0.05):
    within = predictions[predictions["experiment"] == "within_session"]
    if within.duplicated(["test_subject", "trial_index"]).any():
        return None
    rows = []
    for subject, group in within.groupby("test_subject"):
        correct = int((group["true_label"] == group["predicted_label"]).sum())
        test = binomtest(correct, len(group), CHANCE, alternative="greater")
        rows.append({"test_subject": subject, "n_trials": len(group), "n_correct": correct,
                     "accuracy": correct / len(group), "binomial_p": test.pvalue,
                     "above_chance": test.pvalue < alpha})
    return pd.DataFrame(rows)


def summarize(results):
    session_gaps = _session_gaps(results)
    seed_gaps = _seed_gaps(session_gaps, results)

    by_subject = seed_gaps.groupby(level="test_subject")
    subject_summary = by_subject.mean().join(by_subject.std(ddof=1).add_suffix("_seed_sd"))
    subject_summary["n_seeds"] = by_subject.size()

    seed_summary = seed_gaps.groupby(level="seed").mean()

    summary = pd.DataFrame({
        "mean": subject_summary[CONDITIONS + GAPS].mean(),
        "sd_across_subjects": subject_summary[CONDITIONS + GAPS].std(ddof=1),
        "sd_of_cohort_mean_across_seeds": seed_summary.std(ddof=1),
        "min_cohort_mean_across_seeds": seed_summary.min(),
        "max_cohort_mean_across_seeds": seed_summary.max(),
    })
    other = results.groupby("experiment")[["accuracy", "macro_f1"]].mean().add_prefix("mean_")
    summary = summary.join(other)
    summary["n_subjects"] = len(subject_summary)
    summary["n_seeds"] = len(seed_summary)
    summary.index.name = "metric"

    return {
        "session_gaps": session_gaps,
        "seed_gaps": seed_gaps.reset_index(),
        "subject_summary": subject_summary.reset_index(),
        "seed_summary": seed_summary.reset_index(),
        "direction_summary": _direction_summary(session_gaps),
        "summary": summary.reset_index(),
        "statistics": group_statistics(subject_summary),
    }
