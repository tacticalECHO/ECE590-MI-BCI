"""Paired recording-level ICA ablation for cross-subject evaluation."""

import warnings
from dataclasses import replace

import mne
import numpy as np
from mne.preprocessing import ICA
from sklearn.exceptions import ConvergenceWarning

from experiments import cross_subject

PROXY_CHANNELS = ("Fp1", "Fp2")


def _epochs(X, data):
    info = mne.create_info(list(data.ch_names), data.sfreq, ch_types="eeg")
    return mne.EpochsArray(X, info, tmin=0, baseline=None, verbose=False)


def _artifact_components(ica, epochs, threshold, max_exclude):
    proxies = [name for name in PROXY_CHANNELS if name in epochs.ch_names]
    if not proxies:
        raise ValueError(f"ICA requires one of the proxy channels {PROXY_CHANNELS}")

    suggested = set()
    scores = np.zeros(ica.n_components_)
    for channel in proxies:
        indices, channel_scores = ica.find_bads_eog(
            epochs, ch_name=channel, threshold=threshold,
            measure="correlation", verbose=False,
        )
        suggested.update(indices)
        scores = np.maximum(scores, np.abs(channel_scores))

    ranked = [int(index) for index in
              sorted(suggested, key=scores.__getitem__, reverse=True)]
    return ranked[:max_exclude], ranked, scores


def prepare_conditions(
    data,
    analysis_fmin=8,
    analysis_fmax=32,
    random_state=0,
    ica_n_components=.99,
    eog_threshold=.3,
    max_exclude=2,
):
    """Return equally filtered baseline/ICA data and a component report."""
    if not 0 < analysis_fmin < analysis_fmax < data.sfreq / 2:
        raise ValueError("Analysis band must satisfy 0 < fmin < fmax < Nyquist")
    if max_exclude < 0:
        raise ValueError("max_exclude must be non-negative")

    baseline = mne.filter.filter_data(
        data.X, data.sfreq, analysis_fmin, analysis_fmax, copy=True, verbose=False
    )
    cleaned = np.empty_like(data.X)
    report = []

    for subject in sorted(np.unique(data.subject)):
        for session in data.sessions(subject):
            ids = data.ids(subject=subject, session=session)
            epochs = _epochs(data.X[ids], data)
            ica = ICA(
                n_components=ica_n_components, method="fastica",
                random_state=random_state, max_iter="auto",
            )
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.filterwarnings(
                    "ignore", message="The data has not been high-pass filtered"
                )
                ica.fit(epochs, picks="eeg", verbose=False)

            excluded, suggested, scores = _artifact_components(
                ica, epochs, eog_threshold, max_exclude
            )
            ica.exclude = excluded
            filtered = ica.apply(epochs.copy(), verbose=False).get_data(copy=False)
            cleaned[ids] = mne.filter.filter_data(
                filtered, data.sfreq, analysis_fmin, analysis_fmax,
                copy=True, verbose=False,
            )
            report.append({
                "subject": int(subject),
                "session": str(session),
                "n_trials": len(ids),
                "n_ica_components": int(ica.n_components_),
                "n_iterations": int(ica.n_iter_),
                "converged": not any(
                    issubclass(item.category, ConvergenceWarning)
                    for item in fit_warnings
                ),
                "n_suggested": len(suggested),
                "n_excluded": len(excluded),
                "suggested_components": ";".join(map(str, suggested)),
                "excluded_components": ";".join(map(str, excluded)),
                "excluded_scores": ";".join(f"{scores[i]:.6g}" for i in excluded),
            })
            print(f"ICA subject {subject}, session {session}: {excluded or 'none'}")

    return replace(data, X=baseline), replace(data, X=cleaned), report


def run(data, subjects, seeds, build_model, direction="both", split="blocks", **kwargs):
    """Evaluate the same cross-subject splits before and after ICA."""
    baseline, cleaned, components = prepare_conditions(data, **kwargs)
    records, predictions = [], []
    for name, condition in (("baseline", baseline), ("ica", cleaned)):
        condition_records, condition_predictions = cross_subject.run(
            condition, subjects, seeds, build_model, direction, split
        )
        records.extend({**record, "preprocessing": name}
                       for record in condition_records)
        for frame in condition_predictions:
            frame["preprocessing"] = name
        predictions.extend(condition_predictions)
        print(f"{name}: {len(condition_records)} cross-subject evaluations")
    return records, predictions, components
