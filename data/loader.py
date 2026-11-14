import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery

ALL_SUBJECTS = tuple(range(1, 55))
DEFAULT_CACHE = Path(mne.get_config("MNE_DATA") or Path.home() / "mne_data") / "ECE590-cache"
DIRECTIONS = {"1to2": [(0, 1)], "2to1": [(1, 0)], "both": [(0, 1), (1, 0)]}

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", message="Montage name 'standard_1005' is deprecated",
                        category=FutureWarning)


@dataclass(frozen=True)
class EEGData:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray

    def ids(self, subject=None, session=None, exclude_subject=None):
        mask = np.ones(len(self.y), dtype=bool)
        if subject is not None:
            mask &= self.subject == subject
        if session is not None:
            mask &= self.session == session
        if exclude_subject is not None:
            mask &= self.subject != exclude_subject
        return np.flatnonzero(mask)

    def sessions(self, subject):
        return sorted(np.unique(self.session[self.subject == subject]).tolist())

    def session_pairs(self, subject, direction):
        sessions = self.sessions(subject)
        return [(sessions[source], sessions[target]) for source, target in DIRECTIONS[direction]]


def parse_subjects(spec):
    if spec.strip().lower() == "all":
        return list(ALL_SUBJECTS)
    subjects = [int(s) for s in spec.split(",") if s.strip()]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Use unique subject IDs, e.g. 1,2,3,4,5")
    if not set(subjects) <= set(ALL_SUBJECTS):
        raise ValueError("Lee2019_MI subjects must be from 1 to 54")
    return subjects


def download_lee2019(subjects, retries=3):
    dataset = Lee2019_MI(train_run=True, test_run=False)
    failed = []
    for subject in subjects:
        for attempt in range(1, retries + 1):
            try:
                dataset.data_path(subject)
                print(f"subject {subject}: ok", flush=True)
                break
            except Exception as error:
                print(f"subject {subject}: attempt {attempt} failed: {error}", flush=True)
        else:
            failed.append(subject)
    print("failed subjects:", failed, flush=True)
    return failed


def _load_subject(subject, fmin, fmax, resample):
    dataset = Lee2019_MI(train_run=True, test_run=False)
    paradigm = LeftRightImagery(fmin=fmin, fmax=fmax, resample=resample)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    meta = meta.reset_index(drop=True)
    if not len(X) == len(y) == len(meta):
        raise RuntimeError(f"Subject {subject}: epoch, label and metadata counts do not match")
    return {
        "X": np.asarray(X),
        "y": np.asarray(y, dtype=str),
        "subject": meta["subject"].to_numpy(dtype=int),
        "session": np.asarray(meta["session"], dtype=str),
    }


def _cached_subject(subject, fmin, fmax, resample, cache_dir):
    if cache_dir is None:
        return _load_subject(subject, fmin, fmax, resample)
    path = Path(cache_dir) / f"lee2019_mi_{fmin:g}-{fmax:g}Hz_{resample}Hz" / f"subject_{subject:02d}.npz"
    if path.exists():
        with np.load(path) as cached:
            return {key: cached[key] for key in cached.files}
    arrays = _load_subject(subject, fmin, fmax, resample)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)
    return arrays


def load_lee2019(subjects, fmin=8.0, fmax=32.0, resample=128, cache_dir=DEFAULT_CACHE):
    parts = []
    for i, subject in enumerate(subjects, 1):
        parts.append(_cached_subject(subject, fmin, fmax, resample, cache_dir))
        print(f"\rLoaded {i}/{len(subjects)} subjects", end="", flush=True)
    print()

    data = EEGData(**{key: np.concatenate([p[key] for p in parts]) for key in parts[0]})
    for subject in subjects:
        sessions = data.sessions(subject)
        if len(sessions) != 2:
            raise RuntimeError(f"Subject {subject}: expected 2 sessions, got {sessions}")
    return data
