import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from data.loader import EEGData
from experiments.ica_ablation import prepare_conditions, run
from utils.metrics import paired_summary


class ICAAblationTests(unittest.TestCase):
    def test_prepare_conditions_preserves_trials_and_reports_components(self):
        rng = np.random.default_rng(4)
        sfreq = 128.0
        n_trials, n_samples = 24, 256
        ch_names = ("Fp1", "Fp2", "C3", "C4", "Cz")
        X = rng.normal(scale=2e-6, size=(n_trials, len(ch_names), n_samples))

        # A strong low-frequency frontal source gives the automatic detector a
        # deterministic ocular-like component without using class labels.
        time = np.arange(n_samples) / sfreq
        artifact = 35e-6 * np.sin(2 * np.pi * 2 * time)
        X[:, 0] += artifact
        X[:, 1] += .8 * artifact
        X[:, 2:] += .08 * artifact

        data = EEGData(
            X=X,
            y=np.tile(np.array(["left_hand", "right_hand"]), n_trials // 2),
            subject=np.repeat([1, 2], n_trials // 2),
            session=np.tile(np.repeat(["0", "1"], n_trials // 4), 2),
            sfreq=sfreq,
            ch_names=ch_names,
        )
        baseline, cleaned, report = prepare_conditions(
            data,
            analysis_fmin=8,
            analysis_fmax=32,
            random_state=0,
            ica_n_components=4,
            max_exclude=1,
        )

        self.assertEqual(baseline.X.shape, data.X.shape)
        self.assertEqual(cleaned.X.shape, data.X.shape)
        self.assertEqual(len(report), 4)
        report = pd.DataFrame(report)
        self.assertTrue((report["n_excluded"] <= 1).all())
        self.assertTrue((report["n_excluded"] == 1).any())
        self.assertIn("converged", report)
        self.assertTrue(np.isfinite(cleaned.X).all())
        self.assertFalse(np.shares_memory(data.X, cleaned.X))

    def test_summary_computes_paired_delta(self):
        results = pd.DataFrame([
            {"experiment": "cross_subject_all_trials", "test_subject": 1,
             "preprocessing": "baseline", "balanced_accuracy": .60},
            {"experiment": "cross_subject_all_trials", "test_subject": 1,
             "preprocessing": "ica", "balanced_accuracy": .65},
            {"experiment": "cross_subject_all_trials", "test_subject": 2,
             "preprocessing": "baseline", "balanced_accuracy": .55},
            {"experiment": "cross_subject_all_trials", "test_subject": 2,
             "preprocessing": "ica", "balanced_accuracy": .53},
        ])
        paired, statistics = paired_summary(results)

        np.testing.assert_allclose(paired["delta_ica"], [.05, -.02])
        self.assertAlmostEqual(statistics.loc[0, "delta_ica_mean"], .015)
        self.assertEqual(statistics.loc[0, "n_improved"], 1)
        self.assertEqual(statistics.loc[0, "n_worsened"], 1)

    def test_run_pairs_both_conditions_through_cross_subject(self):
        rng = np.random.default_rng(8)
        trials_per_recording = 8
        subject = np.repeat([1, 2, 3], 2 * trials_per_recording)
        session = np.tile(
            np.repeat(["0", "1"], trials_per_recording), 3
        )
        y = np.tile(
            np.array(["left_hand", "right_hand"] * (trials_per_recording // 2)), 6
        )
        X = rng.normal(size=(len(y), 3, 16))
        X[y == "right_hand", 0] += .5
        data = EEGData(
            X=X,
            y=y,
            subject=subject,
            session=session,
            sfreq=64.0,
            ch_names=("Fp1", "Fp2", "Cz"),
        )

        def build_model():
            flatten = FunctionTransformer(
                lambda values: values.reshape(len(values), -1)
            )
            return make_pipeline(flatten, GaussianNB())

        prepared = (
            data,
            replace(data, X=data.X * .99),
            [{"subject": 1, "session": "0", "n_excluded": 1}],
        )
        with patch("experiments.ica_ablation.prepare_conditions", return_value=prepared):
            records, predictions, components = run(
                data,
                subjects=[1, 2, 3],
                seeds=[0],
                build_model=build_model,
                direction="1to2",
                split="random",
            )

        self.assertEqual({row["preprocessing"] for row in records}, {"baseline", "ica"})
        self.assertEqual(
            set(pd.concat(predictions)["preprocessing"]), {"baseline", "ica"}
        )
        self.assertEqual(len(components), 1)


if __name__ == "__main__":
    unittest.main()
