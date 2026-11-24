# ECE590 Motor-Imagery BCI Generalization

Baselines for cross-session and cross-subject generalization of left/right-hand motor imagery on the Lee2019 (OpenBMI) dataset, loaded through MOABB.

## Setup

```bash
conda create -n mibci python=3.11
conda activate mibci
pip install -r requirements.txt
```

Raw data (about 61 GB for all 54 subjects) is downloaded by MOABB on first use. Preprocessed epochs are cached per subject under `<MNE_DATA>/ECE590-cache`.

## Usage

```bash
python run_baselines.py --subjects all --direction both --split random --tag all54_random
python run_baselines.py --subjects all --direction both --split blocks --tag all54_blocks
python run_baselines.py --direction both --tag all54_blocks --plot-only
```

Paired cross-subject ICA ablation (1--40 Hz ICA fit, then identical 8--32 Hz
CSP+LDA evaluation for the baseline and cleaned conditions):

```bash
python run_ica_ablation.py --subjects all --direction both --split blocks --ica-seed 0 --tag all54_ica
python run_ica_ablation.py --subjects all --direction both --split blocks --ica-seed 1 --tag all54_ica_seed1
python run_ica_ablation.py --subjects all --direction both --split blocks --ica-seed 2 --tag all54_ica_seed2
```

Lee2019_MI has no dedicated EOG channel, so the ICA experiment uses Fp1/Fp2 as
pre-declared ocular proxies and removes at most two automatically flagged
components per subject/session recording (absolute correlation threshold 0.3).
This is label-free recording-level
preprocessing; the held-out recording contributes unlabeled data to its own ICA
decomposition. The run config records this scope explicitly.

### ICA stability result

The full 54-subject experiment was repeated with three FastICA initialization
seeds. Delta is ICA minus baseline balanced accuracy.

| ICA seed | Cross-subject delta | Wilcoxon p | All-trials delta | Wilcoxon p |
|---:|---:|---:|---:|---:|
| 0 | -0.00141 | 0.0939 | -0.00120 | 0.1510 |
| 1 | +0.00058 | 0.2192 | +0.00083 | 0.2123 |
| 2 | -0.00279 | 0.0158 | -0.00241 | 0.0178 |

Across the three ICA seeds, the mean delta was -0.00121 for `cross_subject`
and -0.00093 for `cross_subject_all_trials`. All 108 subject/session ICA fits
converged for every seed. The direction varies slightly with initialization,
but ICA does not provide a stable improvement over the paired baseline. Seed 2
shows a small negative signed-rank result, while its bootstrap confidence
interval for the mean still includes zero.

| Option | Values |
|---|---|
| `--subjects` | `all` or comma-separated IDs (1-54) |
| `--direction` | `1to2`, `2to1`, `both` |
| `--split` | `random` (stratified 75/25 per seed), `chrono` (last 25% held out), `blocks` (4 contiguous time blocks) |
| `--seeds` | comma-separated split seeds, default `0..9` |
| `--ica-seed` | FastICA initialization seed (ICA runner only) |
| `--tag` | run folder name; existing runs are never overwritten |

Results are written to `results/<model>/<direction>/<tag>/`.
ICA-ablation results are written to
`results/ica_ablation/csp_lda/<direction>/<tag>/`.

### EEGNet

EEGNet-8,2 requires PyTorch in the project environment.

```bash
python run_eegnet.py
```

By default, it evaluates all 54 Lee2019 subjects in both session directions using four contiguous time blocks.
It reports balanced accuracy for the conditions below, including the supplementary matched-training-size control.
Edit the constants at the top of `run_eegnet.py` to change the run settings.
Tables and figures are saved under `results/eegnet/<direction>/<tag>/`.

## Conditions

| Name | Training data | Test data |
|---|---|---|
| `within_session` | 75% of the source session | remaining 25% of the source session |
| `cross_session` | same 75% of the source session | all trials of the target session |
| `cross_subject` | all trials of the other subjects (LOSO) | same 25% as `within_session` |
| `cross_subject_all_trials` | all trials of the other subjects (LOSO) | all trials of the held-out subject |

Gaps: `G_session = within_session - cross_session`, `G_subject = within_session - cross_subject`. The primary metric is balanced accuracy. Group statistics use a Wilcoxon signed-rank test and a bootstrap 95% CI over subjects; with `--split blocks`, per-subject binomial tests against chance are added.

`cross_subject_matched` (a pooled sample of other subjects matched to the within-session training size) is a supplementary control kept in the CSV files only.

## Results

CSP (6 components) + LDA, all 54 subjects, both session directions. Values are the mean over subjects of balanced accuracy (conditions) or balanced-accuracy difference (gaps), with bootstrap 95% CIs over subjects. Wilcoxon p-values are one-sided against 0.5 for conditions and two-sided against 0 for gaps.

| Metric | Random split, 10 seeds | p | Time blocks, 4 folds | p |
|---|---|---|---|---|
| `within_session` | 0.707 [0.669, 0.747] | < 1e-9 | 0.711 [0.673, 0.750] | < 1e-9 |
| `cross_session` | 0.618 [0.583, 0.656] | < 1e-9 | 0.617 [0.582, 0.656] | < 1e-8 |
| `cross_subject` | 0.646 [0.614, 0.680] | < 1e-9 | 0.647 [0.616, 0.682] | < 1e-9 |
| `G_session` | 0.088 [0.069, 0.109] | < 1e-9 | 0.093 [0.073, 0.115] | < 1e-8 |
| `G_subject` | 0.061 [0.030, 0.093] | 0.001 | 0.063 [0.033, 0.095] | < 0.001 |

| Direction | `within_session` | `cross_session` | `G_session` | `G_subject` |
|---|---|---|---|---|
| Session 1 → 2 | 0.694 | 0.624 | 0.070 | 0.052 |
| Session 2 → 1 | 0.727 | 0.611 | 0.116 | 0.075 |

Direction values are from the time-block split.

- Both gaps are positive and significant. Generalizing to a new session costs more than generalizing to a new subject when the cross-subject model is trained on all 53 other subjects.
- The cross-subject model uses about 140 times more training trials than the within-session model, so `G_subject` is a conservative estimate of the subject shift.
- Random and time-block splits agree (per-subject within-session correlation 0.97), so the random split does not inflate within-session accuracy through temporal leakage.
- Within-session accuracy is significantly above chance (binomial test, p < 0.05) for 43 of 54 subjects.

Full tables and figures are in `results/csp_lda/both/all54_random` and `results/csp_lda/both/all54_blocks`.

## Layout

```
data/loader.py              Lee2019 loading, caching, session directions
models/csp_lda.py           CSP + LDA pipeline
models/eegnet.py            PyTorch EEGNet-8,2 model and training
experiments/                within-session, cross-session, cross-subject runs
experiments/ica_ablation.py paired recording-level ICA ablation
utils/metrics.py            scores, gap tables, statistics
utils/plotting.py           figures
run_baselines.py            command-line entry point
run_eegnet.py               EEGNet time-block evaluation entry point
run_ica_ablation.py         ICA ablation command-line entry point
results/                    saved runs
```
