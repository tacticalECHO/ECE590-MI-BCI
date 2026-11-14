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

| Option | Values |
|---|---|
| `--subjects` | `all` or comma-separated IDs (1-54) |
| `--direction` | `1to2`, `2to1`, `both` |
| `--split` | `random` (stratified 75/25 per seed), `chrono` (last 25% held out), `blocks` (4 contiguous time blocks) |
| `--seeds` | comma-separated split seeds, default `0..9` |
| `--tag` | run folder name; existing runs are never overwritten |

Results are written to `results/<model>/<direction>/<tag>/`.

## Conditions

| Name | Training data | Test data |
|---|---|---|
| `within_session` | 75% of the source session | remaining 25% of the source session |
| `cross_session` | same 75% of the source session | all trials of the target session |
| `cross_subject` | all trials of the other subjects (LOSO) | same 25% as `within_session` |
| `cross_subject_all_trials` | all trials of the other subjects (LOSO) | all trials of the held-out subject |

Gaps: `G_session = within_session - cross_session`, `G_subject = within_session - cross_subject`. The primary metric is balanced accuracy. Group statistics use a Wilcoxon signed-rank test and a bootstrap 95% CI over subjects; with `--split blocks`, per-subject binomial tests against chance are added.

`cross_subject_matched` (a pooled sample of other subjects matched to the within-session training size) is a supplementary control kept in the CSV files only.

## Layout

```
data/loader.py              Lee2019 loading, caching, session directions
models/csp_lda.py           CSP + LDA pipeline
models/eegnet.py            EEGNet (not implemented yet)
experiments/                within-session, cross-session, cross-subject runs
utils/metrics.py            scores, gap tables, statistics
utils/plotting.py           figures
run_baselines.py            command-line entry point
results/                    saved runs
```
