# DASE7506 MP2 - Zhang Yaxing (3036707458)

Frozen full-test selection accuracy: **82.65524625267666** (386/467).
Submitted selector: linear ListNet over 1047 group-comparative features
(23 handcrafted + 1024-d unit trace embedding), ten-seed weight average.
Validation 92.35588972431077 was used for development/selection only.

## Direct evaluation without retraining

Clone the repository (it contains the supplied data and the frozen checkpoint).
All commands below run from `code/`. Use Python 3.12+ (measured: 3.13.9).

~~~sh
python -m venv .venv
~~~

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`
Linux activation: `source .venv/bin/activate`

~~~sh
python -m pip install -r requirements.txt
python check_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python predict.py --method submission.py --checkpoint runs/model_bundle \
      --split test --output runs/test-predictions.json
python score_predictions.py --split test \
    --predictions runs/test-predictions.json --output runs/test-score.json
~~~

Expected test accuracy: 82.65524625267666; validation (`--split validation`)
92.35588972431077. Small hardware differences are not expected: inference is pure
NumPy and the checkpoint is frozen.
Checkpoint SHA256: e018153996394d3e2c83a5e5d1d1ad5734c16b4c3b98c16f1cfe3ceeda1cbcde
(`code/runs/model_bundle/model.npz`, 26,347 bytes).
Data/parser/scorer are supplied and unchanged; inference requires no network.
Without `--checkpoint`, `predict()` degrades to the deterministic majority-vote
baseline (86.2155 / 69.1649), a legal training-free submission on its own.
FILE_MANIFEST.json covers exact source/evidence bytes; ASSET_INVENTORY.json lists
inference assets.

## Environment and retraining

Measured environment: Windows 11, Python 3.13.9, NumPy 2.5.4; AMD Ryzen 7 9800X3D,
four threads (Matplotlib 3.11.2 only for report figures).
Reference machine per guide: Xeon Platinum 8457C, four threads, 30 s budget.

~~~sh
python train.py --variant emb --seeds 10 --out runs/model_bundle
~~~

lr 0.05, l2 3e-3, 800 epochs, 256 groups/batch, Adam (0.9, 0.999); seeds 0..9 via
`numpy.default_rng`; the ten weight vectors are averaged (exact logit ensemble for
linear models). ~192 s CPU. Retraining is deterministic: `model.npz` reproduces
bit-identically (SHA-256 verified) with identical per-pattern validation breakdowns.
Training uses only the supplied train split (4,755 groups); labels come from train
references via `common.parse_answer`.

## Ablations and evidence

`code/ablations.py` (validation only), `code/seed_sweep.py`, `code/analyze.py`,
`code/make_figures.py`; raw records in `code/runs/analysis/` and
`evidence/EXPERIMENT_LEDGER.json`; narrative in REPORT.pdf / REPORT.md.

| Variant | F | val % |
|---|---|---|
| hand only | 23 | 89.10 |
| embeddings only | 1024 | 89.97 |
| full minus style | 1040 | 92.36 |
| full minus anchoring | 1044 | 91.73 |
| full (frozen) | 1047 | 92.36 |

Single seeds span 89.72-92.36 (mean 90.98); the weight average is 92.36.
Development runs from 2026-10-18 (MLP 128-hidden, deviation embeddings, group
weighting) were rejected and are recorded in the ledger, not re-run.

## Resource verification

Three fresh-process full-test repeats (predict + score, four threads):
344 / 333 / 333 ms total, median 0.333 s (limit 30 s on the reference CPU).
Inference assets 26,501 B including meta.json (limit 16 MiB); the loadable asset
`model.npz` is 26,347 B. Timing includes startup, imports, data and asset loading.
Repeated test scoring verifies the frozen selector; it does not select alternatives.

## AI assistance and reuse

Method ideas (group-comparative feature design, ListNet formulation, seed-weight
averaging, experiment plan) were developed in discussion with an AI assistant
(Claude, Anthropic); the author implemented and verified all code after
understanding it, ran every experiment reported here, and takes full responsibility
for the submitted implementation and numbers. The supplied data, parser and scorer
were reused unchanged. No external datasets or pretrained weights were used.

## Data attribution

GSM8K: Cobbe et al., MIT license retained in `code/data/`. Candidate solutions and
fixed representations prepared with Qwen3-0.6B by the course staff; no model weights
distributed. Keep the supplied license and attribution files when redistributing.

## Publication

Public repository: https://github.com/911zyx/dase7506-mp2 (branch `main`).
Submit the code URL pinned to a full commit SHA and the checkpoint URL at the same
commit (`.../tree/<SHA>/code/runs/model_bundle`). See SUBMISSION_CHECKLIST.md.
Review REPORT.pdf before publishing; do not overwrite assets or move tags after
submitting.
