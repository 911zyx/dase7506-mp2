# MP2 Experiment Log (submitted version)

Dev machine: AMD Ryzen 7 9800X3D (8 cores), 32 GB RAM, Windows 11,
Python 3.13.9, NumPy 2.5.4 (Matplotlib 3.11.2 for report figures only).
Reference eval machine per guide: Xeon Platinum 8457C, 4 threads, 30 s budget.
All training and inference ran on CPU with NumPy only.

Protocol: `dase7506-mp2-open-hard-verifier-v5` (`python check_data.py`: passed).
All runs below executed 2026-10-29 on the dev machine; seeds 0..9 via
`numpy.random.default_rng`.

## Results (official scorer)

| Method | Validation | Test | Notes |
|---|---|---|---|
| Majority vote (training-free, ties -> lowest index) | 86.2155 | 69.1649 | `runs/*-mv.json` |
| **Final: linear ListNet, 23 hand + 1024 emb, 10-seed weight avg** | **92.3559** | **82.6552** | frozen, `runs/model_bundle` |

Final test run: 386/467. Fresh-process wall time: predict 0.21 s + score 0.11 s
= **0.32 s** (limit 30 s on reference CPU). Asset: `runs/model_bundle/model.npz`
= **26,347 B** (limit 16 MiB).

## Feature-block ablations (validation only; test frozen after submission freeze)

Refit with identical hyperparameters (lr 0.05, l2 3e-3, 800 ep, batch 256,
10-seed weight average). `code/ablations.py`, raw: `runs/analysis/ablations.json`.

| Variant | F | val % |
|---|---|---|
| hand only (no embeddings) | 23 | 89.10 |
| embeddings only (no hand features) | 1024 | 89.97 |
| full minus style block | 1040 | 92.36 |
| full minus question-anchoring block | 1044 | 91.73 |
| full (frozen bundle) | 1047 | 92.36 |

## Seed variance

`code/seed_sweep.py`, raw: `runs/analysis/seeds.json`.
Single-seed val accuracies (seeds 0..9): 92.36, 89.72, 91.10, 91.85, 90.85,
90.98, 90.48, 89.97, 91.48, 90.98 (min 89.72, max 92.36, mean 90.98).
Ten-seed weight average: 92.36.

## Earlier development runs (2026-10-18, same pipeline, recorded not re-run)

| Variant | val % | Verdict |
|---|---|---|
| MLP, 128 hidden, +embedding | 91.35-92.36 | no gain over linear, rejected |
| linear + deviation embeddings (u-ubar) | 87-89 | overfits, rejected |
| group weighting (down-weight all-correct) | 91.73-91.98 | no gain, rejected |

## Error analysis, calibration, weights

`code/analyze.py` -> `runs/analysis/analysis.md`: vote-pattern error table
(MV 110 wrong / model 61 wrong on validation), correct-candidate distribution
{1:117, 2:123, 3:179, 4:379}, baseline tie-break bias, confidence calibration
(flat within contested groups, separation 0.520, ECE 0.52), learned-weight
tables. Figures: `code/make_figures.py` -> `report/figures/`.

## Reproduce

```bash
cd code
python check_data.py
python train.py --variant emb --seeds 10 --out runs/model_bundle   # ~192 s CPU
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python predict.py --method submission.py --checkpoint runs/model_bundle \
      --split test --output runs/test-predictions.json
python score_predictions.py --split test \
    --predictions runs/test-predictions.json --output runs/test-score.json
```

Determinism: retraining on this machine reproduces `model.npz` bit-identically
(verified by SHA-256) and identical per-pattern validation breakdowns.
Audit robustness: `predict()` checked on a shuffled 37-question sub-bank of
validation; all choices valid Python ints and identical to the full-bank run.

## Asset hashes (submitted checkpoint)

- `code/runs/model_bundle/model.npz`
  sha256 `e018153996394d3e2c83a5e5d1d1ad5734c16b4c3b98c16f1cfe3ceeda1cbcde`
- `code/runs/model_bundle/meta.json`
  sha256 `35f9498d5ac55d224229b63e992304c9e0633b51778ac86df3fa81be84dd0c93`

Training uses only the train split (4,755 groups); validation for development
and selection; test references only for the two official scores above. No
external data or weights.
