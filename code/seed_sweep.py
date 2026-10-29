"""Per-seed validation accuracy of the final feature set (report Table).

Trains the frozen hyperparameters once per seed (0..9) on the FULL feature set
and evaluates each single-seed model on validation, plus the 10-seed weight
average. Reproduces the seed-variance claim of the report on this pipeline.

Usage (from code/):
    python seed_sweep.py
"""
import json
import time
from pathlib import Path

import numpy as np

from common import load_inputs
from submission import build_features
from train import group_labels, train_listnet, evaluate

OUT = Path('runs/analysis')
SEEDS, EPOCHS, LR, L2, BATCH = 10, 800, 0.05, 3e-3, 256


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bank_tr, arr_tr = load_inputs('train')
    y = group_labels(bank_tr, None, 'train')
    X = build_features(bank_tr, arr_tr, with_emb=True)
    Xf = X.reshape(-1, X.shape[2])
    mu, sd = Xf.mean(0), Xf.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    bank_va, arr_va = load_inputs('validation')
    Xva = build_features(bank_va, arr_va, with_emb=True)

    t0 = time.time()
    ws, bs, accs = [], [], []
    for seed in range(SEEDS):
        w, b = train_listnet(Xs, y, seed=seed, epochs=EPOCHS, lr=LR, l2=L2, batch=BATCH)
        ws.append(w); bs.append(b)
        logits = ((Xva - mu) / sd) @ w + b
        choices = [int(x) for x in np.argmax(logits, axis=1)]
        accs.append(evaluate(choices, bank_va, 'validation', f'seed {seed}'))
        print(f'    seed {seed}: {accs[-1]:.2f}%  ({time.time() - t0:.0f}s elapsed)')
    logits = ((Xva - mu) / sd) @ np.mean(ws, axis=0) + np.mean(bs, axis=0)
    choices = [int(x) for x in np.argmax(logits, axis=1)]
    avg = evaluate(choices, bank_va, 'validation', '10-seed weight average')
    res = {'single_seed_accuracies': accs, 'min': min(accs), 'max': max(accs),
           'mean': sum(accs) / len(accs), 'ten_seed_average': avg}
    (OUT / 'seeds.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
