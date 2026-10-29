"""Feature-block ablations for the report (validation only; test stays frozen).

Refits the same linear ListNet pipeline (identical hyperparameters, 10-seed
weight averaging, train split only) on masked feature subsets:

  hand        23 handcrafted features, no embeddings      -> "reading the trace?"
  emb_only    1024-d unit embedding, no handcrafted       -> "judged on its own?"
  no_style    full minus 7 style features (length/format) -> style-bias check
  no_anchor   full minus 3 question-anchoring features    -> dropped-condition check
  full        23 + 1024 (frozen model, evaluated for reference)

Usage (from code/):
    python ablations.py
"""
import json
import time
from pathlib import Path

import numpy as np

from common import load_inputs
from submission import N_HAND, build_features
from train import group_labels, train_listnet, evaluate

OUT = Path('runs/analysis')
EMB = 1024
STYLE = [EMB + i for i in (10, 11, 12, 13, 14, 15, 16)]
ANCHOR = [EMB + i for i in (17, 18, 19)]
ALL = list(range(EMB + N_HAND))

VARIANTS = {
    'hand': list(range(EMB, EMB + N_HAND)),
    'emb_only': list(range(EMB)),
    'no_style': [c for c in ALL if c not in STYLE],
    'no_anchor': [c for c in ALL if c not in ANCHOR],
    'full': ALL,
}

SEEDS, EPOCHS, LR, L2, BATCH = 10, 800, 0.05, 3e-3, 256


def run_variant(name, cols, Xtr_raw, y, bank_va, arr_va):
    t0 = time.time()
    Xtr = Xtr_raw[:, :, cols]
    Xf = Xtr.reshape(-1, len(cols))
    mu, sd = Xf.mean(0), Xf.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (Xtr - mu) / sd
    ws, bs = [], []
    for seed in range(SEEDS):
        w, b = train_listnet(Xs, y, seed=seed, epochs=EPOCHS, lr=LR, l2=L2, batch=BATCH)
        ws.append(w); bs.append(b)
    w, b = np.mean(ws, axis=0), np.mean(bs, axis=0)
    Xva = build_features(bank_va, arr_va, with_emb=True)[:, :, cols]
    logits = ((Xva - mu) / sd) @ w + b
    choices = [int(x) for x in np.argmax(logits, axis=1)]
    acc = evaluate(choices, bank_va, 'validation', f'ablation:{name}')
    print(f'    ({time.time() - t0:.0f}s, F={len(cols)})')
    return {'variant': name, 'n_features': len(cols), 'val_accuracy': acc,
            'seconds': round(time.time() - t0, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bank_tr, arr_tr = load_inputs('train')
    y = group_labels(bank_tr, None, 'train')
    Xtr_raw = build_features(bank_tr, arr_tr, with_emb=True)
    bank_va, arr_va = load_inputs('validation')

    results = []
    for name, cols in VARIANTS.items():
        if name == 'full':
            # frozen bundle: reuse official archived score, no retrain
            score = json.load(open('runs/validation-score.json'))
            acc = score['accuracy']
            print(f'[ablation:full] validation accuracy = {acc:.2f}% (frozen bundle)')
            results.append({'variant': name, 'n_features': len(cols),
                            'val_accuracy': acc, 'seconds': 0.0, 'frozen': True})
            continue
        results.append(run_variant(name, cols, Xtr_raw, y, bank_va, arr_va))

    (OUT / 'ablations.json').write_text(json.dumps(results, indent=2))
    print('\n| variant | F | val acc% |')
    print('|---|---|---|')
    for r in results:
        print(f"| {r['variant']} | {r['n_features']} | {r['val_accuracy']:.2f} |")
    print('saved to', OUT / 'ablations.json')


if __name__ == '__main__':
    main()
