"""Offline training for the MP2 learned selector.

Fits a linear scorer on the TRAIN split only (labels from train references),
saves a tiny NumPy bundle for submission.py, and reports validation accuracy
(validation is used for development/model selection, per the guide).

Usage (from code/):
    python train.py --variant hand --out runs/model_bundle
    python train.py --variant emb  --out runs/model_bundle_emb --l2 1e-3
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from common import DATA, parse_answer, read_jsonl, score_choices
from submission import build_features


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def group_labels(bank, gold, split):
    """y[g, i] = 1 iff candidate i's parsed answer matches the reference."""
    gold_path = DATA / f'gsm8k_{split}.jsonl'
    refs = read_jsonl(gold_path) if gold is None else gold
    assert len(refs) == len(bank)
    y = np.zeros((len(bank), 4))
    for g, (row, q) in enumerate(zip(bank, refs)):
        assert row['id'] == q['id'] and row['question'] == q['question']
        ref = parse_answer(q['answer'])
        for i, c in enumerate(row['candidates']):
            y[g, i] = float(parse_answer(c['text']) == ref)
    assert y.sum(1).min() >= 1, 'every group must have a correct candidate'
    return y


# ---------------------------------------------------------------------------
# ListNet-style group softmax training (pure NumPy, deterministic seed)
# ---------------------------------------------------------------------------

def train_listnet(X, y, seed=0, epochs=400, lr=0.05, l2=1e-4, batch=256):
    """X: [N,4,F] standardized; y: [N,4] binary. Mini-batch Adam."""
    rng = np.random.default_rng(seed)
    n, _, F = X.shape
    w = np.zeros(F)
    b = np.zeros(1)
    # target distribution over the group: uniform across correct candidates
    q = y / np.maximum(y.sum(1, keepdims=True), 1)
    m_w, v_w, m_b, v_b = np.zeros(F), np.zeros(F), np.zeros(1), np.zeros(1)
    b1, b2, eps = 0.9, 0.999, 1e-8
    order_groups = np.arange(n)
    t = 0
    for epoch in range(epochs):
        rng.shuffle(order_groups)
        for s in range(0, n, batch):
            idx = order_groups[s:s + batch]
            Xb = X[idx].reshape(len(idx) * 4, F)
            qb = q[idx]                                  # [B,4]
            logits = (Xb @ w + b).reshape(len(idx), 4)
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            grad_logits = (p - qb) / len(idx)            # [B,4]
            g_w = Xb.T @ grad_logits.reshape(-1) + 2 * l2 * w
            g_b = np.array([grad_logits.sum()])
            t += 1
            m_w = b1 * m_w + (1 - b1) * g_w; v_w = b2 * v_w + (1 - b2) * g_w**2
            m_b = b1 * m_b + (1 - b1) * g_b; v_b = b2 * v_b + (1 - b2) * g_b**2
            w -= lr * (m_w / (1 - b1**t)) / (np.sqrt(v_w / (1 - b2**t)) + eps)
            b -= lr * (m_b / (1 - b1**t)) / (np.sqrt(v_b / (1 - b2**t)) + eps)
    return w, b


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def vote_pattern(bank):
    pats = []
    for row in bank:
        parsed = [parse_answer(c['text']) for c in row['candidates']]
        cnt = Counter(p for p in parsed if p is not None)
        pat = '-'.join(str(k) for k in sorted(cnt.values(), reverse=True)) or 'none'
        if any(p is None for p in parsed):
            pat += '+u'
        pats.append(pat)
    return pats


def evaluate(choices, bank, split, tag):
    res = score_choices(split, [r['id'] for r in bank], choices)
    gold = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    pats = vote_pattern(bank)
    by_pat, by_pat_n = Counter(), Counter()
    for row, q, pat, i in zip(bank, gold, pats, choices):
        ok = parse_answer(row['candidates'][i]['text']) == parse_answer(q['answer'])
        by_pat[pat] += ok
        by_pat_n[pat] += 1
    print(f'[{tag}] {split} accuracy = {res["accuracy"]:.2f}%  ({res["correct"]}/{res["n"]})')
    hard = [(p, by_pat[p], by_pat_n[p]) for p in by_pat_n if not p.startswith('4') and '3+u' not in p and '2+u' not in p]
    for p, c, nn in sorted(hard, key=lambda x: -x[2]):
        print(f'    pattern {p:10s} {c:4d}/{nn:4d} = {100*c/nn:5.1f}%')
    return res['accuracy']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('hand', 'emb'), default='emb')
    ap.add_argument('--out', type=Path, default=Path('runs/model_bundle'))
    ap.add_argument('--seeds', type=int, default=10,
                    help='train this many seeds and average weights (variance reduction)')
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--l2', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()
    with_emb = args.variant == 'emb'

    from common import load_inputs
    t0 = time.time()
    bank_tr, arr_tr = load_inputs('train')
    y = group_labels(bank_tr, None, 'train')
    X = build_features(bank_tr, arr_tr, with_emb=with_emb)
    print(f'train features: {X.shape}  positive labels: {y.sum():.0f}/{y.size}  ({time.time()-t0:.1f}s)')

    Xf = X.reshape(-1, X.shape[2])
    mu = Xf.mean(0)
    sd = Xf.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd

    t0 = time.time()
    ws, bs = [], []
    for seed in range(args.seeds):
        w, b = train_listnet(Xs, y, seed=seed, epochs=args.epochs,
                             lr=args.lr, l2=args.l2, batch=args.batch)
        ws.append(w)
        bs.append(b)
    # Averaging logits of linear models == averaging their weights, so the
    # seed ensemble collapses into a single linear scorer (same asset size).
    w, b = np.mean(ws, axis=0), np.mean(bs, axis=0)
    print(f'training ({args.seeds} seeds): {time.time()-t0:.1f}s  |w|={np.linalg.norm(w):.2f}')

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'model.npz', w=w, b=b, mu=mu, sd=sd,
             with_emb=np.array(with_emb))
    meta = {'variant': args.variant, 'seeds': args.seeds, 'epochs': args.epochs,
            'lr': args.lr, 'l2': args.l2, 'batch': args.batch,
            'n_features': int(X.shape[2]),
            'asset_bytes': int((args.out / 'model.npz').stat().st_size)}
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('bundle saved to', args.out, meta['asset_bytes'], 'bytes')

    # development evaluation on validation (allowed: model selection)
    from submission import predict
    bank_va, arr_va = load_inputs('validation')
    t0 = time.time()
    choices = predict(bank_va, arr_va, args.out)
    dt = time.time() - t0
    evaluate(choices, bank_va, 'validation', f'listnet-{args.variant}')
    print(f'inference on validation: {dt:.2f}s')


if __name__ == '__main__':
    main()
