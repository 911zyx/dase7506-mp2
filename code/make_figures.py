"""Generate report figures (report/figures/*.pdf + *.png previews).

Figure 1: accuracy by vote pattern, majority-vote baseline vs final model.
Figure 2: accuracy vs logit-margin quintile on contested validation groups
          (calibration of the frozen model's confidence).

Palette: validated categorical slots 1-2 (blue #2a78d6, orange #eb6834);
hatching is the secondary encoding for grayscale/CVD printing.

Usage (from code/):
    python make_figures.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from submission import build_features, _majority_vote

FIG = Path('../report/figures')
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#1a1a19', '#6b7280', '#d6d6d3'

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
    'axes.labelcolor': INK, 'legend.frameon': False, 'legend.fontsize': 8.5,
    'hatch.linewidth': 0.6, 'hatch.color': 'white',
})


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def pattern_of(row):
    parsed = [parse_answer(c['text']) for c in row['candidates']]
    cnt = Counter(p for p in parsed if p is not None)
    if any(p is None for p in parsed):
        return 'unparsed'
    return {
        (4,): '4-0', (3, 1): '3-1', (2, 1, 1): '2-1-1',
        (2, 2): '2-2', (1, 1, 1, 1): '1-1-1-1',
    }[tuple(sorted(cnt.values(), reverse=True))]


def fig1(bank, mv_ok, md_ok):
    keys = ['4-0', '3-1', '2-1-1', '2-2', '1-1-1-1', 'unparsed']
    pats = [pattern_of(row) for row in bank]
    mv_acc, md_acc, ns = [], [], []
    for k in keys:
        sel = np.array([i for i, p in enumerate(pats) if p == k])
        ns.append(len(sel))
        mv_acc.append(100 * mv_ok[sel].mean())
        md_acc.append(100 * md_ok[sel].mean())
    cats = [f'{k}\n(n={n})' for k, n in zip(keys, ns)]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(cats))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, mv_acc, w, color=BLUE, hatch='////',
           edgecolor=BLUE, linewidth=0, label='Majority vote (baseline)')
    ax.bar(x + w / 2 + 0.01, md_acc, w, color=ORANGE,
           edgecolor=ORANGE, linewidth=0, label='Final model (ListNet, 10 seeds)')
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 108)
    ax.set_ylabel('Selection accuracy (%)')
    ax.set_title('Accuracy by vote pattern, validation split (baseline vs final model)')
    ax.legend(loc='upper right')
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_patterns.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig1_patterns.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def fig2(bank, arrays, md_ok):
    bundle = np.load('runs/model_bundle/model.npz')
    feats = build_features(bank, arrays, with_emb=bool(bundle['with_emb']))
    logits = ((feats - bundle['mu']) / bundle['sd']) @ bundle['w'] + bundle['b']
    srt = np.sort(logits, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    n_distinct = np.array([
        len({parse_answer(c['text']) for c in row['candidates']} - {None})
        for row in bank])
    contested = n_distinct >= 2
    mc, oc = margin[contested], md_ok[contested]
    qs = np.quantile(mc, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(mc, qs)
    accs, mids, labels = [], [], []
    for bi in range(5):
        sel = b == bi
        accs.append(100 * oc[sel].mean())
        mids.append(mc[sel].mean())
        labels.append(f'Q{bi + 1}\n{mc[sel].min():.2f}-{mc[sel].max():.2f}')

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axhline(100 * oc.mean(), color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(-0.32, 58.0, f'dashed: overall contested accuracy {100 * oc.mean():.1f}%',
            color=MUTED, fontsize=8, ha='left')
    ax.plot(range(5), accs, color=BLUE, linewidth=2, marker='o', markersize=7,
            markerfacecolor=BLUE, markeredgecolor='white', markeredgewidth=1.5)
    ax.set_xticks(range(5), labels, fontsize=8)
    ax.set_xlabel('Logit margin quintile (top1 $-$ top2)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(55, 100)
    ax.set_title('Confidence is flat: accuracy vs margin,\ncontested groups only (N=%d)'
                 % contested.sum())
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_calibration.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'fig2_calibration.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    bank, arrays = load_inputs('validation')
    gold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    refs = [parse_answer(q['answer']) for q in gold]
    correct = np.array([[parse_answer(c['text']) == r for c in row['candidates']]
                        for row, r in zip(bank, refs)])
    mv = np.array(_majority_vote(bank))
    model = np.array(json.load(open('runs/validation-predictions.json'))['selected'])
    ar = np.arange(len(bank))
    fig1(bank, correct[ar, mv], correct[ar, model])
    fig2(bank, arrays, correct[ar, model])
    print('figures saved to', FIG.resolve())


if __name__ == '__main__':
    main()
