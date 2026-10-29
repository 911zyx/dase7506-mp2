"""Error analysis, calibration and weight interpretation for the report.

Not part of the frozen inference path. Reads the frozen bundle in
runs/model_bundle and the archived official predictions, and writes
runs/analysis/analysis.md with:

  1. Vote-pattern error table: majority-vote baseline vs final model (validation).
  2. Correct-candidate-count distribution per group (oracle headroom).
  3. Tie-break bias of the baseline (lowest-index rule).
  4. Calibration of the model's group-softmax confidence (margin / prob buckets),
     stratified into easy (unanimous) vs contested groups.
  5. Learned weight interpretation (top handcrafted weights, block norms).

Usage (from code/):
    python analyze.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from submission import N_HAND, build_features, _majority_vote

OUT = Path('runs/analysis')
BUNDLE = Path('runs/model_bundle')

HAND_NAMES = [
    'agree(same-answer peers)', 'parseable', 'group max votes', 'distinct answers',
    'unparsed in group', 'cos to others', 'cos to centroid', 'cos to majority centroid',
    'cos to same-answer peers', 'embedding norm', 'char length', 'char length (rel)',
    'line count', 'has \\boxed', 'has ####', "count '='", "count '$'",
    'question-number coverage', 'missed question numbers', 'numbers not in question',
    'cos to different-answer', 'vote margin', 'cosine-weighted vote margin',
]
STYLE_IDX = {10, 11, 12, 13, 14, 15, 16}
ANCHOR_IDX = {17, 18, 19}
VOTE_IDX = {0, 1, 2, 3, 4, 21}
GEOM_IDX = {5, 6, 7, 8, 9, 20, 22}


def vote_pattern(row):
    parsed = [parse_answer(c['text']) for c in row['candidates']]
    cnt = Counter(p for p in parsed if p is not None)
    pat = '-'.join(str(k) for k in sorted(cnt.values(), reverse=True)) or 'none'
    if any(p is None for p in parsed):
        pat += '+u'
    return pat, parsed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bank, arrays = load_inputs('validation')
    gold = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    assert all(r['id'] == q['id'] for r, q in zip(bank, gold))
    refs = [parse_answer(q['answer']) for q in gold]
    correct = np.array([[parse_answer(c['text']) == r for c in row['candidates']]
                        for row, r in zip(bank, refs)])          # [N,4] bool

    mv = np.array(_majority_vote(bank))
    model = np.array(json.load(open('runs/validation-predictions.json'))['selected'])
    assert len(model) == len(bank)
    mv_ok = correct[np.arange(len(bank)), mv]
    md_ok = correct[np.arange(len(bank)), model]

    lines = ['# MP2 analysis (validation split, N=%d)' % len(bank), '']

    # ---- 1. pattern table -------------------------------------------------
    pats = [vote_pattern(row)[0] for row in bank]
    lines += ['## 1. Vote-pattern error table', '',
              '| Pattern | #Q | MV wrong | MV err% | Model wrong | Model acc% |',
              '|---|---|---|---|---|---|']
    order = ['4', '3-1', '2-1-1', '2-2', '1-1-1-1']
    groups = Counter(pats)
    for p in order + sorted(k for k in groups if k not in order):
        if p not in groups:
            continue
        sel = np.array([i for i, q in enumerate(pats) if q == p])
        n = len(sel)
        lines.append('| %s | %d | %d | %.1f | %d | %.1f |' % (
            p, n, (~mv_ok[sel]).sum(), 100 * (~mv_ok[sel]).mean(),
            (~md_ok[sel]).sum(), 100 * md_ok[sel].mean()))
    lines += ['', 'MV total wrong: %d   Model total wrong: %d' % (
        (~mv_ok).sum(), (~md_ok).sum()), '']

    # ---- 2. oracle headroom ----------------------------------------------
    ncorr = correct.sum(1)
    dist = Counter(ncorr.tolist())
    lines += ['## 2. Correct-candidate count per group (oracle headroom)', '',
              '| #correct candidates | groups |', '|---|---|']
    for k in sorted(dist):
        lines.append('| %d | %d |' % (k, dist[k]))
    lines += ['', 'Oracle accuracy = 100%% by construction; MV = %.2f%%, model = %.2f%%.'
              % (100 * mv_ok.mean(), 100 * md_ok.mean()), '']

    # ---- 3. tie-break bias -------------------------------------------------
    lines += ['## 3. Baseline tie-break bias (lowest-index rule)', '']
    tie_pats = {'2-2', '1-1-1-1', '1-1', '2-2+u', '1-1+u', '1-1-1-1+u', '1-1-1+u'}
    for p in sorted(tie_pats):
        sel = [i for i, q in enumerate(pats) if q == p]
        if not sel:
            continue
        # among tied winning answers, is the lowest-index holder less often correct?
        first_ok, other_ok, other_n = 0, 0, 0
        for i in sel:
            _, parsed = vote_pattern(bank[i])
            cnt = Counter(x for x in parsed if x is not None)
            top = max(cnt.values())
            winners = {a for a, k in cnt.items() if k == top}
            if len(winners) < 2:
                continue
            holder_first = next(j for j, x in enumerate(parsed) if x in winners)
            first_ok += correct[i, holder_first]
            for j in range(4):
                if j != holder_first and parsed[j] in winners:
                    other_ok += correct[i, j]; other_n += 1
        n_tied = len(sel)
        lines.append('- `%s` (%d groups): lowest-index tied candidate correct %.0f%% '
                     'vs other tied candidates %.0f%% (%d/%d)'
                     % (p, n_tied, 100 * first_ok / max(n_tied, 1),
                        100 * other_ok / max(other_n, 1), other_ok, other_n))
    lines.append('')

    # ---- 4. calibration ----------------------------------------------------
    bundle = np.load(BUNDLE / 'model.npz')
    feats = build_features(bank, arrays, with_emb=bool(bundle['with_emb']))
    z = (feats - bundle['mu']) / bundle['sd']
    logits = z @ bundle['w'] + bundle['b']                     # [N,4]
    srt = np.sort(logits, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    exp = np.exp(logits - logits.max(1, keepdims=True))
    prob = exp / exp.sum(1, keepdims=True)
    p_top = prob.max(1)

    # Stratify: "easy" = at most one distinct parsed answer in the group
    # (all parsed candidates agree). These groups have near-identical logits by
    # construction and are ~always correct, which would otherwise swamp the
    # calibration statistics and invert the margin-accuracy relation.
    n_distinct = np.array([len({p for p in vote_pattern(row)[1] if p is not None})
                           for row in bank])
    easy = n_distinct <= 1
    contested = ~easy

    def _ece(p, ok, edges):
        e = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (p >= a) & (p < b)
            if sel.sum():
                e += abs(p[sel].mean() - ok[sel].mean()) * sel.sum()
        return e / max(len(p), 1)

    def separation(m, ok):
        if ok.sum() == 0 or (~ok).sum() == 0:
            return float('nan')
        return float(np.mean([np.mean(m[~ok] < x) for x in m[ok]]))

    edges = [0, .4, .5, .6, .7, .8, .9, 1.01]
    lines += ['## 4. Confidence calibration (frozen model, validation)', '',
              'Easy groups (<=1 distinct parsed answer): %d, acc %.1f%%; '
              'contested groups: %d, acc %.1f%%'
              % (easy.sum(), 100 * md_ok[easy].mean(),
                 contested.sum(), 100 * md_ok[contested].mean()), '',
              '### All groups: by logit margin (top1-top2), quintiles', '',
              '| Margin bucket | range | #Q | acc% |', '|---|---|---|---|']
    qs = np.quantile(margin, [0.2, 0.4, 0.6, 0.8])
    b_idx = np.digitize(margin, qs)
    for bi in range(5):
        sel = b_idx == bi
        lo, hi = margin[sel].min(), margin[sel].max()
        lines.append('| Q%d | %.2f - %.2f | %d | %.1f |'
                     % (bi + 1, lo, hi, sel.sum(), 100 * md_ok[sel].mean()))
    lines += ['',
              'NOTE: easy (unanimous) groups have near-zero margins by construction',
              'and are ~always correct, so the overall margin-accuracy relation is',
              'inverted. The informative view is contested groups only:', '',
              '### Contested groups only (N=%d)' % contested.sum(), '',
              '| Margin bucket | range | #Q | acc% |', '|---|---|---|---|']
    mc, oc = margin[contested], md_ok[contested]
    qs = np.quantile(mc, [0.2, 0.4, 0.6, 0.8])
    b_idx = np.digitize(mc, qs)
    for bi in range(5):
        sel = b_idx == bi
        lines.append('| Q%d | %.2f - %.2f | %d | %.1f |'
                     % (bi + 1, mc[sel].min(), mc[sel].max(), sel.sum(),
                        100 * oc[sel].mean()))
    lines += ['', '### Group-softmax probability of the pick, fixed bins (all groups)',
              '', '| p_top bin | #Q | acc% |', '|---|---|---|---|']
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (p_top >= a) & (p_top < b)
        if sel.sum():
            lines.append('| %.1f - %.1f | %d | %.1f |' % (a, min(b, 1.0), sel.sum(),
                                                           100 * md_ok[sel].mean()))
    lines += ['', 'Mean p_top when correct: %.3f | when wrong: %.3f (all groups)'
              % (p_top[md_ok].mean(), p_top[~md_ok].mean()),
              'Contested only - mean p_top correct: %.3f | wrong: %.3f'
              % (p_top[contested & md_ok].mean(), p_top[contested & ~md_ok].mean()),
              'Contested only - mean margin correct: %.3f | wrong: %.3f'
              % (margin[contested & md_ok].mean(), margin[contested & ~md_ok].mean()),
              'ECE all groups (7 bins, p_top): %.4f' % _ece(p_top, md_ok, edges),
              'Separation P(margin_correct > margin_wrong): all %.3f | contested %.3f'
              % (separation(margin, md_ok), separation(mc, oc)), '',
              'Interpretation: the ListNet group softmax is a *ranking* loss, not a',
              'probability estimator, so absolute p_top is badly underconfident.',
              'The decision-relevant question is whether the margin ranks correctness',
              'within contested groups; the separation statistic answers it.', '']

    # ---- 5. weight interpretation ------------------------------------------
    w = bundle['w']
    w_emb, w_hand = w[:1024], w[1024:]
    lines += ['## 5. Learned weights (frozen bundle)', '',
              '||w|| embedding block = %.3f   ||w|| handcrafted block = %.3f'
              % (np.linalg.norm(w_emb), np.linalg.norm(w_hand)), '',
              '| rank | handcrafted feature | weight | family |', '|---|---|---|---|']
    fam = {}
    for idxs, name in [(VOTE_IDX, 'vote'), (GEOM_IDX, 'geometry'),
                       (STYLE_IDX, 'style'), (ANCHOR_IDX, 'anchor')]:
        for i in idxs:
            fam[i] = name
    for r, i in enumerate(np.argsort(-np.abs(w_hand))[:15]):
        lines.append('| %d | %s | %+.4f | %s |' % (r + 1, HAND_NAMES[i], w_hand[i], fam[i]))
    fam_norm = {f: np.linalg.norm([w_hand[i] for i in sorted(idxs)])
                for f, idxs in [('vote', VOTE_IDX), ('geometry', GEOM_IDX),
                                ('style', STYLE_IDX), ('anchor', ANCHOR_IDX)]}
    lines += ['', 'Family weight norms (handcrafted): ' +
              ', '.join('%s %.3f' % (k, v) for k, v in fam_norm.items()), '']

    text = '\n'.join(lines)
    (OUT / 'analysis.md').write_text(text, encoding='utf-8')
    print(text)
    print('\nsaved to', OUT / 'analysis.md')


if __name__ == '__main__':
    main()
