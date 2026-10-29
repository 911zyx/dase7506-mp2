"""Submission interface: learned candidate selector for MP2.

Method overview
---------------
For every question we build a small feature vector per candidate, combining
three evidence sources:

1. Vote agreement  - how many of the other candidates reach the same parsed
                     final answer (self-consistency signal).
2. Embedding geometry - cosine similarity of a candidate's frozen trace
                     embedding to the group centroid, to the majority-answer
                     centroid and to same-answer peers (consistency signal).
3. Text statistics - length, structure, answer formatting (style signal;
                     ablated in the report).

A linear scorer (trained offline by train.py on the train split only) maps
each candidate's standardized features to a logit; predict() returns the
argmax candidate per question. Inference is pure NumPy, far below the CPU
and asset budgets.

If no checkpoint bundle is supplied, predict() falls back to deterministic
majority vote, so the file is also a valid training-free baseline.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np

from common import parse_answer

# ---------------------------------------------------------------------------
# Feature construction (shared by training and inference)
# ---------------------------------------------------------------------------

N_HAND = 23  # number of handcrafted features

_NUM_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')


def _numbers(text):
    """Numeric tokens in a text, commas stripped, as a set of strings."""
    return {m.replace(',', '') for m in _NUM_RE.findall(text)}


def _unit(E):
    """Row-wise L2-normalize embeddings; returns (U, norms)."""
    norms = np.linalg.norm(E, axis=-1, keepdims=True)
    U = E / np.maximum(norms, 1e-12)
    return U, norms[..., 0]


def build_features(bank, arrays, with_emb=False):
    """Return float64 features of shape [N, 4, F].

    F = N_HAND, or 1024 + N_HAND when with_emb (unit embedding first).
    """
    n = len(bank)
    E = np.asarray(arrays['embeddings'], dtype=np.float64)  # [N,4,1024]
    U, norms = _unit(E)
    feats = np.zeros((n, 4, N_HAND))

    for g, row in enumerate(bank):
        texts = [c['text'] for c in row['candidates']]
        parsed = [parse_answer(t) for t in texts]
        votes = Counter(p for p in parsed if p is not None)
        maxvote = max(votes.values()) if votes else 0
        winners = {a for a, k in votes.items() if k == maxvote} if votes else set()
        n_unparsed = sum(p is None for p in parsed)

        S = U[g] @ U[g].T                      # 4x4 cosine matrix
        cos_others = (S.sum(1) - np.diag(S)) / 3.0
        centroid = U[g].mean(0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        cos_centroid = U[g] @ centroid
        if winners:
            maj_mask = np.array([p in winners for p in parsed])
            cm = U[g][maj_mask].mean(0)
            cm /= max(np.linalg.norm(cm), 1e-12)
            cos_maj = U[g] @ cm
        else:
            cos_maj = cos_centroid

        cos_same = cos_others.copy()
        cos_diff = cos_others.copy()
        for i in range(4):
            if parsed[i] is not None:
                peers = [j for j in range(4) if j != i and parsed[j] == parsed[i]]
                foes = [j for j in range(4) if j != i and parsed[j] is not None
                        and parsed[j] != parsed[i]]
                if peers:
                    cos_same[i] = float(np.mean([S[i, j] for j in peers]))
                if foes:
                    cos_diff[i] = float(np.mean([S[i, j] for j in foes]))

        qnums = _numbers(row.get('question', ''))
        for i, t in enumerate(texts):
            agree = votes.get(parsed[i], 0) - 1 if parsed[i] is not None else 0
            others_votes = [k for a, k in votes.items() if a != parsed[i]]
            vote_margin = agree - (max(others_votes) if others_votes else 0)
            sim_margin = float(
                sum(S[i, j] for j in range(4) if j != i and parsed[j] == parsed[i])
                - sum(S[i, j] for j in range(4)
                      if j != i and parsed[j] is not None and parsed[j] != parsed[i]))
            cnums = _numbers(t)
            missed = len(qnums - cnums)
            lines = t.count('\n') + 1
            f = feats[g, i]
            f[0] = agree                                   # same-answer peers
            f[1] = 1.0 if parsed[i] is not None else 0.0   # parseable answer
            f[2] = maxvote                                 # group majority size
            f[3] = len(votes)                              # distinct answers
            f[4] = n_unparsed                              # unparsed in group
            f[5] = cos_others[i]
            f[6] = cos_centroid[i]
            f[7] = cos_maj[i]
            f[8] = cos_same[i]
            f[9] = norms[g, i]                             # raw embedding norm
            f[10] = len(t)                                 # chars
            f[11] = len(t) - np.mean([len(x) for x in texts])
            f[12] = lines
            f[13] = 1.0 if '\\boxed' in t else 0.0
            f[14] = 1.0 if '####' in t else 0.0
            f[15] = t.count('=')                           # computation steps
            f[16] = t.count('$')                           # LaTeX usage
            f[17] = 1.0 - missed / max(len(qnums), 1)      # question-number coverage
            f[18] = missed                                 # question conditions skipped
            f[19] = len(cnums - qnums)                     # numbers not in question
            f[20] = cos_diff[i]
            f[21] = vote_margin                            # votes minus best rival answer
            f[22] = sim_margin                             # cosine-weighted vote margin

    if with_emb:
        feats = np.concatenate([U, feats], axis=2)         # [N,4,1024+N_HAND]
    return feats


def _majority_vote(bank):
    """Deterministic training-free fallback: vote, ties to lowest index."""
    choices = []
    for row in bank:
        parsed = [parse_answer(c['text']) for c in row['candidates']]
        votes = Counter(p for p in parsed if p is not None)
        if not votes:
            choices.append(0)
            continue
        top = max(votes.values())
        winners = {a for a, k in votes.items() if k == top}
        choices.append(next(i for i, p in enumerate(parsed) if p in winners))
    return choices


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _select_from_logits(logits):
    """Argmax per group; ties to lowest index. Returns list of Python ints."""
    best = np.argmax(logits, axis=1)
    # np.argmax already returns the first (lowest) index on ties.
    return [int(b) for b in best]


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order."""
    bundle = None
    if checkpoint is not None:
        path = Path(checkpoint) / 'model.npz'
        if path.exists():
            bundle = np.load(path, allow_pickle=False)
    if bundle is None:
        return _majority_vote(bank)

    with_emb = bool(bundle['with_emb'])
    feats = build_features(bank, arrays, with_emb=with_emb)
    z = (feats - bundle['mu']) / bundle['sd']
    logits = z @ bundle['w'] + bundle['b']
    return _select_from_logits(logits)
