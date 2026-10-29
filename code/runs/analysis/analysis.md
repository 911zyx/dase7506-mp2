# MP2 analysis (validation split, N=798)

## 1. Vote-pattern error table

| Pattern | #Q | MV wrong | MV err% | Model wrong | Model acc% |
|---|---|---|---|---|---|
| 4 | 379 | 0 | 0.0 | 0 | 100.0 |
| 3-1 | 151 | 22 | 14.6 | 11 | 92.7 |
| 2-1-1 | 88 | 34 | 38.6 | 24 | 72.7 |
| 2-2 | 34 | 20 | 58.8 | 5 | 85.3 |
| 1-1-1-1 | 20 | 15 | 75.0 | 7 | 65.0 |
| 1+u | 3 | 0 | 0.0 | 0 | 100.0 |
| 1-1+u | 14 | 6 | 42.9 | 3 | 78.6 |
| 1-1-1+u | 20 | 9 | 45.0 | 8 | 60.0 |
| 2+u | 11 | 0 | 0.0 | 0 | 100.0 |
| 2-1+u | 28 | 4 | 14.3 | 3 | 89.3 |
| 3+u | 50 | 0 | 0.0 | 0 | 100.0 |

MV total wrong: 110   Model total wrong: 61

## 2. Correct-candidate count per group (oracle headroom)

| #correct candidates | groups |
|---|---|
| 1 | 117 |
| 2 | 123 |
| 3 | 179 |
| 4 | 379 |

Oracle accuracy = 100% by construction; MV = 86.22%, model = 92.36%.

## 3. Baseline tie-break bias (lowest-index rule)

- `1-1+u` (14 groups): lowest-index tied candidate correct 57% vs other tied candidates 43% (6/14)
- `1-1-1+u` (20 groups): lowest-index tied candidate correct 55% vs other tied candidates 22% (9/40)
- `1-1-1-1` (20 groups): lowest-index tied candidate correct 25% vs other tied candidates 25% (15/60)
- `2-2` (34 groups): lowest-index tied candidate correct 41% vs other tied candidates 53% (54/102)

## 4. Confidence calibration (frozen model, validation)

Easy groups (<=1 distinct parsed answer): 443, acc 100.0%; contested groups: 355, acc 82.8%

### All groups: by logit margin (top1-top2), quintiles

| Margin bucket | range | #Q | acc% |
|---|---|---|---|
| Q1 | 0.00 - 0.07 | 160 | 95.6 |
| Q2 | 0.07 - 0.17 | 159 | 92.5 |
| Q3 | 0.17 - 0.30 | 160 | 93.1 |
| Q4 | 0.30 - 0.53 | 159 | 89.9 |
| Q5 | 0.53 - 5.21 | 160 | 90.6 |

NOTE: easy (unanimous) groups have near-zero margins by construction
and are ~always correct, so the overall margin-accuracy relation is
inverted. The informative view is contested groups only:

### Contested groups only (N=355)

| Margin bucket | range | #Q | acc% |
|---|---|---|---|
| Q1 | 0.00 - 0.11 | 71 | 84.5 |
| Q2 | 0.11 - 0.23 | 71 | 80.3 |
| Q3 | 0.23 - 0.40 | 71 | 81.7 |
| Q4 | 0.40 - 0.71 | 71 | 83.1 |
| Q5 | 0.71 - 3.11 | 71 | 84.5 |

### Group-softmax probability of the pick, fixed bins (all groups)

| p_top bin | #Q | acc% |
|---|---|---|---|
| 0.0 - 0.4 | 497 | 93.8 |
| 0.4 - 0.5 | 172 | 90.1 |
| 0.5 - 0.6 | 74 | 86.5 |
| 0.6 - 0.7 | 29 | 93.1 |
| 0.7 - 0.8 | 17 | 94.1 |
| 0.8 - 0.9 | 4 | 100.0 |
| 0.9 - 1.0 | 5 | 100.0 |

Mean p_top when correct: 0.398 | when wrong: 0.426 (all groups)
Contested only - mean p_top correct: 0.452 | wrong: 0.426
Contested only - mean margin correct: 0.465 | wrong: 0.374
ECE all groups (7 bins, p_top): 0.5236
Separation P(margin_correct > margin_wrong): all 0.432 | contested 0.520

Interpretation: the ListNet group softmax is a *ranking* loss, not a
probability estimator, so absolute p_top is badly underconfident.
The decision-relevant question is whether the margin ranks correctness
within contested groups; the separation statistic answers it.

## 5. Learned weights (frozen bundle)

||w|| embedding block = 1.815   ||w|| handcrafted block = 0.948

| rank | handcrafted feature | weight | family |
|---|---|---|---|
| 1 | parseable | +0.8389 | vote |
| 2 | numbers not in question | +0.3034 | anchor |
| 3 | cos to same-answer peers | -0.2218 | geometry |
| 4 | cos to majority centroid | +0.1606 | geometry |
| 5 | question-number coverage | +0.1014 | anchor |
| 6 | cos to different-answer | +0.0754 | geometry |
| 7 | vote margin | +0.0530 | vote |
| 8 | missed question numbers | -0.0499 | anchor |
| 9 | has #### | -0.0454 | style |
| 10 | cosine-weighted vote margin | +0.0409 | geometry |
| 11 | count '$' | -0.0323 | style |
| 12 | cos to centroid | -0.0247 | geometry |
| 13 | embedding norm | +0.0194 | geometry |
| 14 | cos to others | -0.0185 | geometry |
| 15 | count '=' | +0.0117 | style |

Family weight norms (handcrafted): vote 0.841, geometry 0.289, style 0.059, anchor 0.324
