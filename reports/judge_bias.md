# Local LLM judges vs. human experts: agreement and bias

Each key (MT-Bench question, turn 1, model pair with human votes) was judged
twice, once in each answer order. Agreement is computed on the same keys for
every row, including the released GPT-4 verdicts. 95 % intervals: bootstrap
over questions (2000 resamples).

## llama-3.2-3b

400 keys, 6 without a parsable verdict in one of the two orders.

| Variant | Setup | Judge keys | Pairs with humans | Agreement % [95 % CI] | Kappa [95 % CI] |
|---|---|---|---|---|---|
| single order (model_1 first) | S1 | 394 | 755 | 52.7 [47.7, 57.5] | 0.248 [0.180, 0.315] |
| single order (model_1 first) | S2 | 394 | 561 | 69.0 [63.9, 74.0] | 0.385 [0.285, 0.484] |
| single order (model_2 first) | S1 | 394 | 755 | 58.1 [53.9, 62.2] | 0.330 [0.273, 0.384] |
| single order (model_2 first) | S2 | 394 | 562 | 75.4 [71.1, 79.7] | 0.508 [0.422, 0.590] |
| swap, disagreement -> tie | S1 | 394 | 755 | 54.4 [48.9, 59.9] | 0.331 [0.255, 0.407] |
| swap, disagreement -> tie | S2 | 394 | 354 | 85.0 [80.7, 89.1] | 0.699 [0.610, 0.780] |
| swap, abstain on disagreement | S1 | 233 | 435 | 70.6 [64.7, 75.9] | 0.504 [0.419, 0.583] |
| swap, abstain on disagreement | S2 | 233 | 354 | 85.0 [80.7, 89.0] | 0.699 [0.609, 0.778] |
| released GPT-4 judge (same keys) | S1 | 400 | 763 | 66.3 [61.5, 71.4] | 0.486 [0.415, 0.561] |
| released GPT-4 judge (same keys) | S2 | 400 | 483 | 84.7 [79.9, 89.2] | 0.694 [0.596, 0.783] |
| released GPT-4 judge (judge-consistent keys only) | S1 | 233 | 435 | 71.0 [65.2, 77.2] | 0.547 [0.456, 0.638] |
| released GPT-4 judge (judge-consistent keys only) | S2 | 233 | 306 | 86.3 [80.8, 91.2] | 0.723 [0.603, 0.822] |

Position bias (per call; A = the answer shown first):

- A wins 59.4 %, B wins 37.2 %, tie 3.4 % of 788 calls
- verdict changes when the order is swapped on 40.9 % of keys: 116 favour the first position both times, 28 the second, 17 involve a tie (released GPT-4 on the same keys: 15.8 %)

Length bias (decisive votes, longer answer at least 1.5x the words of the shorter):

| Voter | Votes | Longer answer wins % [95 % CI] |
|---|---|---|
| judge (every call) | 478 | 74.3 [68.1, 80.0] |
| judge (swap-consistent) | 172 | 83.7 [76.8, 90.8] |
| human experts | 369 | 73.2 [65.7, 80.4] |
| released GPT-4 judge | 201 | 79.1 [71.9, 86.0] |

## qwen2.5-7b

200 keys, 0 without a parsable verdict in one of the two orders.

| Variant | Setup | Judge keys | Pairs with humans | Agreement % [95 % CI] | Kappa [95 % CI] |
|---|---|---|---|---|---|
| single order (model_1 first) | S1 | 200 | 382 | 56.5 [50.0, 62.9] | 0.304 [0.215, 0.393] |
| single order (model_1 first) | S2 | 200 | 279 | 75.3 [67.9, 82.4] | 0.503 [0.359, 0.645] |
| single order (model_2 first) | S1 | 200 | 382 | 57.1 [50.2, 63.7] | 0.317 [0.221, 0.414] |
| single order (model_2 first) | S2 | 200 | 286 | 73.1 [65.8, 80.3] | 0.464 [0.321, 0.605] |
| swap, disagreement -> tie | S1 | 200 | 382 | 52.6 [45.4, 60.1] | 0.293 [0.189, 0.397] |
| swap, disagreement -> tie | S2 | 200 | 189 | 85.2 [78.6, 91.5] | 0.702 [0.563, 0.827] |
| swap, abstain on disagreement | S1 | 136 | 250 | 66.8 [58.8, 74.5] | 0.467 [0.356, 0.579] |
| swap, abstain on disagreement | S2 | 136 | 189 | 85.2 [78.0, 91.5] | 0.702 [0.559, 0.825] |
| released GPT-4 judge (same keys) | S1 | 200 | 382 | 68.1 [61.6, 74.2] | 0.510 [0.412, 0.604] |
| released GPT-4 judge (same keys) | S2 | 200 | 251 | 83.7 [76.9, 90.1] | 0.673 [0.535, 0.799] |
| released GPT-4 judge (judge-consistent keys only) | S1 | 136 | 250 | 74.0 [67.4, 80.2] | 0.596 [0.498, 0.683] |
| released GPT-4 judge (judge-consistent keys only) | S2 | 136 | 175 | 89.1 [83.0, 94.3] | 0.782 [0.659, 0.882] |

Position bias (per call; A = the answer shown first):

- A wins 57.8 %, B wins 37.8 %, tie 4.5 % of 400 calls
- verdict changes when the order is swapped on 32.0 % of keys: 48 favour the first position both times, 8 the second, 8 involve a tie (released GPT-4 on the same keys: 16.0 %)

Length bias (decisive votes, longer answer at least 1.5x the words of the shorter):

| Voter | Votes | Longer answer wins % [95 % CI] |
|---|---|---|
| judge (every call) | 226 | 69.0 [60.5, 76.5] |
| judge (swap-consistent) | 89 | 73.0 [61.9, 82.5] |
| human experts | 178 | 72.5 [62.5, 81.1] |
| released GPT-4 judge | 100 | 78.0 [69.4, 86.2] |
