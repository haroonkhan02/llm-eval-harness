# Local LLM judges vs. human experts: agreement and bias

Each key (MT-Bench question, turn 1, model pair with human votes) was judged
twice, once in each answer order. Agreement is computed on the same keys for
every row, including the released GPT-4 verdicts. 95 % intervals: bootstrap
over questions (2000 resamples).

## llama-3.2-3b

200 keys, 3 without a parsable verdict in one of the two orders.

| Variant | Setup | Judge keys | Pairs with humans | Agreement % [95 % CI] | Kappa [95 % CI] |
|---|---|---|---|---|---|
| single order (model_1 first) | S1 | 197 | 379 | 51.4 [44.9, 58.0] | 0.217 [0.131, 0.306] |
| single order (model_1 first) | S2 | 197 | 281 | 68.7 [61.5, 75.5] | 0.369 [0.232, 0.498] |
| single order (model_2 first) | S1 | 197 | 379 | 56.7 [50.6, 63.0] | 0.311 [0.228, 0.396] |
| single order (model_2 first) | S2 | 197 | 283 | 74.2 [67.5, 80.5] | 0.486 [0.353, 0.611] |
| swap, disagreement -> tie | S1 | 197 | 379 | 52.0 [45.1, 59.6] | 0.295 [0.200, 0.398] |
| swap, disagreement -> tie | S2 | 197 | 168 | 85.1 [78.8, 90.6] | 0.698 [0.564, 0.805] |
| swap, abstain on disagreement | S1 | 111 | 209 | 69.4 [61.4, 77.5] | 0.485 [0.373, 0.602] |
| swap, abstain on disagreement | S2 | 111 | 168 | 85.1 [79.1, 90.7] | 0.698 [0.567, 0.811] |
| released GPT-4 judge (same keys) | S1 | 200 | 382 | 68.1 [61.6, 74.2] | 0.510 [0.412, 0.604] |
| released GPT-4 judge (same keys) | S2 | 200 | 251 | 83.7 [76.9, 90.1] | 0.673 [0.535, 0.799] |
| released GPT-4 judge (judge-consistent keys only) | S1 | 111 | 209 | 70.3 [62.9, 78.3] | 0.531 [0.409, 0.648] |
| released GPT-4 judge (judge-consistent keys only) | S2 | 111 | 149 | 83.9 [75.2, 91.4] | 0.668 [0.473, 0.823] |

Position bias (per call; A = the answer shown first):

- A wins 60.4 %, B wins 36.8 %, tie 2.8 % of 394 calls
- verdict changes when the order is swapped on 43.6 % of keys: 63 favour the first position both times, 16 the second, 7 involve a tie (released GPT-4 on the same keys: 16.0 %)

Length bias (decisive votes, longer answer at least 1.5x the words of the shorter):

| Voter | Votes | Longer answer wins % [95 % CI] |
|---|---|---|
| judge (every call) | 225 | 73.3 [65.5, 80.6] |
| judge (swap-consistent) | 78 | 83.3 [72.3, 93.1] |
| human experts | 178 | 72.5 [62.5, 81.1] |
| released GPT-4 judge | 100 | 78.0 [69.4, 86.2] |

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
