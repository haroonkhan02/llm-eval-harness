# GPT-4 judge vs. human experts on MT-Bench

Agreement = probability that a vote of each type agrees on the same (question, turn,
model pair). S1 keeps ties (GPT-4 'inconsistent' verdicts count as ties); S2 drops
every pair where either side voted tie. 95 % intervals: bootstrap over the 80
questions (2000 resamples). Kappa uses each side's own marginals.

| Comparison | Turn | Setup | Pairs | Agreement % [95 % CI] | Chance % | Kappa [95 % CI] | Paper % (votes) |
|---|---|---|---|---|---|---|---|
| G4-Pair vs Human | 1 | S1 | 1689 | 66.6 [62.8, 70.5] | 34.4 | 0.491 [0.435, 0.549] | 66 (1343) |
| G4-Pair vs Human | 1 | S2 | 1060 | 85.9 [82.7, 89.0] | 50.0 | 0.719 [0.654, 0.780] | 85 (859) |
| Human vs Human | 1 | S1 | 1198 | 64.1 [59.6, 69.0] | 34.8 | 0.449 [0.377, 0.522] | 63 (721) |
| Human vs Human | 1 | S2 | 772 | 82.0 [77.3, 86.7] | 50.0 | 0.640 [0.541, 0.734] | 81 (479) |
| G4-Pair vs Human | 2 | S1 | 1666 | 65.7 [62.1, 69.2] | 34.9 | 0.473 [0.420, 0.524] | 66 (1325) |
| G4-Pair vs Human | 2 | S2 | 1069 | 85.3 [82.0, 88.5] | 50.3 | 0.704 [0.639, 0.768] | 85 (864) |
| Human vs Human | 2 | S1 | 1169 | 67.0 [62.7, 71.4] | 35.4 | 0.489 [0.421, 0.556] | 67 (707) |
| Human vs Human | 2 | S2 | 778 | 83.4 [79.8, 87.0] | 50.5 | 0.665 [0.587, 0.739] | 82 (474) |
| G4-Pair vs Human | all | S1 | 3355 | 66.2 [63.2, 69.1] | 34.6 | 0.483 [0.439, 0.524] | – |
| G4-Pair vs Human | all | S2 | 2129 | 85.6 [83.1, 88.1] | 50.1 | 0.712 [0.662, 0.762] | – |
| Human vs Human | all | S1 | 2367 | 65.5 [62.1, 69.1] | 35.0 | 0.470 [0.414, 0.523] | – |
| Human vs Human | all | S2 | 1550 | 82.7 [79.5, 86.0] | 50.1 | 0.653 [0.587, 0.719] | – |

GPT-4 changed its verdict when the answer order was swapped on 15.8 % of its 2,400 pairwise judgments.

Paper: Zheng et al. 2023, arXiv:2306.05685v4, Table 5 (G4-Pair and Human rows).
Human-vs-human pair counts are unordered pairs of different judges on one key.
