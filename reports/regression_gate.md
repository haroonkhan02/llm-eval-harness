# Regression gate on real judge runs

Score per item = share of expert votes on that MT-Bench key (turn 1) that match the judge's verdict. Gate: max drop 0.02, alpha 0.05; `drop` fails on a drop larger than the tolerance with a significant one-sided sign test, `noninferior` fails unless the lower one-sided 95 % bootstrap bound (questions resampled) is above minus the tolerance.

| Comparison | Items | Baseline | Candidate | Delta [90 % CI] | Better / worse | Sign p | drop | noninferior |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B, answer order swapped | 200 | 55.7 | 58.1 | +2.5 [-2.0, +6.7] | 25 / 21 | 0.769 | pass | FAIL |
| Llama-3.2-3B (400 keys), answer order swapped | 394 | 52.6 | 56.9 | +4.3 [+0.5, +8.3] | 69 / 50 | 0.967 | pass | pass |
| Llama-3.2-3B (400 keys), order swapped the other way | 394 | 56.9 | 52.6 | -4.3 [-8.3, -0.5] | 50 / 69 | 0.0493 | FAIL | FAIL |
| Qwen2.5-7B -> Llama-3.2-3B, swap and flip -> tie | 197 | 56.8 | 55.0 | -1.8 [-7.1, +2.9] | 29 / 29 | 0.552 | pass | FAIL |
| Qwen2.5-7B -> Llama-3.2-3B, one order | 197 | 55.5 | 51.2 | -4.3 [-9.0, -0.3] | 17 / 25 | 0.14 | pass | FAIL |
| GPT-4 -> Qwen2.5-7B, swap and flip -> tie | 200 | 67.9 | 56.5 | -11.5 [-17.7, -5.8] | 14 / 44 | 0.0001 | FAIL | FAIL |
| GPT-4 -> Llama-3.2-3B, swap and flip -> tie | 197 | 67.9 | 55.0 | -12.9 [-19.7, -6.2] | 22 / 51 | 0.0005 | FAIL | FAIL |

## Power on simulated regressions

Baseline: released GPT-4 per-key scores (1814 keys, turns 1-2), random subsets of n keys. Churn: 27 % of items re-scored at random in every candidate (measured between the two answer orders: Qwen 23 %, Llama 30 %). Then random correct items are set to 0 until the mean has dropped by the injected amount. 200 simulations per cell; share of candidates failed.

| n items | injected drop | fail rate, drop rule | fail rate, noninferior |
|---|---|---|---|
| 100 | 0 pts | 3 % | 84 % |
| 100 | 2 pts | 12 % | 99 % |
| 100 | 5 pts | 28 % | 100 % |
| 100 | 10 pts | 81 % | 100 % |
| 200 | 0 pts | 3 % | 76 % |
| 200 | 2 pts | 12 % | 98 % |
| 200 | 5 pts | 55 % | 100 % |
| 200 | 10 pts | 96 % | 100 % |
| 400 | 0 pts | 4 % | 59 % |
| 400 | 2 pts | 22 % | 96 % |
| 400 | 5 pts | 80 % | 100 % |
| 400 | 10 pts | 100 % | 100 % |
| 800 | 0 pts | 2 % | 40 % |
| 800 | 2 pts | 40 % | 98 % |
| 800 | 5 pts | 99 % | 100 % |
| 800 | 10 pts | 100 % | 100 % |
