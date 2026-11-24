# Cost-aware judge router

197 MT-Bench turn-1 keys (72 questions) judged by all three tiers. Quality = mean per-key agreement with the expert votes (ties included, flip → tie). Intervals: bootstrap over questions.

Prices per 1M tokens (input, output), illustrative hosted list prices, set in `configs/mtbench.yaml`: llama-3.2-3b 0.06/0.06; qwen2.5-7b 0.3/0.3; gpt4 30.0/60.0. GPT-4 tokens are estimated from the Qwen tokens of the same prompts.

| Policy | Agreement [95 % CI] | Cost per 1,000 keys (USD) | Keys reaching GPT-4 | Answered by llama / qwen / gpt4 (%) |
|---|---|---|---|---|
| always llama-3.2-3b | 55.0 [48.4, 62.0] | 0.10 | 0 % | 100 / 0 / 0 |
| always qwen2.5-7b | 56.8 [50.7, 63.3] | 0.40 | 0 % | 0 / 100 / 0 |
| always gpt4 | 67.9 [61.6, 74.5] | 45.07 | 100 % | 0 / 0 / 100 |
| llama > qwen > gpt4 | 65.1 [58.7, 71.7] | 10.47 | 20 % | 56 / 23 / 20 |
| llama > qwen > gpt4 (tie escalates too) | 65.1 [58.7, 71.7] | 11.22 | 22 % | 55 / 22 / 22 |
| llama > gpt4 | 67.1 [60.6, 73.6] | 20.11 | 44 % | 56 / 44 |
| llama > gpt4 (tie escalates too) | 67.1 [60.6, 73.6] | 20.50 | 45 % | 55 / 45 |
| llama > gpt4, maths/reasoning/coding straight to gpt4 | 68.5 [61.8, 74.8] | 28.02 | 61 % | 39 / 61 |
| qwen > gpt4 | 64.1 [58.0, 70.7] | 14.93 | 31 % | 69 / 31 |
| oracle (uses expert labels) | 80.8 [75.1, 86.2] | 5.72 | 14 % | 72 / 15 / 14 |

## Against always using GPT-4

| Policy | Agreement difference [95 % CI] | Cost saved |
|---|---|---|
| llama > qwen > gpt4 | -2.8 [-7.8, +2.1] | 77 % |
| llama > gpt4 | -0.9 [-4.9, +3.1] | 55 % |
| qwen > gpt4 | -3.9 [-8.0, +0.1] | 67 % |
| llama > gpt4 (tie escalates too) | -0.9 [-4.9, +3.1] | 55 % |

## Against a random mix at the same cost

A router that knew nothing about the keys could reach the same average cost by sending a random share of keys to GPT-4 and the rest to the cheap tier.

| Policy | Agreement minus random mix [95 % CI] | Random share to GPT-4 |
|---|---|---|
| llama > gpt4 | +6.3 [+2.2, +10.2] | 44 % |
| llama > qwen > gpt4 | +7.1 [+1.6, +12.8] | 23 % |
| qwen > gpt4 | +3.7 [+0.2, +6.8] | 32 % |

## Is a flip a sign of trouble?

| Tier | Keys sure / unsure | Agreement when sure | Agreement when unsure |
|---|---|---|---|
| llama-3.2-3b | 111 / 86 | 69.2 | 36.7 |
| qwen2.5-7b | 136 / 61 | 67.2 | 33.7 |
| gpt4 | 166 / 31 | 70.2 | 55.8 |

Mean tokens per key (input, output): llama-3.2-3b 1206/406; qwen2.5-7b 1178/162; gpt4 1178/162. Mean local time per key (both orders): llama-3.2-3b 7.2 s; qwen2.5-7b 9.3 s.

Pareto-optimal policies (cost vs. agreement): always llama-3.2-3b; always qwen2.5-7b; llama > qwen > gpt4; llama > gpt4; llama > gpt4, maths/reasoning/coding straight to gpt4.

Cost of one key relative to the cheapest tier: llama-3.2-3b 1x, qwen2.5-7b 4.2x, gpt4 466x. GPT-4 dominates the bill, so a policy's saving is roughly one minus the share of keys that reach it.
