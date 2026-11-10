<h1 align="center">LLM Evaluation Harness</h1>

<p align="center">
  <a href="https://github.com/haroonkhan02/llm-eval-harness/actions/workflows/ci.yml"><img src="https://github.com/haroonkhan02/llm-eval-harness/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

A small library and CLI for evaluating LLM systems, with a focus on the judge itself:
how often does an LLM-as-judge agree with human experts, how biased is it, and when is a
drop in quality between two runs real rather than noise.

**Work in progress.** The MT-Bench pipeline, the agreement statistics, the pairwise judge
(Anthropic and local MLX backends, A/B swapping, a JSONL run store) and a position- and
length-bias study of two local judges work. Next: error analysis of where the judges and
experts disagree, and a regression gate that fails a pull request when quality drops.

## What it does

- Loads the 3,355 expert votes and 2,400 GPT-4 votes of the MT-Bench human judgments and
  maps every vote to a canonical (question, turn, model pair) key.
- Measures judge vs. human agreement (with and without ties), Cohen's kappa and 95 %
  intervals from a bootstrap over questions, since votes on one question are correlated.
- Judges a pair of answers twice with the order swapped and keeps the verdict only when
  both orders agree.
- Logs every model call (tokens, cost, latency, refusals) to a resumable JSONL store.
- Runs open-weight judges locally on Apple silicon (MLX) and measures their position and
  length bias against the same expert votes.

## Sanity check: GPT-4 vs. human experts

Before trusting my own judges I recomputed Table 5 of the MT-Bench paper
(Zheng et al., 2023) from the released votes.

| Turn | Setup | GPT-4 vs. human | Human vs. human | Paper (GPT-4 / human) |
|---|---|---|---|---|
| 1 | with ties (S1) | 66.6 % [62.8, 70.5] | 64.1 % [59.6, 69.0] | 66 % / 63 % |
| 1 | without ties (S2) | 85.9 % [82.7, 89.0] | 82.0 % [77.3, 86.7] | 85 % / 81 % |
| 2 | with ties (S1) | 65.7 % [62.1, 69.2] | 67.0 % [62.7, 71.4] | 66 % / 67 % |
| 2 | without ties (S2) | 85.3 % [82.0, 88.5] | 83.4 % [79.8, 87.0] | 85 % / 82 % |

All eight cells land within 1.4 points of the paper, which validates the vote handling
and the agreement definition. My pair counts are higher than the paper's (for example
1,689 vs. 1,343 GPT-4/human pairs on turn 1); the paper does not say which votes it left
out, so the small gaps may come from that. GPT-4 agrees with experts about as often as
experts agree with each other, but the intervals are 6–10 points wide: 80 questions is a
small sample. GPT-4 also changed its verdict when the answer order was swapped on 15.8 %
of its pairwise judgments, which is the position bias the next part studies.
Full table with kappa: [reports/gpt4_human_agreement.md](reports/gpt4_human_agreement.md).

## Local judges: position bias and swapping

Two 4-bit open-weight models judged 200 turn-1 MT-Bench pairs (seeded sample of keys with
expert votes), each pair in both answer orders. Llama-3.2-3B also judged 200 more keys;
its 400-key numbers are in [reports/judge_bias.md](reports/judge_bias.md). Agreement is
with the expert votes on the same keys, ties included (S1).

| Judge (same 200 keys) | First answer wins / second | Verdict flips on swap | One order | Swap, flip → tie | Swap, abstain on flip (coverage) |
|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 60.4 / 36.8 % | 43.6 % | 51.4–56.7 % | 52.0 % | 69.4 % (56 %) |
| Qwen2.5-7B-Instruct | 57.8 / 37.8 % | 32.0 % | 56.5–57.1 % | 52.6 % | 66.8 % (68 %) |
| Released GPT-4 votes | – | 16.0 % | – | 68.1 % | 70.3 / 74.0 % on the same subsets |

Both local judges prefer whichever answer is shown first, and flip their verdict two to
three times as often as GPT-4 did. Swapping and turning flips into ties does not raise
agreement with ties included, because the forced ties rarely match an expert's vote.
Abstaining on flips does: on the pairs where a local judge is order-consistent it agrees
with experts about as often as GPT-4 does on those same pairs, but it only answers on
56–68 % of them, and those are the easier pairs. Without ties (S2), the abstaining judges
reach 85 % on that subset. Longer answers won 69–73 % of the local judges' decisive calls
against 72.5 % of expert votes (answers differing by at least 1.5x in words), so on this
sample there is no sign of a length bias beyond what experts show. The intervals are wide
(about ±7 points), so the 3B vs. 7B differences are not significant. Full tables with
kappa and S2: [reports/judge_bias_common_keys.md](reports/judge_bias_common_keys.md).

## Repository structure

```text
llm-eval-harness/
├── configs/mtbench.yaml        # dataset revision, bootstrap settings, judge model
├── src/llmeval/
│   ├── download.py             # llmeval-download: MT-Bench parquet at a pinned revision
│   ├── mtbench.py              # canonical vote loader, llmeval-stats
│   ├── metrics.py              # agreement, kappa, cluster bootstrap
│   ├── agreement.py            # llmeval-agreement: GPT-4 vs. human sanity check
│   ├── judge.py                # prompt, [[A]]/[[B]]/[[C]] parser, swap-and-aggregate
│   ├── backends.py             # Fake, Anthropic and MLX backends with cost and latency
│   ├── store.py                # resumable JSONL run store
│   ├── run_judge.py            # llmeval-judge: judge pairs in both orders
│   └── bias.py                 # llmeval-bias: agreement, position and length bias
├── reports/                    # generated tables
│   └── judgments/              # compact verdicts of each judge run (no model text)
└── tests/                      # synthetic votes and a mocked Anthropic client
```

## Getting started

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,anthropic,mlx]"
llmeval-download        # ~1.4 MB of parquet into data/mtbench
llmeval-stats           # reports/dataset_stats.md
llmeval-agreement       # reports/gpt4_human_agreement.md
llmeval-judge --backend mlx:mlx-community/Qwen2.5-7B-Instruct-4bit \
              --name qwen2.5-7b --n-keys 200      # about 10 s per pair on an M5
llmeval-bias llama-3.2-3b qwen2.5-7b --common-keys --out reports/judge_bias_common_keys
pytest -q
```

The Claude judge requires an Anthropic API key (`ANTHROPIC_API_KEY`); tests mock it.

## License

MIT. The MT-Bench human judgments are released by LMSYS under CC-BY-4.0.
