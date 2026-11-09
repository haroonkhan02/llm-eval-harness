<h1 align="center">LLM Evaluation Harness</h1>

<p align="center">
  <a href="https://github.com/haroonkhan02/llm-eval-harness/actions/workflows/ci.yml"><img src="https://github.com/haroonkhan02/llm-eval-harness/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

A small library and CLI for evaluating LLM systems, with a focus on the judge itself:
how often does an LLM-as-judge agree with human experts, how biased is it, and when is a
drop in quality between two runs real rather than noise.

**Work in progress.** The MT-Bench data pipeline, the agreement statistics and the
pairwise judge (with A/B swapping, an Anthropic backend and a JSONL run store) work, and
the published GPT-4-judge vs. human agreement reproduces from the released votes. Next:
local open-weight judges via MLX, a position- and length-bias study, and a regression
gate that fails a pull request when quality drops.

## What it does

- Loads the 3,355 expert votes and 2,400 GPT-4 votes of the MT-Bench human judgments and
  maps every vote to a canonical (question, turn, model pair) key.
- Measures judge vs. human agreement (with and without ties), Cohen's kappa and 95 %
  intervals from a bootstrap over questions, since votes on one question are correlated.
- Judges a pair of answers twice with the order swapped and keeps the verdict only when
  both orders agree.
- Logs every model call (tokens, cost, latency, refusals) to a resumable JSONL store.

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
│   ├── backends.py             # Fake and Anthropic backends with cost and latency
│   └── store.py                # resumable JSONL run store
├── reports/                    # generated tables
└── tests/                      # synthetic votes and a mocked Anthropic client
```

## Getting started

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,anthropic]"
llmeval-download        # ~1.4 MB of parquet into data/mtbench
llmeval-stats           # reports/dataset_stats.md
llmeval-agreement       # reports/gpt4_human_agreement.md
pytest -q
```

The Claude judge requires an Anthropic API key (`ANTHROPIC_API_KEY`); tests mock it.

## License

MIT. The MT-Bench human judgments are released by LMSYS under CC-BY-4.0.
