<h1 align="center">LLM Evaluation Harness</h1>

<p align="center">
  <a href="https://github.com/haroonkhan02/llm-eval-harness/actions/workflows/ci.yml"><img src="https://github.com/haroonkhan02/llm-eval-harness/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

A small library and CLI for evaluating LLM systems, with a focus on the judge itself:
how often does an LLM-as-judge agree with human experts, how biased is it, and when is a
drop in quality between two runs real rather than noise.

**Work in progress.** The repository is scaffolded with config, CI and tests. Next: the
MT-Bench data pipeline, agreement metrics and a check that reproduces the published
GPT-4-judge vs. human agreement.

## License

MIT. The MT-Bench human judgments are released by LMSYS under CC-BY-4.0.
