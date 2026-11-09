"""Sanity check: GPT-4-as-judge vs. human expert agreement on MT-Bench.

Reproduces the "G4-Pair vs. Human" and "Human vs. Human" cells of Table 5 in Zheng et al.
(2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena") from the released votes,
and adds Cohen's kappa and 95 % cluster-bootstrap intervals over questions.

Writes reports/gpt4_human_agreement.md and .json.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from llmeval.config import load_config, resolve
from llmeval.metrics import pair_judge_vs_reference, pair_within, summarise_pairs
from llmeval.mtbench import load_split

# Table 5 of the MT-Bench paper (arXiv:2306.05685v4): agreement and number of votes.
PAPER = {
    ("G4-Pair vs Human", 1, "S1"): (0.66, 1343), ("G4-Pair vs Human", 1, "S2"): (0.85, 859),
    ("G4-Pair vs Human", 2, "S1"): (0.66, 1325), ("G4-Pair vs Human", 2, "S2"): (0.85, 864),
    ("Human vs Human", 1, "S1"): (0.63, 721), ("Human vs Human", 1, "S2"): (0.81, 479),
    ("Human vs Human", 2, "S1"): (0.67, 707), ("Human vs Human", 2, "S2"): (0.82, 474),
}


def compute(human: pd.DataFrame, gpt4: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    rows = []
    for turn in (1, 2, "all"):
        h = human if turn == "all" else human[human["turn"] == turn]
        g = gpt4 if turn == "all" else gpt4[gpt4["turn"] == turn]
        comparisons = {
            "G4-Pair vs Human": (pair_judge_vs_reference(g, h), 1),
            # Ordered pairs keep kappa symmetric; each unordered pair is counted twice.
            "Human vs Human": (pair_within(h), 2),
        }
        for name, (pairs, dup) in comparisons.items():
            for setup in ("S1", "S2"):
                r = summarise_pairs(pairs, setup, n_boot=n_boot, seed=seed).as_dict()
                r["n_pairs"] //= dup
                paper = PAPER.get((name, turn, setup))
                rows.append({"comparison": name, "turn": turn, **r,
                             "paper_agreement": paper[0] if paper else None,
                             "paper_votes": paper[1] if paper else None})
    return rows


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def to_markdown(rows: list[dict], inconsistent_rate: float, n_boot: int) -> str:
    lines = [
        "# GPT-4 judge vs. human experts on MT-Bench",
        "",
        "Agreement = probability that a vote of each type agrees on the same (question, turn,",
        "model pair). S1 keeps ties (GPT-4 'inconsistent' verdicts count as ties); S2 drops",
        "every pair where either side voted tie. 95 % intervals: bootstrap over the 80",
        f"questions ({n_boot} resamples). Kappa uses each side's own marginals.",
        "",
        "| Comparison | Turn | Setup | Pairs | Agreement % [95 % CI] | Chance % | Kappa [95 % CI]"
        " | Paper % (votes) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        paper = (f"{100 * r['paper_agreement']:.0f} ({r['paper_votes']})"
                 if r["paper_agreement"] is not None else "–")
        lines.append(
            f"| {r['comparison']} | {r['turn']} | {r['setup']} | {r['n_pairs']} | "
            f"{_pct(r['agreement'])} [{_pct(r['ci'][0])}, {_pct(r['ci'][1])}] | "
            f"{_pct(r['chance'])} | {r['kappa']:.3f} [{r['kappa_ci'][0]:.3f}, "
            f"{r['kappa_ci'][1]:.3f}] | {paper} |"
        )
    lines += [
        "",
        f"GPT-4 changed its verdict when the answer order was swapped on "
        f"{_pct(inconsistent_rate)} % of its 2,400 pairwise judgments.",
        "",
        "Paper: Zheng et al. 2023, arXiv:2306.05685v4, Table 5 (G4-Pair and Human rows).",
        "Human-vs-human pair counts are unordered pairs of different judges on one key.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="reports/gpt4_human_agreement")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_dir = resolve(cfg["dataset"]["local_dir"])
    acfg = cfg["agreement"]
    human = load_split(data_dir, "human")
    gpt4 = load_split(data_dir, "gpt4_pair", inconsistent_as=acfg.get("inconsistent_as", "tie"))
    rows = compute(human, gpt4, int(acfg["n_boot"]), int(acfg["seed"]))
    incons = float(gpt4["inconsistent"].mean())
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows, incons, int(acfg["n_boot"]))
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "gpt4_inconsistent_rate": round(incons, 4)}, indent=2),
        encoding="utf-8",
    )
    print(md)


if __name__ == "__main__":
    main()
