"""Judge calibration and bias study on the keys a local judge has scored.

For each judge run (``reports/judgments/<name>.csv``) this measures, on the same keys:

- agreement with the human experts (S1 / S2, kappa, question-bootstrap CIs) for a single
  answer order and for the two swap mitigations (inconsistent -> tie, or abstain);
- position bias: how often the verdict flips when the answers are swapped, and whether
  the answer shown first wins more often than the one shown second;
- length bias: how often the longer answer wins, next to how often humans and the
  released GPT-4 judge picked the longer answer on the same keys.

Writes reports/judge_bias.md and .json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llmeval.config import load_config, resolve
from llmeval.metrics import pair_judge_vs_reference, summarise_pairs
from llmeval.mtbench import KEY, load_split

_FLIP = {"first": "second", "second": "first", "tie": "tie"}


def load_judgments(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("forward", "backward"):
        df[col] = df[col].where(df[col].notna(), None)
    return df


def judge_variants(j: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Judge votes (KEY + outcome) under the different ways of using the two orders."""
    ok = j[j["forward"].notna() & j["backward"].notna()]
    consistent = ok["forward"] == ok["backward"]
    swap_tie = ok.assign(outcome=ok["forward"].where(consistent, "tie"))
    return {
        "single order (model_1 first)": ok.assign(outcome=ok["forward"]),
        "single order (model_2 first)": ok.assign(outcome=ok["backward"]),
        "swap, disagreement -> tie": swap_tie,
        "swap, abstain on disagreement": ok[consistent].assign(outcome=ok["forward"]),
    }


def position_stats(j: pd.DataFrame) -> dict:
    """Verdicts in terms of the shown position (A = answer shown first)."""
    ok = j[j["forward"].notna() & j["backward"].notna()]
    # forward: model_1 shown as A; backward: model_2 shown as A.
    shown = pd.concat([
        ok["forward"].map({"first": "A", "second": "B", "tie": "C"}),
        ok["backward"].map({"second": "A", "first": "B", "tie": "C"}),
    ])
    incons = ok[ok["forward"] != ok["backward"]]
    both_a = ((incons["forward"] == "first") & (incons["backward"] == "second")).sum()
    both_b = ((incons["forward"] == "second") & (incons["backward"] == "first")).sum()
    n = len(ok)
    return {
        "keys": int(len(j)), "parsed": int(n), "unparsed": int(len(j) - n),
        "calls": int(len(shown)),
        "p_first_shown": round(float((shown == "A").mean()), 4) if n else None,
        "p_second_shown": round(float((shown == "B").mean()), 4) if n else None,
        "p_tie": round(float((shown == "C").mean()), 4) if n else None,
        "inconsistent_rate": round(float(len(incons) / n), 4) if n else None,
        "inconsistent_favour_first": int(both_a),
        "inconsistent_favour_second": int(both_b),
        "inconsistent_other": int(len(incons) - both_a - both_b),
    }


def _bootstrap_prop(success: np.ndarray, qids: np.ndarray, n_boot: int, seed: int):
    """Point estimate and question-cluster bootstrap CI of a proportion."""
    if success.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    uq, idx = np.unique(qids, return_inverse=True)
    s = np.bincount(idx, weights=success.astype(float), minlength=len(uq))
    n = np.bincount(idx, minlength=len(uq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uq), size=(n_boot, len(uq)))
    w = np.stack([np.bincount(d, minlength=len(uq)) for d in draws])
    vals = (w @ s) / (w @ n)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(success.mean()), (float(lo), float(hi))


def longer_wins(votes: pd.DataFrame, lengths: pd.DataFrame, min_ratio: float,
                n_boot: int, seed: int) -> dict:
    """Among decisive votes on keys whose answers differ in length, P(longer answer wins)."""
    v = votes[KEY + ["outcome"]].merge(lengths[KEY + ["len_1", "len_2"]], on=KEY)
    longer, shorter = np.maximum(v["len_1"], v["len_2"]), np.minimum(v["len_1"], v["len_2"])
    v = v[(longer >= min_ratio * shorter.clip(lower=1)) & (v["outcome"] != "tie")]
    first_longer = v["len_1"] > v["len_2"]
    win = (v["outcome"] == "first") == first_longer
    p, ci = _bootstrap_prop(win.to_numpy(), v["question_id"].to_numpy(), n_boot, seed)
    return {"n": int(len(v)), "p_longer_wins": round(p, 4),
            "ci": [round(ci[0], 4), round(ci[1], 4)]}


def analyse(j: pd.DataFrame, human: pd.DataFrame, gpt4: pd.DataFrame, n_boot: int,
            seed: int, min_ratio: float) -> dict:
    keys = j[KEY]
    h = human.merge(keys, on=KEY)
    g = gpt4.merge(keys, on=KEY)
    rows = []
    jv = judge_variants(j)
    consistent_keys = jv["swap, abstain on disagreement"][KEY]
    variants = {**jv, "released GPT-4 judge (same keys)": g,
                # Abstaining keeps the easier keys; compare GPT-4 on exactly those.
                "released GPT-4 judge (judge-consistent keys only)":
                    g.merge(consistent_keys, on=KEY)}
    for name, votes in variants.items():
        pairs = pair_judge_vs_reference(votes, h)
        for setup in ("S1", "S2"):
            r = summarise_pairs(pairs, setup, n_boot=n_boot, seed=seed).as_dict()
            rows.append({"variant": name, "judge_keys": int(len(votes)), **r})
    ok = j[j["forward"].notna() & j["backward"].notna()]
    calls = pd.concat([ok[KEY].assign(outcome=ok["forward"]),
                       ok[KEY].assign(outcome=ok["backward"])])
    length = {
        "judge (every call)": longer_wins(calls, j, min_ratio, n_boot, seed),
        "judge (swap-consistent)": longer_wins(
            jv["swap, abstain on disagreement"], j, min_ratio, n_boot, seed),
        "human experts": longer_wins(h, j, min_ratio, n_boot, seed),
        "released GPT-4 judge": longer_wins(g[~g["inconsistent"]], j, min_ratio, n_boot, seed),
    }
    return {"agreement": rows, "position": position_stats(j), "length": length}


def _pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def to_markdown(results: dict[str, dict], n_boot: int, min_ratio: float) -> str:
    out = ["# Local LLM judges vs. human experts: agreement and bias", "",
           "Each key (MT-Bench question, turn 1, model pair with human votes) was judged",
           "twice, once in each answer order. Agreement is computed on the same keys for",
           "every row, including the released GPT-4 verdicts. 95 % intervals: bootstrap",
           f"over questions ({n_boot} resamples).", ""]
    for name, r in results.items():
        p = r["position"]
        out += [f"## {name}", "",
                f"{p['keys']} keys, {p['unparsed']} without a parsable verdict in one of "
                "the two orders.", "",
                "| Variant | Setup | Judge keys | Pairs with humans | Agreement % [95 % CI] | "
                "Kappa [95 % CI] |", "|---|---|---|---|---|---|"]
        for a in r["agreement"]:
            out.append(f"| {a['variant']} | {a['setup']} | {a['judge_keys']} | {a['n_pairs']} | "
                       f"{_pct(a['agreement'])} [{_pct(a['ci'][0])}, {_pct(a['ci'][1])}] | "
                       f"{a['kappa']:.3f} [{a['kappa_ci'][0]:.3f}, {a['kappa_ci'][1]:.3f}] |")
        out += ["", "Position bias (per call; A = the answer shown first):", "",
                f"- A wins {_pct(p['p_first_shown'])} %, B wins {_pct(p['p_second_shown'])} %,"
                f" tie {_pct(p['p_tie'])} % of {p['calls']} calls",
                f"- verdict changes when the order is swapped on "
                f"{_pct(p['inconsistent_rate'])} % of keys: "
                f"{p['inconsistent_favour_first']} favour the first position both times, "
                f"{p['inconsistent_favour_second']} the second, "
                f"{p['inconsistent_other']} involve a tie", "",
                f"Length bias (decisive votes, longer answer at least {min_ratio:g}x the "
                "words of the shorter):", "",
                "| Voter | Votes | Longer answer wins % [95 % CI] |", "|---|---|---|"]
        for voter, lw in r["length"].items():
            out.append(f"| {voter} | {lw['n']} | {_pct(lw['p_longer_wins'])} "
                       f"[{_pct(lw['ci'][0])}, {_pct(lw['ci'][1])}] |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Agreement and bias of local judge runs.")
    ap.add_argument("names", nargs="+", help="run names under reports/judgments/")
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--out", default="reports/judge_bias")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_dir = resolve(cfg["dataset"]["local_dir"])
    n_boot, seed = int(cfg["agreement"]["n_boot"]), int(cfg["agreement"]["seed"])
    human = load_split(data_dir, "human")
    gpt4 = load_split(data_dir, "gpt4_pair")
    results = {}
    for name in args.names:
        j = load_judgments(resolve(f"reports/judgments/{name}.csv"))
        results[name] = analyse(j, human, gpt4, n_boot, seed, args.min_ratio)
    md = to_markdown(results, n_boot, args.min_ratio)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
