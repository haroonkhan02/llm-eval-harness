"""How well does the regression gate work on real judge runs?

Two parts, written to reports/regression_gate.md/.json:

1. Real comparisons. Every score is agreement with the expert votes per MT-Bench key.
   - same judge, answer order swapped (nothing about the judge changed, so a good gate
     should pass);
   - a cheaper judge (Qwen2.5-7B -> Llama-3.2-3B) and local judges vs. released GPT-4.
2. Power simulation on real scores. Baseline = released GPT-4 per-key scores. A simulated
   candidate re-scores a share of the items at random (run-to-run churn, mean-preserving;
   the share is measured from the local judges' two answer orders) and then loses ``d``
   points by setting random correct items to 0. Reports how often each rule fails the
   candidate for several sample sizes and drops. The drop is injected, not observed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from llmeval.bias import judge_variants, load_judgments
from llmeval.config import load_config, resolve
from llmeval.gate import compare, judge_scores
from llmeval.mtbench import KEY, load_split

_VARIANT = {"forward": "single order (model_1 first)",
            "backward": "single order (model_2 first)",
            "swap-tie": "swap, disagreement -> tie"}


def run_scores(j: pd.DataFrame, human: pd.DataFrame, variant: str) -> pd.DataFrame:
    return judge_scores(judge_variants(j)[_VARIANT[variant]], human)


def churn(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Share of common items whose score differs between two runs."""
    m = a.merge(b, on="id")
    return float((m["score_x"] != m["score_y"]).mean())


def inject(base: np.ndarray, churn_rate: float, drop: float,
           rng: np.random.Generator) -> np.ndarray:
    """Candidate scores: random re-scoring of ``churn_rate`` of the items, then a drop."""
    cand = base.copy()
    n = base.size
    k = int(round(churn_rate * n))
    idx = rng.choice(n, k, replace=False)
    cand[idx] = rng.choice(base, k, replace=True)          # keeps the mean on average
    need = drop * n
    for i in rng.permutation(np.flatnonzero(cand > 0)):
        if need <= 0:
            break
        need -= cand[i]
        cand[i] = 0.0
    return cand


def power(pop: pd.DataFrame, sizes, drops, churn_rate: float, max_drop: float,
          alpha: float, n_sims: int, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        for d in drops:
            fails = {"drop": 0, "noninferior": 0}
            for _ in range(n_sims):
                s = pop.iloc[rng.choice(len(pop), n, replace=False)].reset_index(drop=True)
                cand = s.assign(score=inject(s["score"].to_numpy(float), churn_rate, d, rng))
                for rule in fails:
                    r = compare(s, cand, max_drop, alpha, rule, n_boot,
                                seed=int(rng.integers(1 << 31)))
                    fails[rule] += not r.passed
            out.append({"n_items": n, "injected_drop": d,
                        **{f"fail_rate_{k}": round(v / n_sims, 3) for k, v in fails.items()}})
    return out


def comparisons(runs: dict[str, pd.DataFrame], human: pd.DataFrame, gpt4: pd.DataFrame,
                max_drop: float, alpha: float, n_boot: int) -> list[dict]:
    qwen, llama = runs["qwen2.5-7b"], runs["llama-3.2-3b"]
    common = qwen[KEY].merge(llama[KEY], on=KEY)
    llama_c = llama.merge(common, on=KEY)
    g200 = judge_scores(gpt4.merge(common, on=KEY), human)
    cases = [
        ("Qwen2.5-7B, answer order swapped", "same judge",
         run_scores(qwen, human, "forward"), run_scores(qwen, human, "backward")),
        ("Llama-3.2-3B (400 keys), answer order swapped", "same judge",
         run_scores(llama, human, "forward"), run_scores(llama, human, "backward")),
        ("Llama-3.2-3B (400 keys), order swapped the other way", "same judge",
         run_scores(llama, human, "backward"), run_scores(llama, human, "forward")),
        ("Qwen2.5-7B -> Llama-3.2-3B, swap and flip -> tie", "cheaper judge",
         run_scores(qwen, human, "swap-tie"), run_scores(llama_c, human, "swap-tie")),
        ("Qwen2.5-7B -> Llama-3.2-3B, one order", "cheaper judge",
         run_scores(qwen, human, "forward"), run_scores(llama_c, human, "forward")),
        ("GPT-4 -> Qwen2.5-7B, swap and flip -> tie", "local vs. GPT-4",
         g200, run_scores(qwen, human, "swap-tie")),
        ("GPT-4 -> Llama-3.2-3B, swap and flip -> tie", "local vs. GPT-4",
         g200, run_scores(llama_c, human, "swap-tie")),
    ]
    out = []
    for name, kind, base, cand in cases:
        row = {"comparison": name, "kind": kind}
        for rule in ("drop", "noninferior"):
            r = compare(base, cand, max_drop, alpha, rule, n_boot, allow_missing=True)
            row[rule] = r.as_dict()
        out.append(row)
    return out


def to_markdown(res: dict) -> str:
    p = res["params"]
    out = ["# Regression gate on real judge runs", "",
           "Score per item = share of expert votes on that MT-Bench key (turn 1) that match "
           f"the judge's verdict. Gate: max drop {p['max_drop']}, alpha {p['alpha']}; "
           "`drop` fails on a drop larger than the tolerance with a significant one-sided "
           "sign test, `noninferior` fails unless the lower one-sided 95 % bootstrap bound "
           "(questions resampled) is above minus the tolerance.", "",
           "| Comparison | Items | Baseline | Candidate | Delta [90 % CI] | Better / worse "
           "| Sign p | drop | noninferior |", "|---|---|---|---|---|---|---|---|---|"]
    for c in res["comparisons"]:
        d, ni = c["drop"], c["noninferior"]
        out.append(
            f"| {c['comparison']} | {d['n_items']} | {100 * d['baseline_mean']:.1f} | "
            f"{100 * d['candidate_mean']:.1f} | {100 * d['delta']:+.1f} "
            f"[{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] | {d['n_better']} / "
            f"{d['n_worse']} | {d['p_worse']:.3g} | {'pass' if d['passed'] else 'FAIL'} | "
            f"{'pass' if ni['passed'] else 'FAIL'} |")
    c = res["churn"]
    out += ["", "## Power on simulated regressions", "",
            f"Baseline: released GPT-4 per-key scores ({p['population']} keys, turns 1-2), "
            f"random subsets of n keys. Churn: {100 * c['used']:.0f} % of items re-scored at "
            f"random in every candidate (measured between the two answer orders: Qwen "
            f"{100 * c['qwen2.5-7b']:.0f} %, Llama {100 * c['llama-3.2-3b']:.0f} %). Then "
            "random correct items are set to 0 until the mean has dropped by the injected "
            f"amount. {p['n_sims']} simulations per cell; share of candidates failed.", "",
            "| n items | injected drop | fail rate, drop rule | fail rate, noninferior |",
            "|---|---|---|---|"]
    for r in res["power"]:
        fd, fn = 100 * r["fail_rate_drop"], 100 * r["fail_rate_noninferior"]
        out.append(f"| {r['n_items']} | {100 * r['injected_drop']:.0f} pts | "
                   f"{fd:.0f} % | {fn:.0f} % |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Regression gate on real runs + power study.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--sizes", default="100,200,400,800")
    ap.add_argument("--drops", default="0,0.02,0.05,0.1")
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="reports/regression_gate")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_dir = resolve(cfg["dataset"]["local_dir"])
    seed = int(cfg["agreement"]["seed"])
    human, gpt4 = load_split(data_dir, "human"), load_split(data_dir, "gpt4_pair")
    runs = {n: load_judgments(resolve(f"reports/judgments/{n}.csv"))
            for n in ("qwen2.5-7b", "llama-3.2-3b")}
    ch = {n: round(churn(run_scores(j, human, "forward"), run_scores(j, human, "backward")), 4)
          for n, j in runs.items()}
    ch["used"] = round(float(np.mean(list(ch.values()))), 4)
    pop = judge_scores(gpt4, human)
    res = {
        "params": {"max_drop": args.max_drop, "alpha": args.alpha, "n_sims": args.n_sims,
                   "n_boot": args.n_boot, "population": len(pop)},
        "comparisons": comparisons(runs, human, gpt4, args.max_drop, args.alpha, args.n_boot),
        "churn": ch,
        "power": power(pop, [int(s) for s in args.sizes.split(",")],
                       [float(d) for d in args.drops.split(",")], ch["used"], args.max_drop,
                       args.alpha, args.n_sims, args.n_boot, seed),
    }
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(to_markdown(res), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(to_markdown(res))


if __name__ == "__main__":
    main()
