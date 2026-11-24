"""Does routing between judges save money without losing agreement with the experts?

Replays the recorded verdicts of three judges on the keys they all judged (Llama-3.2-3B,
Qwen2.5-7B and the released GPT-4 votes) under several routing policies and compares
quality (mean per-key agreement with the expert votes, as in the regression gate) with
cost. Costs:

- tokens of the local runs, measured (``reports/judgments/tokens.csv``, exported from the
  git-ignored ``runs/`` by this command when they exist);
- GPT-4 token counts are *not* measured: the Qwen tokens for the same prompts stand in
  (different tokenizers, so only roughly right);
- prices per tier come from ``configs/mtbench.yaml`` (``router.prices``) and are
  illustrative hosted list prices, not what the local runs cost. A price-free view is the
  share of keys that reach the top tier.

Writes reports/router.md/.json and reports/figures/router_frontier.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llmeval.bias import judge_variants, load_judgments
from llmeval.config import load_config, resolve
from llmeval.errors import category
from llmeval.gate import paired_bootstrap
from llmeval.mtbench import KEY, load_split
from llmeval.router import (
    Policy,
    PolicyResult,
    evaluate_policy,
    human_shares,
    oracle,
    outcome_table,
    paired_delta,
    pareto,
    price_costs,
    random_mix,
    tier_scores,
)

LOCAL = ("llama-3.2-3b", "qwen2.5-7b")
TOP = "gpt4"
HARD = ("math", "reasoning", "coding")


def local_outcomes(j: pd.DataFrame) -> pd.DataFrame:
    v = judge_variants(j)["swap, disagreement -> tie"]
    return v.assign(consistent=v["forward"] == v["backward"])[KEY + ["outcome", "consistent"]]


def gpt4_outcomes(gpt4_raw: pd.DataFrame) -> pd.DataFrame:
    return gpt4_raw.assign(consistent=~gpt4_raw["inconsistent"])[KEY + ["outcome", "consistent"]]


def export_tokens(name: str, runs_dir: Path) -> pd.DataFrame:
    """Per-key tokens (both orders summed) and latency of a local run."""
    rows = []
    with open(runs_dir / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = r["completions"]
            rows.append({"question_id": r["question_id"], "turn": r["turn"],
                         "model_1": r["model_1"], "model_2": r["model_2"],
                         f"{name}__in": sum(x["input_tokens"] for x in c),
                         f"{name}__out": sum(x["output_tokens"] for x in c),
                         f"{name}__s": sum(x["latency_s"] for x in c)})
    return pd.DataFrame(rows)


def load_tokens(path: Path, runs_dir: Path) -> pd.DataFrame:
    if all((runs_dir / f"{n}.jsonl").exists() for n in LOCAL):
        tok = None
        for n in LOCAL:
            t = export_tokens(n, runs_dir)
            tok = t if tok is None else tok.merge(t, on=KEY, how="outer")
        path.parent.mkdir(parents=True, exist_ok=True)
        tok.to_csv(path, index=False)
        return tok
    return pd.read_csv(path)


def bootstrap_mean(x: np.ndarray, groups: np.ndarray, n_boot: int, seed: int):
    return paired_bootstrap(np.asarray(x, dtype=float), groups, n_boot, seed)


def matched_random(scores: np.ndarray, costs: np.ndarray, lo: int, hi: int, target: float,
                   groups: np.ndarray, own_scores: np.ndarray, n_boot: int, seed: int):
    """Policy score minus the random lo/hi mix with the same mean cost.

    Returns (difference, 95 % CI, mix share p). The mix probability is solved on each
    bootstrap resample (questions resampled together).
    """
    def solve(idx):
        c_lo, c_hi = costs[idx, lo].mean(), costs[idx, hi].mean()
        p = np.clip((target_cost(idx) - c_lo) / (c_hi - c_lo), 0, 1)
        s, _ = random_mix(scores[idx], costs[idx], lo, hi, p)
        return own_scores[idx].mean() - s, p

    def target_cost(idx):
        return target[idx].mean()

    full = np.arange(len(scores))
    d0, p0 = solve(full)
    uq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uq), len(uq))
        idx = np.concatenate([members[g] for g in pick])
        vals.append(solve(idx)[0])
    lo_ci, hi_ci = np.quantile(vals, [0.025, 0.975])
    return d0, (float(lo_ci), float(hi_ci)), float(p0)


def run_study(cfg: dict, n_boot: int, seed: int) -> dict:
    data_dir = resolve(cfg["dataset"]["local_dir"])
    human = load_split(data_dir, "human")
    gpt4 = gpt4_outcomes(load_split(data_dir, "gpt4_pair"))
    jd = resolve("reports/judgments")
    judged = {n: local_outcomes(load_judgments(jd / f"{n}.csv")) for n in LOCAL}
    judged[TOP] = gpt4
    tiers = [*LOCAL, TOP]
    table = outcome_table({t: judged[t] for t in tiers})
    shares = human_shares(human, table)
    scores = tier_scores(table, tiers, shares)
    groups = table["question_id"].to_numpy()
    cats = [category(q) for q in groups]

    tok = load_tokens(jd / "tokens.csv", resolve("runs")).merge(table[KEY], on=KEY)
    tok = table[KEY].merge(tok, on=KEY, how="left")
    tok[f"{TOP}__in"] = tok[f"{LOCAL[1]}__in"]          # estimate: Qwen tokens stand in
    tok[f"{TOP}__out"] = tok[f"{LOCAL[1]}__out"]
    prices = {t: tuple(cfg["router"]["prices"][t]) for t in tiers}
    usd = price_costs(tok, tiers, prices)

    def run(policies, sub):
        cols = [tiers.index(t) for t in sub]
        return [evaluate_policy(p, table, list(sub), scores[:, cols], usd[:, cols], cats)
                for p in policies]

    llama, qwen, top = tiers
    hard_start = tuple((c, 1) for c in HARD)             # index within the 2-tier subset
    fixed = run([Policy(f"always {t}", start=i, escalate_on="never")
                 for i, t in enumerate(tiers)], tiers)
    three = run([Policy("llama > qwen > gpt4"),
                 Policy("llama > qwen > gpt4 (tie escalates too)", escalate_on="flip_or_tie")],
                tiers)
    two = run([Policy("llama > gpt4"),
               Policy("llama > gpt4 (tie escalates too)", escalate_on="flip_or_tie"),
               Policy("llama > gpt4, maths/reasoning/coding straight to gpt4",
                      start_by_group=hard_start)], [llama, top])
    two_q = run([Policy("qwen > gpt4")], [qwen, top])
    results: list[PolicyResult] = fixed + three + two + two_q
    orc = oracle(scores, usd)

    def row(r: PolicyResult) -> dict:
        d = r.as_dict()
        d["score_ci"] = [round(v, 4) for v in bootstrap_mean(r.per_key_score, groups,
                                                              n_boot, seed)]
        d["cost_per_1k"] = round(r.cost * 1000, 2)
        return d

    rows = [row(r) for r in results] + [row(orc)]
    by_name = {r.name: r for r in results}
    gpt = by_name["always gpt4"]
    deltas = {}
    for n in ("llama > qwen > gpt4", "llama > gpt4", "qwen > gpt4",
              "llama > gpt4 (tie escalates too)"):
        r = by_name[n]
        d, ci = paired_delta(r.per_key_score, gpt.per_key_score, groups, n_boot, seed)
        saving = 1 - r.cost / gpt.cost
        deltas[n] = {"vs_gpt4_delta": round(d, 4), "ci": [round(v, 4) for v in ci],
                     "cost_saving": round(saving, 4)}
    control = {}
    for n, lo in (("llama > gpt4", 0), ("llama > qwen > gpt4", 0), ("qwen > gpt4", 1)):
        r = by_name[n]
        hi = 2
        sc, co = scores, usd
        d, ci, p = matched_random(sc, co, lo, hi, r.per_key_cost, groups, r.per_key_score,
                                  n_boot, seed)
        control[n] = {"score_minus_random_mix": round(d, 4),
                      "ci": [round(v, 4) for v in ci], "random_mix_p_top": round(p, 3)}

    cheap_cost = float(usd[:, 0].mean())
    out = {
        "n_keys": int(len(table)), "n_questions": int(len(np.unique(groups))),
        "prices_per_1m_tokens": {t: list(prices[t]) for t in tiers},
        "mean_tokens_per_key": {t: [round(float(tok[f"{t}__in"].mean()), 1),
                                    round(float(tok[f"{t}__out"].mean()), 1)] for t in tiers},
        "mean_local_seconds_per_key": {t: round(float(tok[f"{t}__s"].mean()), 2)
                                       for t in LOCAL},
        "tier_cost_ratio_vs_llama": {t: round(float(usd[:, i].mean() / cheap_cost), 1)
                                     for i, t in enumerate(tiers)},
        "policies": rows, "vs_always_gpt4": deltas, "vs_random_mix_same_cost": control,
        "pareto": pareto([(r["name"], r["cost"], r["score"]) for r in rows[:-1]]),
    }
    out["flip_rate_by_tier"] = {
        t: round(float((~table[f"{t}__sure"]).mean()), 4) for t in tiers}
    out["escalation_precision"] = _escalation_precision(table, tiers, scores)
    out["hard_categories_share"] = round(float(np.mean([c in HARD for c in cats])), 3)
    out["frontier"] = _frontier(scores, usd, tiers)
    return out


def _escalation_precision(table: pd.DataFrame, tiers, scores: np.ndarray) -> dict:
    """Is a flip a real sign of trouble? Score of each tier on sure vs. unsure keys."""
    res = {}
    for i, t in enumerate(tiers):
        sure = table[f"{t}__sure"].to_numpy()
        res[t] = {"sure_keys": int(sure.sum()), "unsure_keys": int((~sure).sum()),
                  "score_when_sure": round(float(scores[sure, i].mean()), 4)
                  if sure.any() else None,
                  "score_when_unsure": round(float(scores[~sure, i].mean()), 4)
                  if (~sure).any() else None}
    return res


def _frontier(scores: np.ndarray, usd: np.ndarray, tiers) -> dict:
    """Random-mix control line between each cheap tier and GPT-4."""
    out = {}
    for lo in (0, 1):
        out[f"{tiers[lo]} -> {tiers[2]}"] = [
            {"p": round(p, 2), "score": round(s, 4), "cost": round(c, 6)}
            for p, s, c in ((p, *random_mix(scores, usd, lo, 2, p))
                            for p in np.linspace(0, 1, 11))]
    return out


def to_markdown(res: dict) -> str:
    L = [
        "# Cost-aware judge router",
        "",
        f"{res['n_keys']} MT-Bench turn-1 keys ({res['n_questions']} questions) judged by all "
        "three tiers. Quality = mean per-key agreement with the expert votes (ties "
        "included, flip → tie). Intervals: bootstrap over questions.",
        "",
        "Prices per 1M tokens (input, output), illustrative hosted list prices, set in "
        "`configs/mtbench.yaml`: "
        + "; ".join(f"{t} {p[0]}/{p[1]}" for t, p in res["prices_per_1m_tokens"].items())
        + ". GPT-4 tokens are estimated from the Qwen tokens of the same prompts.",
        "",
        "| Policy | Agreement [95 % CI] | Cost per 1,000 keys (USD) | Keys reaching GPT-4 |"
        " Answered by llama / qwen / gpt4 (%) |",
        "|---|---|---|---|---|",
    ]
    for r in res["policies"]:
        ci = r.get("score_ci")
        sc = f"{100 * r['score']:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci \
            else f"{100 * r['score']:.1f}"
        sh = r["share_by_tier"]
        used = " / ".join(f"{100 * s:.0f}" for s in sh)
        L.append(f"| {r['name']} | {sc} | {r['cost_per_1k']:.2f} | "
                 f"{100 * r['share_top']:.0f} % | {used} |")
    L += ["", "## Against always using GPT-4", "",
          "| Policy | Agreement difference [95 % CI] | Cost saved |", "|---|---|---|"]
    for n, d in res["vs_always_gpt4"].items():
        L.append(f"| {n} | {100 * d['vs_gpt4_delta']:+.1f} "
                 f"[{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] | "
                 f"{100 * d['cost_saving']:.0f} % |")
    L += ["", "## Against a random mix at the same cost", "",
          "A router that knew nothing about the keys could reach the same average cost by "
          "sending a random share of keys to GPT-4 and the rest to the cheap tier.", "",
          "| Policy | Agreement minus random mix [95 % CI] | Random share to GPT-4 |",
          "|---|---|---|"]
    for n, d in res["vs_random_mix_same_cost"].items():
        L.append(f"| {n} | {100 * d['score_minus_random_mix']:+.1f} "
                 f"[{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] | "
                 f"{100 * d['random_mix_p_top']:.0f} % |")
    L += ["", "## Is a flip a sign of trouble?", "",
          "| Tier | Keys sure / unsure | Agreement when sure | Agreement when unsure |",
          "|---|---|---|---|"]
    for t, e in res["escalation_precision"].items():
        f = lambda v: "-" if v is None else f"{100 * v:.1f}"  # noqa: E731
        L.append(f"| {t} | {e['sure_keys']} / {e['unsure_keys']} | "
                 f"{f(e['score_when_sure'])} | {f(e['score_when_unsure'])} |")
    L += ["", "Mean tokens per key (input, output): "
          + "; ".join(f"{t} {v[0]:.0f}/{v[1]:.0f}"
                      for t, v in res["mean_tokens_per_key"].items())
          + ". Mean local time per key (both orders): "
          + "; ".join(f"{t} {s:.1f} s" for t, s in res["mean_local_seconds_per_key"].items())
          + ".", "", "Pareto-optimal policies (cost vs. agreement): "
          + "; ".join(res["pareto"]) + ".",
          "", "Cost of one key relative to the cheapest tier: "
          + ", ".join(f"{t} {v:g}x" for t, v in res["tier_cost_ratio_vs_llama"].items())
          + ". GPT-4 dominates the bill, so a policy's saving is roughly one minus the share "
          "of keys that reach it."]
    return "\n".join(L) + "\n"


def plot_frontier(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, pts in res["frontier"].items():
        ax.plot([p["cost"] * 1000 for p in pts], [100 * p["score"] for p in pts],
                ls="--", lw=1, label=f"random mix {name}")
    for r in res["policies"][:-1]:
        if "tie escalates" in r["name"]:      # same points as the plain cascade
            continue
        label = r["name"].replace("maths/reasoning/coding straight to gpt4", "hard cats. direct")
        ax.scatter(r["cost_per_1k"], 100 * r["score"], s=28, zorder=3)
        ax.annotate(label, (r["cost_per_1k"], 100 * r["score"]), fontsize=6,
                    xytext=(-4, -9), textcoords="offset points", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("cost per 1,000 keys (USD, illustrative prices, log scale)")
    ax.set_ylabel("agreement with experts (%)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate judge routing policies.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/router")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    res = run_study(cfg, args.n_boot, args.seed)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(res), encoding="utf-8")
    plot_frontier(res, resolve("reports/figures/router_frontier.png"))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
