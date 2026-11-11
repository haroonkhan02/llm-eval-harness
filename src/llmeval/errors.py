"""Error analysis: where do judges disagree with the human experts?

Breaks judge vs. expert agreement (S1, ties included) down by:

- MT-Bench category (question ids 81-160 come in eight blocks of ten);
- the kind of expert vote (tie or decisive) and the judge's confusion matrix;
- expert consensus on keys with two or more expert votes (unanimous vs. split);
- the strength gap of the model pair (difference in the models' overall expert win rate).

Judges: the local runs in ``reports/judgments/`` (swap with disagreement -> tie, and swap
with abstention) and the released GPT-4 votes on the same turn-1 keys and on all keys.
Writes reports/error_analysis.md/.json and reports/figures/agreement_by_category.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llmeval.bias import _bootstrap_prop, judge_variants, load_judgments
from llmeval.config import load_config, resolve
from llmeval.metrics import pair_judge_vs_reference
from llmeval.mtbench import KEY, OUTCOMES, load_split

CATEGORIES = ("writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem",
              "humanities")


def category(qid: int) -> str:
    """MT-Bench category of a question id (81-90 writing, ..., 151-160 humanities)."""
    i = (int(qid) - 81) // 10
    if not 0 <= i < len(CATEGORIES):
        raise ValueError(f"question id {qid} is outside MT-Bench 81-160")
    return CATEGORIES[i]


def model_strength(human: pd.DataFrame) -> pd.Series:
    """Each model's expert win rate over all its votes (a tie counts as half a win)."""
    rows = []
    for col, win in (("model_1", "first"), ("model_2", "second")):
        s = human[[col, "outcome"]].rename(columns={col: "model"})
        s = s.assign(score=np.where(s["outcome"] == win, 1.0,
                                    np.where(s["outcome"] == "tie", 0.5, 0.0)))
        rows.append(s[["model", "score"]])
    return pd.concat(rows).groupby("model")["score"].mean()


def annotate(pairs: pd.DataFrame, human: pd.DataFrame, strength: pd.Series,
             gap_edges: tuple[float, ...]) -> pd.DataFrame:
    """Add category, expert-consensus and strength-gap columns to judge/human pairs."""
    per_key = human.groupby(KEY)["outcome"].agg(n_votes="size", n_distinct="nunique")
    p = pairs.merge(per_key.reset_index(), on=KEY, how="left")
    gap = (p["model_1"].map(strength) - p["model_2"].map(strength)).abs()
    labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(gap_edges[:-1], gap_edges[1:],
                                                       strict=True)]
    return p.assign(
        category=p["question_id"].map(category),
        consensus=np.where(p["n_votes"] < 2, "single vote",
                           np.where(p["n_distinct"] == 1, "experts unanimous",
                                    "experts split")),
        expert_vote=np.where(p["y"] == "tie", "expert tie", "expert decisive"),
        gap=pd.cut(gap, bins=list(gap_edges), labels=labels, include_lowest=True),
        agree=(p["x"] == p["y"]),
    )


def breakdown(p: pd.DataFrame, by: str, n_boot: int, seed: int) -> list[dict]:
    """S1 agreement with a question-bootstrap CI for every level of column ``by``."""
    out = []
    levels = p[by].cat.categories if hasattr(p[by], "cat") else sorted(p[by].unique())
    for level in levels:
        s = p[p[by] == level]
        if s.empty:
            continue
        a, ci = _bootstrap_prop(s["agree"].to_numpy(), s["question_id"].to_numpy(),
                                n_boot, seed)
        out.append({"level": str(level), "n_pairs": int(len(s)),
                    "n_questions": int(s["question_id"].nunique()),
                    "agreement": round(a, 4), "ci": [round(ci[0], 4), round(ci[1], 4)]})
    return out


def confusion(p: pd.DataFrame) -> list[list[int]]:
    """Rows: judge first/second/tie; columns: expert first/second/tie."""
    m = pd.crosstab(p["x"], p["y"]).reindex(index=OUTCOMES, columns=OUTCOMES, fill_value=0)
    return m.to_numpy().astype(int).tolist()


def flips_by_category(j: pd.DataFrame) -> dict[str, dict]:
    ok = j[j["forward"].notna() & j["backward"].notna()]
    ok = ok.assign(category=ok["question_id"].map(category),
                   flip=ok["forward"] != ok["backward"])
    g = ok.groupby("category")["flip"].agg(["mean", "size"])
    return {c: {"flip_rate": round(float(g.loc[c, "mean"]), 4), "keys": int(g.loc[c, "size"])}
            for c in CATEGORIES if c in g.index}


def judge_sets(runs: dict[str, pd.DataFrame], gpt4: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Judge votes to analyse, named for the report."""
    sets = {}
    for name, j in runs.items():
        v = judge_variants(j)
        sets[f"{name} (swap, flip -> tie)"] = v["swap, disagreement -> tie"]
        sets[f"{name} (swap, abstain)"] = v["swap, abstain on disagreement"]
    common = None
    for j in runs.values():
        common = j[KEY] if common is None else common.merge(j[KEY], on=KEY)
    if common is not None:
        sets["GPT-4 (same keys as the local judges)"] = gpt4.merge(common, on=KEY)
    sets["GPT-4 (all turn-1 keys)"] = gpt4[gpt4["turn"] == 1]
    sets["GPT-4 (all keys, turns 1-2)"] = gpt4
    return sets


def analyse(runs: dict[str, pd.DataFrame], human: pd.DataFrame, gpt4: pd.DataFrame,
            n_boot: int, seed: int, gap_edges: tuple[float, ...]) -> dict:
    strength = model_strength(human)
    res = {"model_strength": {m: round(float(s), 4) for m, s in strength.items()},
           "judges": {}, "flips_by_category": {n: flips_by_category(j)
                                               for n, j in runs.items()}}
    for name, votes in judge_sets(runs, gpt4).items():
        p = annotate(pair_judge_vs_reference(votes, human), human, strength, gap_edges)
        p["category"] = pd.Categorical(p["category"], categories=CATEGORIES)
        res["judges"][name] = {
            "n_pairs": int(len(p)),
            "agreement": round(float(p["agree"].mean()), 4),
            "by_category": breakdown(p, "category", n_boot, seed),
            "by_expert_vote": breakdown(p, "expert_vote", n_boot, seed),
            "by_consensus": breakdown(p, "consensus", n_boot, seed),
            "by_gap": breakdown(p, "gap", n_boot, seed),
            "confusion": confusion(p),
        }
    return res


def _cell(r: dict | None) -> str:
    if r is None:
        return "–"
    lo, hi = r["ci"]
    return f"{100 * r['agreement']:.0f} [{100 * lo:.0f}, {100 * hi:.0f}] ({r['n_pairs']})"


def _table(res: dict, part: str, levels: list[str], title: str) -> list[str]:
    names = list(res["judges"])
    out = [f"## {title}", "", "| Judge | " + " | ".join(levels) + " |",
           "|---|" + "---|" * len(levels)]
    for n in names:
        by = {r["level"]: r for r in res["judges"][n][part]}
        out.append(f"| {n} | " + " | ".join(_cell(by.get(lv)) for lv in levels) + " |")
    return out + [""]


def to_markdown(res: dict, n_boot: int) -> str:
    first = next(iter(res["judges"].values()))
    gaps = [r["level"] for r in first["by_gap"]]
    out = ["# Where judges disagree with human experts", "",
           "Agreement with expert votes, ties included (S1), as % with a 95 % bootstrap "
           f"interval over questions ({n_boot} resamples) and the number of judge/expert "
           "vote pairs in brackets. Turn 1 unless stated. Categories have only 10 "
           "questions each, so their intervals are wide.", ""]
    out += _table(res, "by_category", list(CATEGORIES), "By MT-Bench category")
    out += _table(res, "by_expert_vote", ["expert decisive", "expert tie"],
                  "By the expert's vote")
    out += _table(res, "by_consensus", ["experts unanimous", "experts split", "single vote"],
                  "By expert consensus (keys with 2+ expert votes)")
    out += _table(res, "by_gap", gaps, "By strength gap of the model pair")
    out += ["Strength = a model's expert win rate over all its votes (tie = half): " +
            ", ".join(f"{m} {100 * s:.0f} %" for m, s in
                      sorted(res["model_strength"].items(), key=lambda kv: -kv[1])) + ".",
            "", "## Confusion matrices (rows: judge, columns: expert; first / second / tie)",
            ""]
    for n, r in res["judges"].items():
        c = r["confusion"]
        out.append(f"- {n}: " + "; ".join(" / ".join(str(v) for v in row) for row in c))
    out += ["", "## Verdict flips under A/B swap by category (local judges)", "",
            "| Judge | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for n, f in res["flips_by_category"].items():
        out.append(f"| {n} | " + " | ".join(
            f"{100 * f[c]['flip_rate']:.0f} ({f[c]['keys']})" if c in f else "–"
            for c in CATEGORIES) + " |")
    return "\n".join(out) + "\n"


def plot_categories(res: dict, path: Path, names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.8 / len(names)
    x = np.arange(len(CATEGORIES))
    for i, n in enumerate(names):
        by = {r["level"]: r for r in res["judges"][n]["by_category"]}
        a = np.array([100 * by[c]["agreement"] for c in CATEGORIES])
        lo = np.array([100 * by[c]["ci"][0] for c in CATEGORIES])
        hi = np.array([100 * by[c]["ci"][1] for c in CATEGORIES])
        ax.bar(x + (i - (len(names) - 1) / 2) * width, a, width, label=n,
               yerr=[a - lo, hi - a], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(100 / 3, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x, CATEGORIES)
    ax.set_ylabel("agreement with experts, S1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=len(names), loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Break judge/expert disagreement down.")
    ap.add_argument("names", nargs="+", help="run names under reports/judgments/")
    ap.add_argument("--config", default=None)
    ap.add_argument("--gap-edges", default="0,0.15,0.3,1",
                    help="bin edges for the model-pair strength gap")
    ap.add_argument("--out", default="reports/error_analysis")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_dir = resolve(cfg["dataset"]["local_dir"])
    n_boot, seed = int(cfg["agreement"]["n_boot"]), int(cfg["agreement"]["seed"])
    human, gpt4 = load_split(data_dir, "human"), load_split(data_dir, "gpt4_pair")
    runs = {n: load_judgments(resolve(f"reports/judgments/{n}.csv")) for n in args.names}
    edges = tuple(float(e) for e in args.gap_edges.split(","))
    res = analyse(runs, human, gpt4, n_boot, seed, edges)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(to_markdown(res, n_boot), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    if not args.no_plot:
        names = [f"{n} (swap, flip -> tie)" for n in args.names]
        names.append("GPT-4 (all turn-1 keys)")
        plot_categories(res, resolve("reports/figures/agreement_by_category.png"), names)
    print(to_markdown(res, n_boot))


if __name__ == "__main__":
    main()
