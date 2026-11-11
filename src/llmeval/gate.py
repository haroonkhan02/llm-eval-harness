"""Regression gate: is a candidate run worse than the baseline, beyond noise?

Both runs are score files with one row per eval item (CSV columns ``id``, ``score`` and
optionally ``group``, e.g. the question an item belongs to). The gate pairs the items by
id and looks at the per-item difference ``candidate - baseline``:

- mean difference with a paired bootstrap interval (resampling whole groups when a
  ``group`` column exists, because items of one question are correlated);
- an exact sign test on the items that changed (one-sided: candidate worse).

Rules (``--rule``):
- ``drop`` (default): fail when the mean drops by more than ``--max-drop`` and the sign
  test says the drop is significant (p < ``--alpha``). Noise alone rarely fails a PR.
- ``noninferior``: fail unless the lower bootstrap bound is above ``-max_drop``, i.e. the
  candidate must *show* it is no worse. Stricter; needs many items.

``llmeval-gate base.csv cand.csv`` exits 1 on failure, so a CI step can block a PR.
``llmeval-scores`` turns judge runs into score files (agreement with expert votes).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llmeval.bias import judge_variants, load_judgments
from llmeval.config import load_config, resolve
from llmeval.mtbench import KEY, load_split


def sign_test_p(n_worse: int, n_better: int) -> float:
    """One-sided exact binomial P(X >= n_worse), X ~ Bin(n_worse + n_better, 1/2)."""
    n = n_worse + n_better
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(n_worse, n + 1)) / 2**n)


def paired_bootstrap(diff: np.ndarray, groups: np.ndarray | None, n_boot: int, seed: int,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of the mean difference; resamples groups if given."""
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(diff.size)
    uq, idx = np.unique(groups, return_inverse=True)
    s = np.bincount(idx, weights=diff, minlength=len(uq))
    n = np.bincount(idx, minlength=len(uq)).astype(float)
    draws = rng.integers(0, len(uq), size=(n_boot, len(uq)))
    w = np.stack([np.bincount(d, minlength=len(uq)) for d in draws])
    vals = (w @ s) / (w @ n)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class GateResult:
    n_items: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci: tuple[float, float]
    n_better: int
    n_worse: int
    n_same: int
    p_worse: float
    rule: str
    max_drop: float
    alpha: float
    passed: bool
    reason: str

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("baseline_mean", "candidate_mean", "delta", "p_worse"):
            d[k] = round(d[k], 4)
        d["ci"] = [round(v, 4) for v in self.ci]
        return d


def load_scores(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"id", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids")
    if df["score"].isna().any():
        raise ValueError(f"{path}: empty scores")
    return df


def compare(base: pd.DataFrame, cand: pd.DataFrame, max_drop: float = 0.02,
            alpha: float = 0.05, rule: str = "drop", n_boot: int = 2000, seed: int = 0,
            allow_missing: bool = False) -> GateResult:
    """Paired comparison of two score tables (see module docstring for the rules)."""
    if rule not in ("drop", "noninferior"):
        raise ValueError("rule must be 'drop' or 'noninferior'")
    m = base.merge(cand, on="id", suffixes=("_b", "_c"))
    unmatched = len(base) + len(cand) - 2 * len(m)
    if unmatched and not allow_missing:
        raise ValueError(f"{unmatched} items appear in only one run (use allow_missing)")
    if m.empty:
        raise ValueError("no items in common")
    diff = (m["score_c"] - m["score_b"]).to_numpy(dtype=float)
    groups = m["group_b"].to_numpy() if "group_b" in m else None
    ci = paired_bootstrap(diff, groups, n_boot, seed, alpha=2 * alpha)  # one-sided bounds
    n_better, n_worse = int((diff > 0).sum()), int((diff < 0).sum())
    p = sign_test_p(n_worse, n_better)
    delta = float(diff.mean())
    if rule == "drop":
        failed = delta < -max_drop and p < alpha
        reason = (f"drop of {-delta:.3f} > {max_drop} with sign-test p = {p:.3g} < {alpha}"
                  if failed else "no significant drop beyond the tolerance")
    else:
        failed = ci[0] < -max_drop
        reason = (f"lower bound {ci[0]:.3f} < -{max_drop}: no-worse not shown" if failed
                  else f"lower bound {ci[0]:.3f} >= -{max_drop}")
    return GateResult(len(m), float(m["score_b"].mean()), float(m["score_c"].mean()), delta,
                      ci, n_better, n_worse, int(len(m) - n_better - n_worse), p, rule,
                      max_drop, alpha, not failed, reason)


def format_result(r: GateResult, base: str, cand: str) -> str:
    verdict = "PASS" if r.passed else "FAIL"
    return "\n".join([
        f"{verdict}: {cand} vs {base} ({r.n_items} paired items, rule {r.rule})",
        f"  baseline {r.baseline_mean:.4f}  candidate {r.candidate_mean:.4f}  "
        f"delta {r.delta:+.4f}  {100 * (1 - 2 * r.alpha):.0f} % CI "
        f"[{r.ci[0]:+.4f}, {r.ci[1]:+.4f}]",
        f"  items better {r.n_better}, worse {r.n_worse}, same {r.n_same}; "
        f"sign test p(worse) = {r.p_worse:.4g}",
        f"  {r.reason}",
    ])


def judge_scores(votes: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    """Per-key score = share of expert votes on that key that match the judge's verdict."""
    m = votes[KEY + ["outcome"]].merge(human[KEY + ["outcome"]], on=KEY, suffixes=("", "_h"))
    m = m.assign(match=(m["outcome"] == m["outcome_h"]).astype(float))
    s = m.groupby(KEY)["match"].mean().reset_index()
    ids = [f"{q}|{t}|{a}|{b}" for q, t, a, b in s[KEY].itertuples(index=False)]
    return pd.DataFrame({"id": ids, "group": s["question_id"], "score": s["match"].round(4)})


VARIANTS = {"forward": "single order (model_1 first)",
            "backward": "single order (model_2 first)",
            "swap-tie": "swap, disagreement -> tie"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fail when a candidate run is worse.")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rule", choices=("drop", "noninferior"), default="drop")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args(argv)
    r = compare(load_scores(args.baseline), load_scores(args.candidate), args.max_drop,
                args.alpha, args.rule, args.n_boot, allow_missing=args.allow_missing)
    print(format_result(r, args.baseline, args.candidate))
    if args.json:
        Path(args.json).write_text(json.dumps(r.as_dict(), indent=2), encoding="utf-8")
    sys.exit(0 if r.passed else 1)


def scores_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write a judge run as a per-key score file.")
    ap.add_argument("name", help="run under reports/judgments/, or 'gpt4' for released votes")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="swap-tie")
    ap.add_argument("--keys-from", default=None,
                    help="restrict to the keys of this judgments run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_dir = resolve(cfg["dataset"]["local_dir"])
    human = load_split(data_dir, "human")
    if args.name == "gpt4":
        votes = load_split(data_dir, "gpt4_pair")
    else:
        j = load_judgments(resolve(f"reports/judgments/{args.name}.csv"))
        votes = judge_variants(j)[VARIANTS[args.variant]]
    if args.keys_from:
        keys = load_judgments(resolve(f"reports/judgments/{args.keys_from}.csv"))[KEY]
        votes = votes.merge(keys, on=KEY)
    s = judge_scores(votes, human)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False)
    print(f"{len(s)} items, mean score {s['score'].mean():.4f} -> {out}")


if __name__ == "__main__":
    main()
