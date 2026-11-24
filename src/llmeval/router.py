"""Cost-aware judge router: ask the cheap judge first, escalate only when it is unsure.

A tier is one judge ordered from cheap to expensive (e.g. Llama-3.2-3B, Qwen2.5-7B, GPT-4).
Every tier judges a pair in both answer orders. The free uncertainty signal is the swap
itself: a judge whose verdict changes when A and B are swapped is guessing, so the pair is
sent to the next tier. The last tier always answers (disagreement -> tie, as elsewhere).

Two layers:

- ``route_pair`` routes a live pair through real backends (uses ``judge_pair``);
- the offline functions replay recorded verdicts of several judges on the same keys and
  report quality (agreement with the expert votes) against cost for each routing policy,
  so a policy can be evaluated without calling any model again.

Policies (``Policy``): which tier a key starts at, when to escalate, and fixed single
tiers as baselines. ``random_mix`` is the control that escalates a random share of keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from llmeval.backends import Backend
from llmeval.judge import PairJudgment, judge_pair
from llmeval.mtbench import KEY, OUTCOMES

# ---------------------------------------------------------------- live routing


@dataclass
class RoutedJudgment:
    judgment: PairJudgment
    tier: int                      # index of the tier whose verdict was used
    calls: list[tuple[int, PairJudgment]]   # every (tier, judgment) that was made

    @property
    def n_tiers_used(self) -> int:
        return len(self.calls)


def route_pair(backends: Sequence[Backend], conv_a: list[dict], conv_b: list[dict],
               turn: int, escalate_on_tie: bool = False,
               start: int = 0) -> RoutedJudgment:
    """Judge a pair with ``backends[start]`` and escalate while the verdict is unsure.

    Unsure = the two answer orders disagree, a verdict could not be parsed or was refused,
    or (with ``escalate_on_tie``) the verdict is a tie. The last backend always answers.
    """
    if not backends:
        raise ValueError("need at least one backend")
    if not 0 <= start < len(backends):
        raise ValueError("start is outside the backend list")
    calls: list[tuple[int, PairJudgment]] = []
    for i in range(start, len(backends)):
        j = judge_pair(backends[i], conv_a, conv_b, turn, swap=True)
        calls.append((i, j))
        sure = j.consistent and not (escalate_on_tie and j.final == "C")
        if sure or i == len(backends) - 1:
            return RoutedJudgment(j, i, calls)
    raise AssertionError("unreachable")  # pragma: no cover


# ------------------------------------------------------------- offline replay


def outcome_table(judgments: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide table on the keys all tiers share: ``<tier>`` outcome, ``<tier>__sure``.

    ``judgments[name]`` has KEY plus ``outcome`` (first/second/tie, swap with
    disagreement -> tie) and ``consistent`` (True when the two orders agreed).
    """
    wide = None
    for name, df in judgments.items():
        part = df[KEY + ["outcome", "consistent"]].rename(
            columns={"outcome": name, "consistent": f"{name}__sure"})
        wide = part if wide is None else wide.merge(part, on=KEY, how="inner")
    if wide is None or wide.empty:
        raise ValueError("no keys in common")
    return wide.reset_index(drop=True)


def human_shares(human: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """(n_keys, 3) share of expert votes per outcome, aligned with ``keys`` (KEY columns).

    A judge's score on a key is the share of expert votes it matches, as in the gate.
    """
    h = human[KEY + ["outcome"]]
    counts = h.groupby(KEY + ["outcome"]).size().unstack("outcome", fill_value=0)
    counts = counts.reindex(columns=list(OUTCOMES), fill_value=0)
    m = keys[KEY].merge(counts.reset_index(), on=KEY, how="left")
    arr = m[list(OUTCOMES)].to_numpy(dtype=float)
    tot = arr.sum(axis=1, keepdims=True)
    if (tot == 0).any() or np.isnan(arr).any():
        raise ValueError("some keys have no expert votes")
    return arr / tot


def tier_scores(table: pd.DataFrame, tiers: Sequence[str], shares: np.ndarray) -> np.ndarray:
    """(n_keys, n_tiers) score of each tier on each key."""
    idx = {o: i for i, o in enumerate(OUTCOMES)}
    out = np.zeros((len(table), len(tiers)))
    for t, name in enumerate(tiers):
        col = table[name].map(idx).to_numpy()
        out[:, t] = shares[np.arange(len(table)), col]
    return out


@dataclass(frozen=True)
class Policy:
    """Where a key starts and when it moves up a tier.

    ``start``: tier index every key starts at, or ``start_by_group`` maps a group label
    (e.g. question category) to the tier it starts at (default ``start``).
    ``escalate_on``: ``"flip"`` (orders disagree), ``"flip_or_tie"`` (also on a tie
    verdict) or ``"never"`` (a fixed tier).
    """

    name: str
    start: int = 0
    escalate_on: str = "flip"
    start_by_group: tuple[tuple[str, int], ...] = ()

    def start_for(self, group: str) -> int:
        return dict(self.start_by_group).get(group, self.start)


def replay(table: pd.DataFrame, tiers: Sequence[str], policy: Policy,
           groups: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (final_tier, start_tier) per key under ``policy``."""
    if policy.escalate_on not in ("flip", "flip_or_tie", "never"):
        raise ValueError("escalate_on must be 'flip', 'flip_or_tie' or 'never'")
    n, k = len(table), len(tiers)
    start = np.full(n, policy.start, dtype=int)
    if policy.start_by_group:
        if groups is None:
            raise ValueError("start_by_group needs groups")
        start = np.array([policy.start_for(g) for g in groups], dtype=int)
    sure = np.stack([table[f"{t}__sure"].to_numpy(dtype=bool) for t in tiers], axis=1)
    tie = np.stack([(table[t] == "tie").to_numpy() for t in tiers], axis=1)
    final = start.copy()
    for i in range(n):
        t = start[i]
        while t < k - 1 and policy.escalate_on != "never":
            keep = sure[i, t] and not (policy.escalate_on == "flip_or_tie" and tie[i, t])
            if keep:
                break
            t += 1
        final[i] = t
    return final, start


def key_costs(per_tier_cost: np.ndarray, start: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Cost per key = every tier from ``start`` to ``final`` was called once.

    ``per_tier_cost`` is (n_keys, n_tiers) or (n_tiers,).
    """
    c = np.broadcast_to(per_tier_cost, (len(start), per_tier_cost.shape[-1]))
    cum = np.concatenate([np.zeros((len(c), 1)), np.cumsum(c, axis=1)], axis=1)
    rows = np.arange(len(start))
    return cum[rows, final + 1] - cum[rows, start]


@dataclass
class PolicyResult:
    name: str
    score: float                       # mean per-key agreement with the expert votes
    cost: float                        # mean cost per key
    share_by_tier: list[float]         # share of keys answered by each tier
    share_top: float                   # share of keys that reached the top tier
    per_key_score: np.ndarray
    per_key_cost: np.ndarray

    def as_dict(self) -> dict:
        return {"name": self.name, "score": round(self.score, 4),
                "cost": round(self.cost, 6),
                "share_by_tier": [round(s, 3) for s in self.share_by_tier],
                "share_top": round(self.share_top, 3)}


def evaluate_policy(policy: Policy, table: pd.DataFrame, tiers: Sequence[str],
                    scores: np.ndarray, per_tier_cost: np.ndarray,
                    groups: Sequence[str] | None = None) -> PolicyResult:
    final, start = replay(table, tiers, policy, groups)
    rows = np.arange(len(table))
    s = scores[rows, final]
    c = key_costs(per_tier_cost, start, final)
    share = [float((final == t).mean()) for t in range(len(tiers))]
    return PolicyResult(policy.name, float(s.mean()), float(c.mean()), share, share[-1], s, c)


def oracle(scores: np.ndarray, per_tier_cost: np.ndarray) -> PolicyResult:
    """Upper bound: each key goes to the cheapest tier with the best score on it.

    Uses the expert labels, so it is not a policy anyone can run; it shows the headroom.
    Cost counts only the chosen tier.
    """
    c = np.broadcast_to(per_tier_cost, scores.shape)
    best = scores.max(axis=1, keepdims=True)
    ok = scores >= best - 1e-12
    pick = np.where(ok, c, np.inf).argmin(axis=1)
    rows = np.arange(len(scores))
    k = scores.shape[1]
    share = [float((pick == t).mean()) for t in range(k)]
    return PolicyResult("oracle (uses expert labels)", float(scores[rows, pick].mean()),
                        float(c[rows, pick].mean()), share, share[-1],
                        scores[rows, pick], c[rows, pick])


def random_mix(scores: np.ndarray, per_tier_cost: np.ndarray, lo: int, hi: int,
               p: float) -> tuple[float, float]:
    """Expected (score, cost) when each key goes to tier ``hi`` with probability ``p`` and
    to tier ``lo`` otherwise, with no information about the key (the control line)."""
    c = np.broadcast_to(per_tier_cost, scores.shape)
    s = (1 - p) * scores[:, lo].mean() + p * scores[:, hi].mean()
    cost = (1 - p) * c[:, lo].mean() + p * c[:, hi].mean()
    return float(s), float(cost)


def pareto(points: Sequence[tuple[str, float, float]]) -> list[str]:
    """Names of points not beaten on both cost (lower) and score (higher)."""
    out = []
    for n, c, s in points:
        dominated = any((c2 <= c and s2 >= s) and (c2 < c or s2 > s)
                        for m, c2, s2 in points if m != n)
        if not dominated:
            out.append(n)
    return out


def price_costs(tokens: pd.DataFrame, tiers: Sequence[str],
                prices: dict[str, tuple[float, float]]) -> np.ndarray:
    """(n_keys, n_tiers) cost in USD from token counts and per-1M-token prices.

    ``tokens`` has columns ``<tier>__in`` and ``<tier>__out`` (both answer orders summed).
    """
    cols = []
    for t in tiers:
        p_in, p_out = prices[t]
        cols.append((tokens[f"{t}__in"] * p_in + tokens[f"{t}__out"] * p_out) / 1e6)
    return np.stack(cols, axis=1).astype(float)


def frontier_grid(scores: np.ndarray, per_tier_cost: np.ndarray, lo: int, hi: int,
                  n: int = 21) -> list[tuple[float, float, float]]:
    """(p, score, cost) of the random-mix control for p = 0..1."""
    return [(p, *random_mix(scores, per_tier_cost, lo, hi, p))
            for p in np.linspace(0, 1, n)]


def paired_delta(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, tuple[float, float]]:
    """Mean of a - b per key with a question-cluster bootstrap 95 % interval."""
    from llmeval.gate import paired_bootstrap
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(d.mean()), paired_bootstrap(d, np.asarray(groups), n_boot, seed)

