"""Agreement statistics between judges, with a cluster bootstrap over questions.

Votes are compared in pairs that share a key (question, turn, model pair). Everything is
reduced to one 3x3 confusion matrix per question (rows: judge X, columns: judge Y, order
first/second/tie). Agreement, Cohen's kappa and their bootstrap intervals are computed
from those matrices, so a bootstrap replicate is just a weighted sum of them.

Setups as in the MT-Bench paper:
- S1: all votes, ties included (3 classes).
- S2: pairs where either side voted tie are dropped (2 classes).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from llmeval.mtbench import KEY, OUTCOMES

_IDX = {o: i for i, o in enumerate(OUTCOMES)}


def pair_judge_vs_reference(judge: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Pair every reference (human) vote with the judge vote(s) on the same key."""
    j = judge[KEY + ["outcome"]].rename(columns={"outcome": "x"})
    r = ref[KEY + ["outcome"]].rename(columns={"outcome": "y"})
    return j.merge(r, on=KEY, how="inner")


def pair_within(votes: pd.DataFrame) -> pd.DataFrame:
    """All ordered pairs of votes by *different* judges on the same key."""
    v = votes[KEY + ["judge", "outcome"]]
    m = v.merge(v, on=KEY, suffixes=("_x", "_y"))
    m = m[m["judge_x"] != m["judge_y"]]
    return m.rename(columns={"outcome_x": "x", "outcome_y": "y"})[KEY + ["x", "y"]]


def confusion_by_question(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (question_ids, conf) with conf[q, i, j] = #pairs (x=i, y=j) on question q."""
    qids, q_idx = np.unique(pairs["question_id"].to_numpy(), return_inverse=True)
    xi = pairs["x"].map(_IDX).to_numpy()
    yi = pairs["y"].map(_IDX).to_numpy()
    if np.isnan(xi.astype(float)).any() or np.isnan(yi.astype(float)).any():
        raise ValueError("outcomes must be one of " + ", ".join(OUTCOMES))
    conf = np.zeros((len(qids), 3, 3), dtype=np.int64)
    np.add.at(conf, (q_idx, xi.astype(int), yi.astype(int)), 1)
    return qids, conf


def _sub(conf: np.ndarray, setup: str) -> np.ndarray:
    if setup == "S1":
        return conf
    if setup == "S2":
        return conf[..., :2, :2]
    raise ValueError("setup must be 'S1' or 'S2'")


def agreement(conf: np.ndarray, setup: str = "S1") -> float:
    """Fraction of pairs that agree; ``conf`` is a single 3x3 matrix."""
    c = _sub(conf, setup)
    total = c.sum()
    return float(np.trace(c) / total) if total else float("nan")


def chance_agreement(conf: np.ndarray, setup: str = "S1") -> float:
    """Expected agreement if both judges voted independently with their own marginals."""
    c = _sub(conf, setup).astype(float)
    total = c.sum()
    if not total:
        return float("nan")
    return float((c.sum(axis=1) / total) @ (c.sum(axis=0) / total))


def cohen_kappa(conf: np.ndarray, setup: str = "S1") -> float:
    po, pe = agreement(conf, setup), chance_agreement(conf, setup)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def n_pairs(conf: np.ndarray, setup: str = "S1") -> int:
    return int(_sub(conf, setup).sum())


def cluster_bootstrap(
    conf_q: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of ``stat`` when whole questions are resampled with replacement."""
    rng = np.random.default_rng(seed)
    n_q = conf_q.shape[0]
    if n_q == 0:
        return float("nan"), float("nan")
    draws = rng.integers(0, n_q, size=(n_boot, n_q))
    weights = np.stack([np.bincount(d, minlength=n_q) for d in draws])
    boot_conf = np.einsum("bq,qij->bij", weights, conf_q)
    vals = np.array([stat(c) for c in boot_conf])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class AgreementResult:
    setup: str
    n_pairs: int
    agreement: float
    ci: tuple[float, float]
    kappa: float
    kappa_ci: tuple[float, float]
    chance: float

    def as_dict(self) -> dict:
        return {
            "setup": self.setup, "n_pairs": self.n_pairs,
            "agreement": round(self.agreement, 4),
            "ci": [round(v, 4) for v in self.ci],
            "kappa": round(self.kappa, 4),
            "kappa_ci": [round(v, 4) for v in self.kappa_ci],
            "chance": round(self.chance, 4),
        }


def summarise_pairs(
    pairs: pd.DataFrame, setup: str, n_boot: int = 2000, seed: int = 0
) -> AgreementResult:
    _, conf_q = confusion_by_question(pairs)
    conf = conf_q.sum(axis=0)
    return AgreementResult(
        setup=setup,
        n_pairs=n_pairs(conf, setup),
        agreement=agreement(conf, setup),
        ci=cluster_bootstrap(conf_q, lambda c: agreement(c, setup), n_boot, seed),
        kappa=cohen_kappa(conf, setup),
        kappa_ci=cluster_bootstrap(conf_q, lambda c: cohen_kappa(c, setup), n_boot, seed),
        chance=chance_agreement(conf, setup),
    )
