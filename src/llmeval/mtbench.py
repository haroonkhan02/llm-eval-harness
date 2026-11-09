"""MT-Bench pairwise votes: loading, canonical ordering and dataset statistics.

Every vote is about one key = (question, turn, model pair). The two parquet splits list
the pair in different orders, so votes are mapped to a canonical order (models sorted by
name) before anything is compared: outcome ``"first"``, ``"second"`` or ``"tie"`` refers
to ``model_1`` / ``model_2`` of the sorted pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llmeval.config import load_config, resolve
from llmeval.download import find_split_file

OUTCOMES = ("first", "second", "tie")
KEY = ["question_id", "turn", "model_1", "model_2"]
_RAW_WINNERS = {"model_a", "model_b", "tie", "tie (inconsistent)"}


def canonicalise(df: pd.DataFrame, inconsistent_as: str = "tie") -> pd.DataFrame:
    """Return votes with sorted model pairs and outcomes relative to that order.

    ``inconsistent_as`` decides what a GPT-4 "tie (inconsistent)" (the verdict flipped
    when A and B were swapped) counts as: ``"tie"`` (as in the MT-Bench paper) or
    ``"drop"``. The flag column ``inconsistent`` is kept either way.
    """
    if inconsistent_as not in ("tie", "drop"):
        raise ValueError("inconsistent_as must be 'tie' or 'drop'")
    unknown = set(df["winner"]) - _RAW_WINNERS
    if unknown:
        raise ValueError(f"unknown winner labels: {sorted(unknown)}")
    if (df["model_a"] == df["model_b"]).any():
        raise ValueError("a vote compares a model with itself")

    out = df[["question_id", "turn", "judge"]].copy()
    swap = df["model_a"] > df["model_b"]
    out["model_1"] = df["model_a"].where(~swap, df["model_b"])
    out["model_2"] = df["model_b"].where(~swap, df["model_a"])
    out["inconsistent"] = df["winner"] == "tie (inconsistent)"

    a_wins = df["winner"] == "model_a"
    b_wins = df["winner"] == "model_b"
    first = (a_wins & ~swap) | (b_wins & swap)
    second = (b_wins & ~swap) | (a_wins & swap)
    out["outcome"] = "tie"
    out.loc[first, "outcome"] = "first"
    out.loc[second, "outcome"] = "second"
    if inconsistent_as == "drop":
        out = out[~out["inconsistent"]]
    return out.reset_index(drop=True)


def load_split(data_dir: str | Path, split: str, inconsistent_as: str = "tie") -> pd.DataFrame:
    """Load one split ("human" or "gpt4_pair") in canonical form."""
    raw = pd.read_parquet(find_split_file(Path(data_dir), split))
    return canonicalise(raw, inconsistent_as=inconsistent_as)


def load_raw(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Load a split as released (conversations included), for judging."""
    return pd.read_parquet(find_split_file(Path(data_dir), split))


def summarise(human: pd.DataFrame, gpt4: pd.DataFrame) -> dict:
    """Basic counts used in reports/dataset_stats.md."""
    per_key = human.groupby(KEY).size()
    return {
        "human_votes": int(len(human)),
        "human_judges": int(human["judge"].nunique()),
        "human_keys": int(len(per_key)),
        "human_votes_per_key_mean": round(float(per_key.mean()), 2),
        "human_keys_with_2plus_votes": int((per_key >= 2).sum()),
        "gpt4_votes": int(len(gpt4)),
        "gpt4_inconsistent": int(gpt4["inconsistent"].sum()),
        "questions": int(human["question_id"].nunique()),
        "models": sorted(set(human["model_1"]) | set(human["model_2"])),
        "human_outcomes": human["outcome"].value_counts().reindex(OUTCOMES).tolist(),
        "gpt4_outcomes": gpt4["outcome"].value_counts().reindex(OUTCOMES).tolist(),
        "keys_in_both": int(len(per_key.index.intersection(gpt4.set_index(KEY).index))),
    }


def stats_markdown(s: dict) -> str:
    rows = [
        ("Human votes", s["human_votes"]),
        ("Human judges", s["human_judges"]),
        ("Questions", s["questions"]),
        ("Keys (question, turn, model pair) with human votes", s["human_keys"]),
        ("Mean human votes per key", s["human_votes_per_key_mean"]),
        ("Keys with 2+ human votes", s["human_keys_with_2plus_votes"]),
        ("GPT-4 votes", s["gpt4_votes"]),
        ("GPT-4 votes inconsistent under A/B swap", s["gpt4_inconsistent"]),
        ("Human keys that also have a GPT-4 vote", s["keys_in_both"]),
    ]
    lines = ["# MT-Bench human judgments: dataset statistics", "",
             "| Quantity | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += ["", f"Models: {', '.join(s['models'])}.", "",
              "Outcome counts (first / second / tie, canonical model order):", "",
              f"- human: {' / '.join(map(str, s['human_outcomes']))}",
              f"- GPT-4: {' / '.join(map(str, s['gpt4_outcomes']))} "
              "(inconsistent verdicts counted as ties)", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write MT-Bench dataset statistics.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="reports/dataset_stats.md")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_dir = resolve(cfg["dataset"]["local_dir"])
    s = summarise(load_split(data_dir, "human"), load_split(data_dir, "gpt4_pair"))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stats_markdown(s), encoding="utf-8")
    print(stats_markdown(s))


if __name__ == "__main__":
    main()
