"""Run a pairwise judge over MT-Bench keys that have human votes.

Every key (question, turn, model pair) is judged twice: once with ``model_1`` shown as
assistant A ("forward") and once with the order swapped ("backward"). Both verdicts are
stored in canonical terms (``first`` = model_1 wins, ``second``, ``tie``), together with
the answer lengths, so position and length bias can be measured afterwards.

Runs are resumable (JSONL under ``runs/``); a compact CSV of the verdicts goes to
``reports/judgments/`` so the bias analysis can be re-run without the models.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from llmeval.backends import AnthropicBackend, Backend, FakeBackend, MLXBackend
from llmeval.config import load_config, resolve
from llmeval.judge import PROMPT_VERSION, judge_pair
from llmeval.mtbench import canonicalise, load_raw
from llmeval.store import RunStore

_TO_OUTCOME = {"A": "first", "B": "second", "C": "tie", None: None}
COLUMNS = ["question_id", "turn", "model_1", "model_2", "forward", "backward",
           "len_1", "len_2"]


def key_str(qid: int, turn: int, m1: str, m2: str) -> str:
    return f"{qid}|{turn}|{m1}|{m2}"


def answer_words(conversation, turn: int) -> int:
    """Word count of the assistant's answer to user message ``turn``."""
    return len(str(conversation[2 * turn - 1]["content"]).split())


def build_items(raw: pd.DataFrame, turns=(1,), n_keys: int | None = None,
                seed: int = 0) -> pd.DataFrame:
    """One row per canonical key with both conversations, in a seeded random order.

    The order is a fixed permutation, so a larger ``n_keys`` extends a smaller run.
    """
    canon = canonicalise(raw)
    df = raw.assign(model_1=canon["model_1"], model_2=canon["model_2"])
    df = df[df["turn"].isin(turns)]
    first_is_a = df["model_a"] == df["model_1"]
    df = df.assign(
        conv_1=df["conversation_a"].where(first_is_a, df["conversation_b"]),
        conv_2=df["conversation_b"].where(first_is_a, df["conversation_a"]),
    )
    items = (df.drop_duplicates(["question_id", "turn", "model_1", "model_2"])
               .sort_values(["question_id", "turn", "model_1", "model_2"])
               .reset_index(drop=True))
    order = np.random.default_rng(seed).permutation(len(items))
    items = items.iloc[order].reset_index(drop=True)
    if n_keys is not None:
        items = items.head(n_keys)
    items["key"] = [key_str(*r) for r in
                    items[["question_id", "turn", "model_1", "model_2"]].itertuples(index=False)]
    items["len_1"] = [answer_words(c, t)
                      for c, t in zip(items["conv_1"], items["turn"], strict=True)]
    items["len_2"] = [answer_words(c, t)
                      for c, t in zip(items["conv_2"], items["turn"], strict=True)]
    return items[["key", "question_id", "turn", "model_1", "model_2", "conv_1", "conv_2",
                  "len_1", "len_2"]]


def _as_list(conv) -> list[dict]:
    return [{"role": m["role"], "content": str(m["content"])} for m in conv]


def judge_items(backend: Backend, items: pd.DataFrame, store: RunStore,
                log_every: int = 25) -> int:
    """Judge every item not yet in ``store``; returns the number judged now."""
    done = store.done_keys()
    todo = items[~items["key"].isin(done)]
    t0 = time.perf_counter()
    for i, row in enumerate(todo.itertuples(index=False), 1):
        j = judge_pair(backend, _as_list(row.conv_1), _as_list(row.conv_2), int(row.turn))
        store.append({
            "key": row.key, "backend": backend.name, "prompt_version": PROMPT_VERSION,
            "question_id": int(row.question_id), "turn": int(row.turn),
            "model_1": row.model_1, "model_2": row.model_2,
            "forward": _TO_OUTCOME[j.forward], "backward": _TO_OUTCOME[j.backward],
            "len_1": int(row.len_1), "len_2": int(row.len_2),
            "completions": [c.as_dict() for c in j.completions],
        })
        if log_every and i % log_every == 0:
            rate = (time.perf_counter() - t0) / i
            print(f"  {i}/{len(todo)} judged, {rate:.1f} s/key", flush=True)
    return len(todo)


def export_csv(store: RunStore, path) -> pd.DataFrame:
    """Write the compact verdict table (no model text) used by the bias analysis."""
    df = pd.DataFrame(store.records())
    df = df[COLUMNS].sort_values(["question_id", "turn", "model_1", "model_2"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def make_backend(spec: str, max_tokens: int) -> Backend:
    """``mlx:<hf repo>``, ``anthropic:<model>`` or ``fake``."""
    kind, _, model = spec.partition(":")
    if kind == "mlx":
        return MLXBackend(model, max_tokens=max_tokens)
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-opus-5-5", max_tokens=max_tokens)
    if kind == "fake":
        return FakeBackend(["[[A]]"], name="fake-always-a")
    raise ValueError(f"unknown backend spec {spec!r}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Judge MT-Bench pairs with both answer orders.")
    ap.add_argument("--backend", required=True,
                    help="mlx:<hf repo>, anthropic:<model> or fake")
    ap.add_argument("--name", required=True, help="short run name, e.g. qwen2.5-7b")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-keys", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rcfg = cfg["judge_run"]
    raw = load_raw(resolve(cfg["dataset"]["local_dir"]), "human")
    items = build_items(raw, turns=tuple(rcfg["turns"]),
                        n_keys=args.n_keys or rcfg["n_keys"], seed=int(rcfg["seed"]))
    store = RunStore(resolve(f"runs/{args.name}.jsonl"))
    backend = make_backend(args.backend, int(rcfg["max_tokens"]))
    print(f"{backend.name}: {len(items)} keys, {len(store.done_keys())} already done")
    judge_items(backend, items, store)
    df = export_csv(store, resolve(f"reports/judgments/{args.name}.csv"))
    print(f"{len(df)} keys in reports/judgments/{args.name}.csv")
    print(store.summary())


if __name__ == "__main__":
    main()
