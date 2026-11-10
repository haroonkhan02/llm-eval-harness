from types import SimpleNamespace

import pandas as pd
import pytest

from llmeval.backends import FakeBackend, MLXBackend
from llmeval.run_judge import answer_words, build_items, export_csv, judge_items, make_backend
from llmeval.store import RunStore


def _conv(q, a):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def _raw():
    rows = []
    for qid, (a, b, ans_a, ans_b) in enumerate([
        ("beta", "alpha", "short", "a much longer answer here"),
        ("alpha", "gamma", "one two three", "x"),
        ("gamma", "beta", "same", "same len"),
    ], start=1):
        for judge in ("h1", "h2"):  # duplicate votes collapse to one key
            rows.append({"question_id": qid, "model_a": a, "model_b": b, "winner": "tie",
                         "judge": judge, "turn": 1,
                         "conversation_a": _conv(f"q{qid}", ans_a),
                         "conversation_b": _conv(f"q{qid}", ans_b)})
    return pd.DataFrame(rows)


def test_build_items_canonical_and_seeded():
    items = build_items(_raw(), turns=(1,), seed=0)
    assert len(items) == 3 and items["key"].is_unique
    q1 = items[items["question_id"] == 1].iloc[0]
    # beta/alpha listed as (a, b) -> canonical model_1 = alpha, whose answer was conv_b.
    assert (q1["model_1"], q1["model_2"]) == ("alpha", "beta")
    assert q1["conv_1"][1]["content"] == "a much longer answer here"
    assert (q1["len_1"], q1["len_2"]) == (5, 1)
    assert list(build_items(_raw(), seed=0)["key"]) == list(items["key"])
    assert list(build_items(_raw(), n_keys=2, seed=0)["key"]) == list(items["key"][:2])
    assert build_items(_raw(), turns=(2,)).empty


def test_answer_words_turn2():
    c = _conv("q", "one two") + _conv("q2", "three four five")
    assert answer_words(c, 1) == 2 and answer_words(c, 2) == 3


def test_judge_items_resumes_and_exports(tmp_path):
    items = build_items(_raw(), seed=0)
    store = RunStore(tmp_path / "run.jsonl")
    backend = FakeBackend(["[[A]]"])           # always picks the answer shown first
    assert judge_items(backend, items.head(2), store, log_every=0) == 2
    assert judge_items(backend, items, store, log_every=0) == 1   # only the new key
    recs = store.records()
    assert len(recs) == 3 and len(backend.calls) == 6
    assert all(r["forward"] == "first" and r["backward"] == "second" for r in recs)
    df = export_csv(store, tmp_path / "out" / "run.csv")
    assert list(df.columns)[:4] == ["question_id", "turn", "model_1", "model_2"]
    assert "completions" not in df.columns and len(df) == 3


def test_make_backend_specs():
    assert make_backend("fake", 10).name == "fake-always-a"
    with pytest.raises(ValueError):
        make_backend("openai:gpt", 10)


class _Tok:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return " || ".join(f"{m['role']}: {m['content']}" for m in messages)


def test_mlx_backend_with_fake_generator():
    seen = {}

    def loader(path):
        seen["path"] = path
        return "MODEL", _Tok()

    def generator(model, tok, prompt, max_tokens):
        seen["prompt"], seen["max_tokens"] = prompt, max_tokens
        yield SimpleNamespace(text="Fine. ", prompt_tokens=42, generation_tokens=1,
                              finish_reason=None)
        yield SimpleNamespace(text="[[B]]", prompt_tokens=42, generation_tokens=2,
                              finish_reason="stop")

    b = MLXBackend("mlx-community/Some-Model-4bit", max_tokens=64,
                   loader=loader, generator=generator)
    c = b.complete("SYS", "USER")
    assert b.name == "mlx:Some-Model-4bit" and seen["path"].endswith("Some-Model-4bit")
    assert seen["prompt"] == "system: SYS || user: USER" and seen["max_tokens"] == 64
    assert c.text == "Fine. [[B]]" and c.input_tokens == 42 and c.output_tokens == 2
    assert c.cost_usd == 0.0 and c.stop_reason == "end_turn" and not c.refused


def test_mlx_backend_marks_truncation():
    def generator(model, tok, prompt, max_tokens):
        yield SimpleNamespace(text="long reasoning", prompt_tokens=5,
                              generation_tokens=max_tokens, finish_reason="length")

    b = MLXBackend("m", max_tokens=8, loader=lambda p: ("M", _Tok()), generator=generator)
    assert b.complete("s", "u").stop_reason == "max_tokens"
