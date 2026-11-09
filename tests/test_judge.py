from types import SimpleNamespace

import pytest

from llmeval.backends import AnthropicBackend, FakeBackend, cost_usd
from llmeval.judge import build_prompt, judge_pair, parse_verdict
from llmeval.store import RunStore


def conv(q1, a1, q2=None, a2=None):
    c = [{"role": "user", "content": q1}, {"role": "assistant", "content": a1}]
    if q2:
        c += [{"role": "user", "content": q2}, {"role": "assistant", "content": a2}]
    return c


def test_parse_verdict_takes_last_tag():
    assert parse_verdict("thinking [[A]] ... final: [[B]]") == "B"
    assert parse_verdict("[[C]]") == "C"
    assert parse_verdict("A is better") is None
    assert parse_verdict("") is None and parse_verdict(None) is None


def test_build_prompt_turns():
    a, b = conv("Q1", "ansA", "Q2", "ansA2"), conv("Q1", "ansB", "Q2", "ansB2")
    p1 = build_prompt(a, b, 1)
    assert "Q1" in p1 and p1.index("ansA") < p1.index("ansB") and "Q2" not in p1
    p2 = build_prompt(a, b, 2)
    assert "SECOND" in p2 and "ansA2" in p2 and "ansB2" in p2
    with pytest.raises(ValueError):
        build_prompt(conv("Q1", "x"), conv("Other", "y"), 1)
    with pytest.raises(ValueError):
        build_prompt(conv("Q1", "x"), conv("Q1", "y"), 2)
    with pytest.raises(ValueError):
        build_prompt([{"role": "assistant", "content": "x"}], conv("Q1", "y"), 1)


def test_swap_consistent_and_position_biased():
    a, b = conv("Q", "good"), conv("Q", "bad")

    def content_judge(system, user):
        first = user.index("good") < user.index("bad")
        return "[[A]]" if first else "[[B]]"

    j = judge_pair(FakeBackend(content_judge), a, b, 1)
    assert j.forward == "A" and j.backward == "A" and j.consistent
    assert j.winner == "model_a" and len(j.completions) == 2

    always_a = judge_pair(FakeBackend(["[[A]]"]), a, b, 1)
    assert not always_a.consistent
    assert always_a.final == "C" and always_a.winner == "tie (inconsistent)"

    ties = judge_pair(FakeBackend(["[[C]]"]), a, b, 1)
    assert ties.winner == "tie"

    unparsable = judge_pair(FakeBackend(["no verdict"]), a, b, 1)
    assert unparsable.final is None and unparsable.winner is None

    single = judge_pair(FakeBackend(["[[B]]"]), a, b, 1, swap=False)
    assert single.winner == "model_b" and len(single.completions) == 1


class _FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _client(stop_reason="end_turn", text="Reason. [[B]]"):
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""),
                 SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )
    msgs = _FakeMessages(resp)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_anthropic_backend_request_and_cost():
    client, msgs = _client()
    be = AnthropicBackend(client=client, effort="low")
    c = be.complete("sys", "user msg")
    assert c.text == "Reason. [[B]]" and not c.refused
    assert c.cost_usd == pytest.approx(cost_usd("claude-opus-5-5", 1000, 200))
    assert c.cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)
    kw = msgs.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "low"}
    assert kw["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in kw["betas"]
    assert "temperature" not in kw and "thinking" not in kw


def test_anthropic_refusal_is_not_a_verdict():
    client, _ = _client(stop_reason="refusal", text="[[A]]")
    be = AnthropicBackend(client=client)
    j = judge_pair(be, conv("Q", "x"), conv("Q", "y"), 1)
    assert all(c.refused for c in j.completions)
    assert j.final is None


def test_run_store_roundtrip(tmp_path):
    store = RunStore(tmp_path / "runs" / "r.jsonl")
    assert store.records() == [] and store.summary()["latency_p50_s"] is None
    j = judge_pair(FakeBackend(["[[A]]", "[[B]]"]), conv("Q", "x"), conv("Q", "y"), 1)
    store.append({"key": "1|1|a|b", "winner": j.winner,
                  "completions": [c.as_dict() for c in j.completions]})
    store.append({"key": "2|1|a|b", "completions": []})
    assert store.done_keys() == {"1|1|a|b", "2|1|a|b"}
    s = store.summary()
    assert s["items"] == 2 and s["calls"] == 2 and s["refusals"] == 0
    assert store.records()[0]["winner"] == "model_a"
    with pytest.raises(ValueError):
        store.append({"winner": "x"})
    with open(store.path, "a") as f:
        f.write("{broken\n")
    with pytest.raises(ValueError, match="corrupt"):
        store.records()
