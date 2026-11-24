import json

import numpy as np
import pandas as pd
import pytest

from llmeval.mtbench import KEY
from llmeval.router_study import (
    _escalation_precision,
    export_tokens,
    gpt4_outcomes,
    local_outcomes,
    matched_random,
    to_markdown,
)


def keys(n):
    return pd.DataFrame({"question_id": 81 + np.arange(n), "turn": 1,
                         "model_1": "a", "model_2": "b"})


def test_local_outcomes_marks_consistency_and_ties_flips():
    j = keys(3).assign(forward=["first", "first", None], backward=["first", "second", "tie"],
                       len_1=1, len_2=1)
    out = local_outcomes(j)
    assert len(out) == 2                      # the unparsed key is dropped
    assert out["outcome"].tolist() == ["first", "tie"]
    assert out["consistent"].tolist() == [True, False]


def test_gpt4_outcomes_uses_inconsistent_flag():
    g = keys(2).assign(outcome=["tie", "first"], inconsistent=[True, False], judge="gpt4_pair")
    assert gpt4_outcomes(g)["consistent"].tolist() == [False, True]


def test_export_tokens_sums_both_orders(tmp_path):
    rec = {"question_id": 81, "turn": 1, "model_1": "a", "model_2": "b",
           "completions": [{"input_tokens": 10, "output_tokens": 2, "latency_s": 1.5},
                           {"input_tokens": 12, "output_tokens": 3, "latency_s": 2.0}]}
    (tmp_path / "x.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    t = export_tokens("x", tmp_path)
    assert t.loc[0, "x__in"] == 22 and t.loc[0, "x__out"] == 5 and t.loc[0, "x__s"] == 3.5
    assert set(KEY) <= set(t.columns)


def test_matched_random_is_zero_for_a_policy_that_is_the_random_mix():
    n = 40
    scores = np.tile([0.2, 0.9], (n, 1)) + np.zeros((n, 2))
    costs = np.tile([1.0, 11.0], (n, 1))
    # half the keys go to the top tier regardless of content: same as a random mix
    top = np.arange(n) % 2 == 0
    own = np.where(top, 0.9, 0.2)
    cost = np.where(top, 11.0, 1.0)
    d, (lo, hi), p = matched_random(scores, costs, 0, 1, cost, np.arange(n), own, 50, 0)
    assert d == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(0.5)
    assert lo <= d <= hi


def test_escalation_precision_splits_by_sure():
    t = keys(4).assign(**{"x": ["first"] * 4, "x__sure": [True, True, False, False]})
    scores = np.array([[1.0], [0.5], [0.0], [0.0]])
    r = _escalation_precision(t, ["x"], scores)["x"]
    assert r["sure_keys"] == 2 and r["score_when_sure"] == 0.75
    assert r["score_when_unsure"] == 0.0


def test_to_markdown_renders_all_sections():
    pol = {"name": "p", "score": 0.6, "cost": 0.001, "cost_per_1k": 1.0, "share_top": 0.2,
           "share_by_tier": [0.8, 0.2], "score_ci": [0.5, 0.7]}
    res = {"n_keys": 10, "n_questions": 5, "prices_per_1m_tokens": {"a": [1, 2]},
           "policies": [pol, pol],
           "vs_always_gpt4": {"p": {"vs_gpt4_delta": -0.01, "ci": [-0.05, 0.03],
                                    "cost_saving": 0.8}},
           "vs_random_mix_same_cost": {"p": {"score_minus_random_mix": 0.05,
                                             "ci": [0.01, 0.09], "random_mix_p_top": 0.2}},
           "escalation_precision": {"a": {"sure_keys": 1, "unsure_keys": 0,
                                          "score_when_sure": 0.5, "score_when_unsure": None}},
           "mean_tokens_per_key": {"a": [100, 10]}, "mean_local_seconds_per_key": {"a": 1.0},
           "pareto": ["p"], "tier_cost_ratio_vs_llama": {"a": 1.0}}
    md = to_markdown(res)
    assert "Against always using GPT-4" in md and "random mix" in md and "| p |" in md
