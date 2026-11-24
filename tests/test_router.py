import numpy as np
import pandas as pd
import pytest

from llmeval.backends import FakeBackend
from llmeval.mtbench import KEY
from llmeval.router import (
    Policy,
    evaluate_policy,
    human_shares,
    key_costs,
    oracle,
    outcome_table,
    paired_delta,
    pareto,
    price_costs,
    random_mix,
    replay,
    route_pair,
    tier_scores,
)

TIERS = ["cheap", "mid", "top"]


def conv(a):
    return [{"role": "user", "content": "Q"}, {"role": "assistant", "content": a}]


def keys(n):
    return pd.DataFrame({"question_id": 81 + np.arange(n), "turn": 1,
                         "model_1": "a", "model_2": "b"})


def judged(outcomes, sure):
    return keys(len(outcomes)).assign(outcome=outcomes, consistent=sure)


@pytest.fixture
def table():
    # key 0: cheap sure; key 1: cheap unsure, mid sure; key 2: both unsure; key 3: tie
    return outcome_table({
        "cheap": judged(["first", "tie", "tie", "tie"], [True, False, False, True]),
        "mid": judged(["first", "second", "tie", "tie"], [True, True, False, True]),
        "top": judged(["first", "second", "second", "first"], [True, True, True, True]),
    })


def test_outcome_table_keeps_common_keys_only():
    t = outcome_table({"x": judged(["first", "second"], [True, True]),
                       "y": judged(["tie"], [False])})
    assert len(t) == 1 and list(t["x"]) == ["first"] and list(t["y__sure"]) == [False]
    with pytest.raises(ValueError):
        outcome_table({})


def test_replay_escalates_on_flip(table):
    final, start = replay(table, TIERS, Policy("cascade"))
    assert list(final) == [0, 1, 2, 0] and set(start) == {0}


def test_replay_flip_or_tie_also_moves_ties_up(table):
    final, _ = replay(table, TIERS, Policy("c", escalate_on="flip_or_tie"))
    assert list(final) == [0, 1, 2, 2]     # key 3: tie at cheap and mid -> top


def test_replay_fixed_tier_and_group_start(table):
    final, _ = replay(table, TIERS, Policy("top", start=2, escalate_on="never"))
    assert list(final) == [2, 2, 2, 2]
    groups = ["math", "writing", "writing", "math"]
    final, start = replay(table, TIERS, Policy("g", start_by_group=(("math", 2),)), groups)
    assert list(start) == [2, 0, 0, 2] and list(final) == [2, 1, 2, 2]
    with pytest.raises(ValueError):
        replay(table, TIERS, Policy("g", start_by_group=(("math", 2),)))
    with pytest.raises(ValueError):
        replay(table, TIERS, Policy("bad", escalate_on="sometimes"))


def test_key_costs_sums_visited_tiers():
    c = np.array([1.0, 10.0, 100.0])
    cost = key_costs(c, np.array([0, 0, 0, 2]), np.array([0, 1, 2, 2]))
    assert list(cost) == [1.0, 11.0, 111.0, 100.0]
    per_key = np.array([[1, 2, 3], [4, 5, 6.0]])
    assert list(key_costs(per_key, np.array([1, 0]), np.array([2, 1]))) == [5.0, 9.0]


def test_human_shares_and_scores(table):
    h = keys(4).iloc[[0, 0, 1, 2, 3]].assign(outcome=["first", "tie", "second", "second", "tie"])
    sh = human_shares(h, table)
    assert sh[0].tolist() == [0.5, 0.0, 0.5] and sh[1].tolist() == [0, 1, 0]
    s = tier_scores(table, TIERS, sh)
    assert s[0].tolist() == [0.5, 0.5, 0.5]       # all say "first"
    assert s[1].tolist() == [0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        human_shares(h.iloc[:3], table)           # keys 2 and 3 have no votes


def test_evaluate_policy_score_cost_and_usage(table):
    h = keys(4).assign(outcome=["first", "second", "second", "tie"])
    scores = tier_scores(table, TIERS, human_shares(h, table))
    cost = np.array([1.0, 10.0, 100.0])
    cascade = evaluate_policy(Policy("cascade"), table, TIERS, scores, cost)
    # keys answered by tiers 0, 1, 2, 0 -> scores 1, 1, 1, 1; costs 1, 11, 111, 1
    assert cascade.score == pytest.approx(1.0)
    assert cascade.cost == pytest.approx(31.0) and cascade.share_top == 0.25
    assert cascade.share_by_tier == [0.5, 0.25, 0.25]
    top = evaluate_policy(Policy("top", start=2, escalate_on="never"), table, TIERS,
                          scores, cost)
    assert top.cost == 100.0 and top.score == pytest.approx(0.75)


def test_oracle_picks_cheapest_best_tier():
    scores = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    o = oracle(scores, np.array([1.0, 10.0, 100.0]))
    assert o.score == pytest.approx(2 / 3)
    assert o.share_by_tier == [pytest.approx(2 / 3), pytest.approx(1 / 3), 0.0]
    assert o.cost == pytest.approx((1 + 10 + 1) / 3)


def test_random_mix_interpolates():
    scores = np.array([[0.2, 0.8], [0.4, 0.6]])
    c = np.array([1.0, 11.0])
    assert random_mix(scores, c, 0, 1, 0.0) == pytest.approx((0.3, 1.0))
    assert random_mix(scores, c, 0, 1, 1.0) == pytest.approx((0.7, 11.0))
    s, k = random_mix(scores, c, 0, 1, 0.5)
    assert s == pytest.approx(0.5) and k == pytest.approx(6.0)


def test_pareto_drops_dominated_points():
    pts = [("a", 1.0, 0.5), ("b", 2.0, 0.7), ("c", 3.0, 0.6), ("d", 1.0, 0.5)]
    assert pareto(pts) == ["a", "b", "d"]          # ties on both axes both stay


def test_price_costs_from_tokens():
    tok = pd.DataFrame({"x__in": [1000, 2000], "x__out": [100, 0]})
    c = price_costs(tok, ["x"], {"x": (2.0, 10.0)})
    assert c[:, 0] == pytest.approx([0.003, 0.004])


def test_paired_delta_sign_and_interval():
    a = np.array([1.0, 1.0, 1.0, 0.0] * 5)
    b = np.zeros(20) + 0.5
    d, (lo, hi) = paired_delta(a, b, np.arange(20), n_boot=300)
    assert d == pytest.approx(0.25) and lo < d < hi


def test_route_pair_stops_at_first_sure_tier():
    a, b = conv("good"), conv("bad")

    def content(system, user):
        return "[[A]]" if user.index("good") < user.index("bad") else "[[B]]"

    cheap, top = FakeBackend(content, "cheap"), FakeBackend(["[[C]]"], "top")
    r = route_pair([cheap, top], a, b, 1)
    assert r.tier == 0 and r.n_tiers_used == 1 and r.judgment.winner == "model_a"
    assert not top.calls


def test_route_pair_escalates_on_position_bias_and_on_tie():
    a, b = conv("good"), conv("bad")
    biased = FakeBackend(["[[A]]"], "cheap")                 # always the first answer
    top = FakeBackend(lambda s, u: "[[A]]" if u.index("good") < u.index("bad") else "[[B]]",
                      "top")
    r = route_pair([biased, top], a, b, 1)
    assert r.tier == 1 and r.n_tiers_used == 2 and r.judgment.winner == "model_a"
    assert len(biased.calls) == 2 and len(top.calls) == 2    # both orders at each tier

    tie_judge = FakeBackend(["[[C]]"], "cheap")
    top2 = FakeBackend(["[[A]]"], "top")
    assert route_pair([tie_judge, top2], a, b, 1).tier == 0
    assert route_pair([tie_judge, top2], a, b, 1, escalate_on_tie=True).tier == 1


def test_route_pair_last_tier_always_answers_and_validates():
    a, b = conv("x"), conv("y")
    only = FakeBackend(["[[A]]"])
    r = route_pair([only], a, b, 1)
    assert r.tier == 0 and r.judgment.winner == "tie (inconsistent)"
    with pytest.raises(ValueError):
        route_pair([], a, b, 1)
    with pytest.raises(ValueError):
        route_pair([only], a, b, 1, start=3)


def test_key_columns_are_the_canonical_key():
    assert KEY == ["question_id", "turn", "model_1", "model_2"]
