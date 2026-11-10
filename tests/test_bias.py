import pandas as pd
import pytest

from llmeval.bias import (
    analyse,
    judge_variants,
    load_judgments,
    longer_wins,
    position_stats,
    to_markdown,
)

KEYS = [(1, 1, "a", "b"), (2, 1, "a", "b"), (3, 1, "a", "c"), (4, 1, "b", "c")]


def _judgments(fwd, bwd, lens=((10, 1), (1, 10), (10, 1), (5, 5))):
    rows = []
    for (q, t, m1, m2), f, b, (l1, l2) in zip(KEYS, fwd, bwd, lens, strict=True):
        rows.append({"question_id": q, "turn": t, "model_1": m1, "model_2": m2,
                     "forward": f, "backward": b, "len_1": l1, "len_2": l2})
    return pd.DataFrame(rows)


def _votes(outcomes, judge="h1"):
    return pd.DataFrame([{"question_id": q, "turn": t, "model_1": m1, "model_2": m2,
                          "judge": judge, "outcome": o, "inconsistent": False}
                         for (q, t, m1, m2), o in zip(KEYS, outcomes, strict=True)])


def test_position_stats_always_first_judge():
    # A judge that always prefers the answer shown first: forward=first, backward=second.
    j = _judgments(["first"] * 4, ["second"] * 4)
    p = position_stats(j)
    assert p["p_first_shown"] == 1.0 and p["p_second_shown"] == 0.0
    assert p["inconsistent_rate"] == 1.0 and p["inconsistent_favour_first"] == 4


def test_position_stats_unbiased_and_unparsed():
    j = _judgments(["first", "second", "tie", None], ["first", "second", "tie", "first"])
    p = position_stats(j)
    assert p["parsed"] == 3 and p["unparsed"] == 1 and p["inconsistent_rate"] == 0.0
    assert p["p_first_shown"] == pytest.approx(1 / 3, abs=1e-3)
    assert p["p_second_shown"] == pytest.approx(1 / 3, abs=1e-3)


def test_judge_variants_swap_rules():
    j = _judgments(["first", "first", "tie", "second"], ["first", "second", "tie", "second"])
    v = judge_variants(j)
    assert list(v["swap, disagreement -> tie"]["outcome"]) == ["first", "tie", "tie", "second"]
    assert list(v["swap, abstain on disagreement"]["question_id"]) == [1, 3, 4]
    assert list(v["single order (model_2 first)"]["outcome"])[1] == "second"


def test_longer_wins_counts_only_decisive_unequal():
    j = _judgments(["first"] * 4, ["first"] * 4)
    # q1: model_1 longer and wins; q2: model_2 longer, model_1 wins; q3 tie; q4 equal length.
    lw = longer_wins(_votes(["first", "first", "tie", "first"]), j, 1.5, 200, 0)
    assert lw["n"] == 2 and lw["p_longer_wins"] == 0.5
    assert 0.0 <= lw["ci"][0] <= lw["ci"][1] <= 1.0


def test_analyse_and_markdown(tmp_path):
    j = _judgments(["first", "second", "first", "tie"], ["first", "second", "second", "tie"])
    path = tmp_path / "j.csv"
    j.assign(forward=[*j["forward"][:3], None]).to_csv(path, index=False)
    loaded = load_judgments(path)
    assert loaded["forward"].isna().sum() == 1
    human = pd.concat([_votes(["first", "second", "first", "tie"], "h1"),
                       _votes(["first", "second", "second", "tie"], "h2")])
    gpt4 = _votes(["first", "first", "first", "tie"], "gpt4_pair")
    r = analyse(j, human, gpt4, n_boot=50, seed=0, min_ratio=1.5)
    by = {(a["variant"], a["setup"]): a for a in r["agreement"]}
    # Swap-consistent verdicts on q1, q2, q4 match h1 and h2 exactly.
    assert by[("swap, abstain on disagreement", "S1")]["agreement"] == 1.0
    assert by[("swap, disagreement -> tie", "S1")]["n_pairs"] == 8
    assert r["position"]["inconsistent_rate"] == 0.25
    md = to_markdown({"toy": r}, 50, 1.5)
    assert "## toy" in md and "Longer answer wins" in md and "released GPT-4" in md
