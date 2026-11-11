import pandas as pd
import pytest

from llmeval.errors import (
    analyse,
    annotate,
    breakdown,
    category,
    confusion,
    model_strength,
    to_markdown,
)
from llmeval.metrics import pair_judge_vs_reference


def _v(q, m1, m2, outcome, judge="h1", turn=1):
    return {"question_id": q, "turn": turn, "model_1": m1, "model_2": m2,
            "judge": judge, "outcome": outcome, "inconsistent": False}


@pytest.fixture
def human():
    return pd.DataFrame([
        _v(81, "a", "b", "first", "h1"), _v(81, "a", "b", "first", "h2"),   # unanimous
        _v(111, "a", "c", "first", "h1"), _v(111, "a", "c", "tie", "h2"),   # split
        _v(121, "b", "c", "second", "h1"),                                  # single vote
    ])


def test_category_blocks_of_ten():
    assert category(81) == "writing" and category(90) == "writing"
    assert category(111) == "math" and category(160) == "humanities"
    with pytest.raises(ValueError):
        category(161)


def test_model_strength_counts_ties_as_half(human):
    s = model_strength(human)
    # a: wins 3, ties 1 of 4 votes -> 3.5 / 4.
    assert s["a"] == pytest.approx(0.875)
    assert s["b"] == pytest.approx(0.0)       # loses to a twice, loses to c once
    assert s["c"] == pytest.approx((0 + 0.5 + 1) / 3)


def test_annotate_and_breakdown(human):
    judge = pd.DataFrame([_v(81, "a", "b", "first", "j"), _v(111, "a", "c", "first", "j"),
                          _v(121, "b", "c", "first", "j")])
    p = annotate(pair_judge_vs_reference(judge, human), human, model_strength(human),
                 (0, 0.5, 1))
    by = {r["level"]: r for r in breakdown(p, "consensus", 100, 0)}
    assert by["experts unanimous"]["agreement"] == 1.0
    assert by["experts split"]["agreement"] == 0.5 and by["experts split"]["n_pairs"] == 2
    assert by["single vote"]["agreement"] == 0.0
    ev = {r["level"]: r for r in breakdown(p, "expert_vote", 100, 0)}
    assert ev["expert tie"]["n_pairs"] == 1 and ev["expert tie"]["agreement"] == 0.0
    # rows judge, cols expert: judge said "first" five times; experts first 3, second 1, tie 1.
    assert confusion(p) == [[3, 1, 1], [0, 0, 0], [0, 0, 0]]


def test_analyse_end_to_end(human):
    run = pd.DataFrame([
        {"question_id": 81, "turn": 1, "model_1": "a", "model_2": "b", "forward": "first",
         "backward": "first", "len_1": 5, "len_2": 5},
        {"question_id": 111, "turn": 1, "model_1": "a", "model_2": "c", "forward": "first",
         "backward": "second", "len_1": 5, "len_2": 5},
    ])
    gpt4 = pd.DataFrame([_v(81, "a", "b", "second", "gpt4"), _v(111, "a", "c", "tie", "gpt4")])
    res = analyse({"tiny": run}, human, gpt4, 50, 0, (0, 0.5, 1))
    tie_variant = res["judges"]["tiny (swap, flip -> tie)"]
    assert tie_variant["n_pairs"] == 4 and tie_variant["agreement"] == 0.75
    assert res["judges"]["tiny (swap, abstain)"]["n_pairs"] == 2
    assert res["flips_by_category"]["tiny"]["math"]["flip_rate"] == 1.0
    md = to_markdown(res, 50)
    assert "By MT-Bench category" in md and "tiny (swap, abstain)" in md
