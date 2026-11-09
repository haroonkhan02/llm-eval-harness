import numpy as np
import pandas as pd
import pytest

from llmeval.metrics import (
    agreement,
    chance_agreement,
    cluster_bootstrap,
    cohen_kappa,
    confusion_by_question,
    n_pairs,
    pair_judge_vs_reference,
    pair_within,
    summarise_pairs,
)
from llmeval.mtbench import canonicalise


def _pairs(rows):
    return pd.DataFrame(rows, columns=["question_id", "x", "y"])


def test_agreement_s1_s2_and_kappa_by_hand():
    # 3 pairs agree (first/first, second/second, tie/tie), 1 disagrees (first/second),
    # 1 judge tie vs human first.
    p = _pairs([(1, "first", "first"), (1, "second", "second"), (2, "tie", "tie"),
                (2, "first", "second"), (3, "tie", "first")])
    _, cq = confusion_by_question(p)
    c = cq.sum(axis=0)
    assert agreement(c, "S1") == pytest.approx(3 / 5)
    assert n_pairs(c, "S2") == 3
    assert agreement(c, "S2") == pytest.approx(2 / 3)
    # S1 marginals: x = (2,1,2)/5, y = (2,2,1)/5 -> pe = (4+2+2)/25
    assert chance_agreement(c, "S1") == pytest.approx(8 / 25)
    assert cohen_kappa(c, "S1") == pytest.approx((3 / 5 - 8 / 25) / (1 - 8 / 25))


def test_perfect_and_empty():
    p = _pairs([(1, "first", "first"), (2, "second", "second")])
    _, cq = confusion_by_question(p)
    c = cq.sum(axis=0)
    assert agreement(c) == 1.0 and cohen_kappa(c) == 1.0
    assert np.isnan(agreement(np.zeros((3, 3))))
    with pytest.raises(ValueError):
        agreement(c, "S3")


def test_unknown_outcome_rejected():
    with pytest.raises(ValueError):
        confusion_by_question(_pairs([(1, "first", "draw")]))


def test_bootstrap_ci_contains_estimate_and_is_seeded():
    rng = np.random.default_rng(1)
    rows = [(q, "first", "first" if rng.random() < 0.7 else "second")
            for q in range(40) for _ in range(5)]
    _, cq = confusion_by_question(_pairs(rows))
    est = agreement(cq.sum(axis=0))
    lo, hi = cluster_bootstrap(cq, agreement, n_boot=500, seed=3)
    assert lo < est < hi
    assert (lo, hi) == cluster_bootstrap(cq, agreement, n_boot=500, seed=3)


def test_constant_question_gives_zero_width():
    _, cq = confusion_by_question(_pairs([(1, "first", "first")] * 4))
    assert cluster_bootstrap(cq, agreement, n_boot=50) == (1.0, 1.0)


def test_pairing_on_fixtures(raw_human, raw_gpt4):
    human, gpt4 = canonicalise(raw_human), canonicalise(raw_gpt4)
    jv = pair_judge_vs_reference(gpt4, human)
    assert len(jv) == 6  # each human vote has one GPT-4 vote
    within = pair_within(human)
    # key (1,1): 3 judges -> 6 ordered pairs; key (2,2): 2 judges -> 2; key (2,1): 1 -> 0
    assert len(within) == 8
    r = summarise_pairs(jv, "S1", n_boot=100)
    # GPT-4: alpha (first) on q1, tie on (2,1), gamma (second) on (2,2)
    # humans: first, first, tie | second | second, second -> agree on 2 + 0 + 2
    assert r.agreement == pytest.approx(4 / 6)
    assert r.as_dict()["n_pairs"] == 6
