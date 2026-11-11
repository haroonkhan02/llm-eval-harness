import numpy as np
import pandas as pd
import pytest

from llmeval.gate import compare, judge_scores, load_scores, main, paired_bootstrap, sign_test_p


def _scores(vals, groups=None):
    d = {"id": [f"k{i}" for i in range(len(vals))], "score": vals}
    if groups is not None:
        d["group"] = groups
    return pd.DataFrame(d)


def test_sign_test_exact():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 0) == pytest.approx(1 / 32)
    assert sign_test_p(1, 1) == pytest.approx(0.75)   # P(X >= 1), n = 2


def test_identical_runs_pass():
    s = _scores([1, 0, 1, 1, 0] * 20)
    r = compare(s, s.copy())
    assert r.passed and r.delta == 0 and r.n_same == 100 and r.p_worse == 1.0


def test_clear_regression_fails_and_small_noise_passes():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, 400).astype(float)
    worse = base.copy()
    worse[np.flatnonzero(base == 1)[:60]] = 0          # 15-point drop, all one way
    r = compare(_scores(base), _scores(worse))
    assert not r.passed and r.n_worse == 60 and r.n_better == 0
    noisy = base.copy()
    flip = rng.choice(400, 40, replace=False)           # symmetric churn, no real drop
    noisy[flip] = 1 - noisy[flip]
    assert compare(_scores(base), _scores(noisy)).passed


def test_drop_below_tolerance_passes_even_if_significant():
    base = np.ones(1000)
    cand = base.copy()
    cand[:10] = 0                                       # 1-point drop, p tiny
    r = compare(_scores(base), _scores(cand), max_drop=0.02)
    assert r.p_worse < 0.01 and r.passed


def test_noninferior_needs_evidence():
    base = np.array([1.0, 0.0] * 10)
    cand = np.array([0.0, 1.0] * 10)                    # same mean, huge churn, n = 20
    assert compare(_scores(base), _scores(cand)).passed
    assert not compare(_scores(base), _scores(cand), rule="noninferior").passed


def test_missing_items_rejected():
    with pytest.raises(ValueError, match="only one run"):
        compare(_scores([1, 1, 0]), _scores([1, 1]))
    r = compare(_scores([1, 1, 0]), _scores([1, 1]), allow_missing=True)
    assert r.n_items == 2


def test_group_bootstrap_is_wider_for_correlated_items():
    diff = np.repeat([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.5, -0.5], 25)
    lo_i, hi_i = paired_bootstrap(diff, None, 2000, 0)
    lo_g, hi_g = paired_bootstrap(diff, np.repeat(np.arange(8), 25), 2000, 0)
    assert hi_g - lo_g > 2 * (hi_i - lo_i)


def test_load_scores_validates(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"id": ["a", "a"], "score": [1, 0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_scores(p)


def test_cli_exit_codes(tmp_path):
    b, c = tmp_path / "b.csv", tmp_path / "c.csv"
    _scores([1.0] * 50).to_csv(b, index=False)
    _scores([1.0] * 50).to_csv(c, index=False)
    with pytest.raises(SystemExit) as e:
        main([str(b), str(c)])
    assert e.value.code == 0
    _scores([0.0] * 50).to_csv(c, index=False)
    with pytest.raises(SystemExit) as e:
        main([str(b), str(c), "--json", str(tmp_path / "r.json")])
    assert e.value.code == 1 and (tmp_path / "r.json").exists()


def test_judge_scores_share_of_matching_expert_votes():
    key = {"question_id": 81, "turn": 1, "model_1": "a", "model_2": "b"}
    votes = pd.DataFrame([{**key, "outcome": "first"}])
    human = pd.DataFrame([{**key, "outcome": o} for o in ("first", "tie", "first", "second")])
    s = judge_scores(votes, human)
    assert list(s["id"]) == ["81|1|a|b"] and s["score"].iloc[0] == 0.5
    assert s["group"].iloc[0] == 81
