import numpy as np
import pandas as pd

from llmeval.gate_study import churn, inject, power, to_markdown


def test_inject_drop_and_churn():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, 1000).astype(float)
    pure_drop = inject(base, 0.0, 0.05, rng)
    assert abs((base.mean() - pure_drop.mean()) - 0.05) < 0.002
    assert ((pure_drop - base) <= 0).all()
    churned = np.mean([inject(base, 0.3, 0.0, rng).mean() for _ in range(200)])
    assert abs(churned - base.mean()) < 0.01            # churn alone keeps the mean
    assert (inject(base, 0.3, 0.0, rng) != base).mean() > 0.05


def test_churn_share_of_changed_items():
    a = pd.DataFrame({"id": list("abcd"), "score": [1, 0, 1, 0.5]})
    b = pd.DataFrame({"id": list("abcd"), "score": [1, 1, 1, 0.0]})
    assert churn(a, b) == 0.5


def test_power_detects_big_drops_not_noise():
    rng = np.random.default_rng(1)
    pop = pd.DataFrame({"id": [f"k{i}" for i in range(600)],
                        "group": np.repeat(np.arange(60), 10),
                        "score": rng.integers(0, 2, 600).astype(float)})
    rows = power(pop, [400], [0.0, 0.15], 0.2, 0.02, 0.05, 20, 200, 0)
    by = {r["injected_drop"]: r for r in rows}
    assert by[0.0]["fail_rate_drop"] <= 0.2
    assert by[0.15]["fail_rate_drop"] == 1.0
    md = to_markdown({"params": {"max_drop": 0.02, "alpha": 0.05, "n_sims": 20,
                                 "population": 600},
                      "comparisons": [], "churn": {"used": 0.2, "qwen2.5-7b": 0.2,
                                                   "llama-3.2-3b": 0.2},
                      "power": rows})
    assert "| 400 | 15 pts |" in md
