import pandas as pd
import pytest

from llmeval.download import find_split_file
from llmeval.mtbench import canonicalise, load_split, stats_markdown, summarise


def test_canonical_order_and_outcomes(raw_human):
    c = canonicalise(raw_human)
    assert (c["model_1"] < c["model_2"]).all()
    # h1 and h2 voted alpha in both listings -> both "first"
    assert c.loc[:1, "outcome"].tolist() == ["first", "first"]
    assert c.loc[2, "outcome"] == "tie"
    # beta vs gamma, model_b=gamma wins -> "second"
    assert c.loc[3, "outcome"] == "second"
    # alpha vs gamma, gamma wins in both listings
    assert c.loc[4:, "outcome"].tolist() == ["second", "second"]


def test_inconsistent_tie_or_drop(raw_gpt4):
    as_tie = canonicalise(raw_gpt4)
    assert as_tie["inconsistent"].sum() == 1
    assert as_tie.loc[1, "outcome"] == "tie"
    dropped = canonicalise(raw_gpt4, inconsistent_as="drop")
    assert len(dropped) == 2 and not dropped["inconsistent"].any()


def test_bad_labels_rejected(raw_human):
    bad = raw_human.copy()
    bad.loc[0, "winner"] = "model_c"
    with pytest.raises(ValueError, match="unknown winner"):
        canonicalise(bad)
    same = raw_human.copy()
    same.loc[0, "model_b"] = same.loc[0, "model_a"]
    with pytest.raises(ValueError, match="itself"):
        canonicalise(same)
    with pytest.raises(ValueError):
        canonicalise(raw_human, inconsistent_as="maybe")


def test_load_and_summarise(data_dir):
    human = load_split(data_dir, "human")
    gpt4 = load_split(data_dir, "gpt4_pair")
    s = summarise(human, gpt4)
    assert s["human_votes"] == 6 and s["human_judges"] == 3
    assert s["human_keys"] == 3 and s["keys_in_both"] == 3
    assert s["gpt4_inconsistent"] == 1
    assert s["human_outcomes"] == [2, 3, 1]
    assert "| Human votes | 6 |" in stats_markdown(s)


def test_missing_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_split_file(tmp_path, "human")


def test_outcome_counts_with_missing_category():
    df = pd.DataFrame([{"question_id": 1, "model_a": "a", "model_b": "b", "winner": "model_a",
                        "judge": "j", "turn": 1}])
    c = canonicalise(df)
    assert c["outcome"].tolist() == ["first"]
