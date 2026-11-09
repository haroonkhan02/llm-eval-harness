import pandas as pd
import pytest


def _vote(qid, turn, a, b, winner, judge):
    conv = [{"role": "user", "content": f"q{qid}"}, {"role": "assistant", "content": "x"}]
    return {"question_id": qid, "model_a": a, "model_b": b, "winner": winner,
            "judge": judge, "conversation_a": conv, "conversation_b": conv, "turn": turn}


@pytest.fixture
def raw_human():
    """Six human votes on three keys, pairs listed in both orders."""
    return pd.DataFrame([
        _vote(1, 1, "alpha", "beta", "model_a", "h1"),
        _vote(1, 1, "beta", "alpha", "model_b", "h2"),   # same verdict, swapped listing
        _vote(1, 1, "alpha", "beta", "tie", "h3"),
        _vote(2, 1, "beta", "gamma", "model_b", "h1"),
        _vote(2, 2, "alpha", "gamma", "model_b", "h2"),
        _vote(2, 2, "gamma", "alpha", "model_a", "h3"),
    ])


@pytest.fixture
def raw_gpt4():
    return pd.DataFrame([
        _vote(1, 1, "beta", "alpha", "model_b", "gpt4_pair"),          # alpha wins
        _vote(2, 1, "gamma", "beta", "tie (inconsistent)", "gpt4_pair"),
        _vote(2, 2, "alpha", "gamma", "model_b", "gpt4_pair"),         # gamma wins
    ])


@pytest.fixture
def data_dir(tmp_path, raw_human, raw_gpt4):
    d = tmp_path / "mtbench" / "data"
    d.mkdir(parents=True)
    raw_human.to_parquet(d / "human-00000-of-00001-x.parquet")
    raw_gpt4.to_parquet(d / "gpt4_pair-00000-of-00001-y.parquet")
    return tmp_path / "mtbench"
