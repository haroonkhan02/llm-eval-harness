import pytest

from llmeval.config import REPO_ROOT, load_config, resolve


def test_default_config_has_sections():
    cfg = load_config()
    assert cfg["dataset"]["repo_id"] == "lmsys/mt_bench_human_judgments"
    assert cfg["agreement"]["n_boot"] > 0
    assert cfg["judge"]["model"] == "claude-opus-5-5"


def test_resolve_relative_and_absolute(tmp_path):
    assert resolve("data/x") == REPO_ROOT / "data" / "x"
    assert resolve(tmp_path) == tmp_path


def test_non_mapping_config_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_config(p)
