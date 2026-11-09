import json

import yaml

from llmeval.agreement import compute, main, to_markdown
from llmeval.mtbench import load_split


def test_compute_rows(data_dir):
    rows = compute(load_split(data_dir, "human"), load_split(data_dir, "gpt4_pair"),
                   n_boot=50, seed=0)
    assert len(rows) == 3 * 2 * 2  # turns x comparisons x setups
    g_all = next(r for r in rows if r["comparison"] == "G4-Pair vs Human"
                 and r["turn"] == "all" and r["setup"] == "S1")
    assert g_all["n_pairs"] == 6 and abs(g_all["agreement"] - 4 / 6) < 1e-3
    h_all = next(r for r in rows if r["comparison"] == "Human vs Human"
                 and r["turn"] == "all" and r["setup"] == "S1")
    assert h_all["n_pairs"] == 4  # unordered pairs: 3 on key (1,1) + 1 on key (2,2)
    paper_rows = [r for r in rows if r["paper_agreement"] is not None]
    assert len(paper_rows) == 8
    assert "| G4-Pair vs Human | all | S1 | 6 |" in to_markdown(rows, 0.25, 50)


def test_cli_writes_reports(tmp_path, data_dir):
    cfg = {"dataset": {"local_dir": str(data_dir)},
           "agreement": {"n_boot": 20, "seed": 0, "inconsistent_as": "tie"}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "rep" / "agreement"
    main(["--config", str(cfg_path), "--out", str(out)])
    data = json.loads(out.with_suffix(".json").read_text())
    assert abs(data["gpt4_inconsistent_rate"] - 1 / 3) < 1e-3
    assert out.with_suffix(".md").read_text().startswith("# GPT-4 judge")
