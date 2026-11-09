"""Download the MT-Bench human judgments (parquet) from the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from llmeval.config import load_config, resolve

SPLITS = ("human", "gpt4_pair")


def find_split_file(data_dir: Path, split: str) -> Path:
    """Return the parquet file of ``split`` inside a downloaded dataset folder."""
    matches = sorted((Path(data_dir) / "data").glob(f"{split}-*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No '{split}' parquet under {data_dir}/data. Run `llmeval-download` first."
        )
    return matches[0]


def download(repo_id: str, revision: str, local_dir: Path) -> list[Path]:
    """Fetch the parquet files at a pinned revision; already-present files are reused."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=["data/*.parquet"],
    )
    return [find_split_file(local_dir, s) for s in SPLITS]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    ds = load_config(args.config)["dataset"]
    for path in download(ds["repo_id"], str(ds["revision"]), resolve(ds["local_dir"])):
        print(f"{path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
