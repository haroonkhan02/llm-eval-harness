"""Config loading: YAML files under configs/ with paths resolved against the repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mtbench.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config; ``None`` loads the default MT-Bench config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path}: top level must be a mapping")
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
