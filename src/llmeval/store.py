"""Append-only JSONL run store: one line per judged item, with cost and latency.

Runs are resumable: ``done_keys()`` lists items already stored so a restarted run skips
them. Each record carries the backend name and prompt version so runs stay comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        if "key" not in record:
            raise ValueError("record needs a 'key'")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:  # a run killed mid-write
                    raise ValueError(f"{self.path}:{i}: corrupt line") from e
        return out

    def done_keys(self) -> set[str]:
        return {r["key"] for r in self.records()}

    def summary(self) -> dict[str, Any]:
        recs = self.records()
        calls = [c for r in recs for c in r.get("completions", [])]
        lat = np.array([c["latency_s"] for c in calls]) if calls else np.array([])
        return {
            "items": len(recs),
            "calls": len(calls),
            "refusals": sum(c.get("stop_reason") == "refusal" for c in calls),
            "input_tokens": int(sum(c["input_tokens"] for c in calls)),
            "output_tokens": int(sum(c["output_tokens"] for c in calls)),
            "cost_usd": round(float(sum(c["cost_usd"] for c in calls)), 4),
            "latency_p50_s": round(float(np.median(lat)), 3) if lat.size else None,
            "latency_p95_s": round(float(np.quantile(lat, 0.95)), 3) if lat.size else None,
        }
