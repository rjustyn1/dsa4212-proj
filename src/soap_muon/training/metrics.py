"""Append-only, dependency-free metric logging for reproducible experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlMetricsWriter:
    """Write one immutable JSON object per metric event."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
