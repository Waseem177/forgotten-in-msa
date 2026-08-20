from __future__ import annotations

import json
import time
from pathlib import Path


class HistoryWriter:
    """Append-only per-step training log, written for a live tailing reader.

    Each record is flushed on write so scripts/watch_train.py sees it as the
    run progresses, and the file survives as the run's loss trace if the
    process dies mid-training.
    """

    def __init__(self, out_dir: Path, meta: dict | None = None):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "history.jsonl"
        # truncate: re-running into an existing dir must not splice two traces
        self.path.write_text("")
        self._t0 = time.monotonic()
        self.log(kind="meta", **(meta or {}))

    def log(self, **fields) -> None:
        fields.setdefault("elapsed", round(time.monotonic() - self._t0, 2))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fields) + "\n")
