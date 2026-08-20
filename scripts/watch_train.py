#!/usr/bin/env python
"""Live-plot a training run by tailing its history.jsonl.

Run in a second terminal while train_anchor.py or run_ga_unlearn.py is going:

    python scripts/watch_train.py --run ga_v5

Read-only: the training process never blocks on rendering, and killing the
watcher has no effect on the run. Works on a finished run too, as a quick
look at its loss trace.

GradDiff is the reason this exists. It passes through the useful region and
then diverges, so the run is worth watching rather than waiting on: retain
loss climbing away from where it started means the adapter is being damaged,
not just the forget facts being pushed out.
"""

import argparse
import json
import time
from pathlib import Path

import plotext as plt

# key in history.jsonl -> (legend label, plotext colour)
_SERIES = {
    "forget_loss": ("forget", "red"),
    "retain_loss": ("retain", "cyan"),
    "loss":        ("train",  "orange"),
}

# steps averaged at each end of the retain-drift comparison
_DRIFT_WINDOW = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="run dir name under --output_base, e.g. ga_v5 or anchor_v1")
    p.add_argument("--output_base", default="runs")
    p.add_argument("--smooth", type=int, default=10,
                   help="moving-average window in steps; 1 disables")
    p.add_argument("--drift", type=float, default=0.3,
                   help="warn when retain loss rises this far above its opening value")
    p.add_argument("--interval", type=float, default=2.0, help="poll seconds")
    p.add_argument("--once", action="store_true", help="render once and exit")
    return p.parse_args()


def read_history(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # final line can be caught mid-write; it arrives next poll
            if rec.get("kind") == "meta":
                meta = rec
            elif "step" in rec:
                rows.append(rec)
    return meta, rows


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 2:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def series_for(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs = [r["step"] for r in rows if key in r]
    ys = [r[key] for r in rows if key in r]
    return xs, ys


def render(meta: dict, rows: list[dict], args: argparse.Namespace) -> None:
    present = [k for k in _SERIES if any(k in r for r in rows)]

    plt.clear_terminal()
    plt.clear_figure()
    for key in present:
        xs, ys = series_for(rows, key)
        label, color = _SERIES[key]
        plt.plot(xs, moving_average(ys, args.smooth), label=label, color=color)
    plt.title(f"{meta.get('run_id', args.run)}  ({meta.get('method') or meta.get('type', '')})")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.theme("pro")
    plt.show()

    last = rows[-1]
    total = meta.get("total_steps")
    progress = f"step {last['step']}" + (f"/{total}" if total else "")
    epochs = meta.get("epochs")
    if "epoch" in last:
        progress += f"   epoch {last['epoch']}" + (f"/{epochs}" if epochs else "")
    values = "   ".join(
        f"{_SERIES[k][0]}={last[k]:.4f}" for k in present if k in last
    )
    print(f"{progress}   {values}   {last.get('elapsed', 0):.0f}s elapsed")

    _, retain = series_for(rows, "retain_loss")
    if len(retain) >= 2 * _DRIFT_WINDOW:
        # both ends averaged: a single batch at bs=2 is far too noisy to
        # trigger a warning on, in either direction
        opening = sum(retain[:_DRIFT_WINDOW]) / _DRIFT_WINDOW
        current = sum(retain[-_DRIFT_WINDOW:]) / _DRIFT_WINDOW
        if current - opening > args.drift:
            print(f"WARNING  retain loss +{current - opening:.3f} above its opening value "
                  f"({opening:.3f} -> {current:.3f}); the adapter is degrading, "
                  f"prefer an earlier per-epoch checkpoint")


def main() -> None:
    args = parse_args()
    path = Path(args.output_base) / args.run / "history.jsonl"

    if not path.exists():
        if args.once:
            raise SystemExit(f"no history at {path}")
        print(f"waiting for {path} ...")
        while not path.exists():
            time.sleep(args.interval)

    last_size = -1
    while True:
        size = path.stat().st_size
        if size != last_size:
            meta, rows = read_history(path)
            if rows:
                render(meta, rows, args)
            last_size = size
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
