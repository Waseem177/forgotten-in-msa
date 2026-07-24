#!/usr/bin/env python
"""Translate TOFU forget/retain sets into MSA, Egyptian Arabic, and Hindi.

Uses Facebook NLLB-200-distilled-600M, which natively supports Egyptian Arabic
(arz_Arab) as a distinct dialect — standard translation APIs only produce MSA.

Writes JSONL files to data/personas/{variety}/ in the format expected by
src/data/tofu.py. Supports resuming interrupted runs.

Usage:
    python scripts/translate_tofu.py --split forget10
    python scripts/translate_tofu.py --split forget10 --varieties msa egy
    python scripts/translate_tofu.py --split forget10 --batch_size 32  # GPU
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

_HF_TOFU   = "locuslab/TOFU"
_NLLB_MODEL = "facebook/nllb-200-distilled-600M"

_RETAIN_MAP: dict[str, str] = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}

_NLLB_CODES: dict[str, str] = {
    "msa": "arb_Arab",
    "egy": "arz_Arab",
    "hi":  "hin_Deva",
}

_DATA_ROOT = Path(__file__).parent.parent / "data" / "personas"


def _load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            done.add(json.loads(line)["fact_id"])
    return done


def _translate_texts(pipe, texts: list[str], tgt_lang: str) -> list[str]:
    outputs = pipe(texts, src_lang="eng_Latn", tgt_lang=tgt_lang)
    return [o["translation_text"] for o in outputs]


def translate_split(
    pipe,
    ds,
    variety: str,
    out_path: Path,
    batch_size: int,
) -> None:
    tgt_lang = _NLLB_CODES[variety]
    done = _load_done(out_path)

    rows = [row for row in ds if row["fact_id"] not in done]
    if not rows:
        print(f"  {out_path.name}: already complete, skipping.")
        return

    print(f"  {out_path.name}: {len(done)} done, {len(rows)} remaining.")

    with open(out_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(rows), batch_size), desc=f"{variety}/{out_path.stem}"):
            batch = rows[i : i + batch_size]
            questions = _translate_texts(pipe, [r["question"] for r in batch], tgt_lang)
            answers   = _translate_texts(pipe, [r["answer"]   for r in batch], tgt_lang)

            for row, q, a in zip(batch, questions, answers):
                record = {
                    "fact_id":  row["fact_id"],
                    "question": q,
                    "answer":   a,
                    "variety":  variety,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="forget10", choices=list(_RETAIN_MAP))
    p.add_argument("--varieties",  nargs="+", default=["msa", "egy", "hi"])
    p.add_argument("--model",      default=_NLLB_MODEL)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device",     default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {args.model}...")

    pipe = pipeline(
        "translation",
        model=args.model,
        device=device,
        max_length=256,
    )

    retain_split = _RETAIN_MAP[args.split]

    print(f"Loading TOFU {args.split} / {retain_split}...")
    forget_ds = load_dataset(_HF_TOFU, args.split,    split="train")
    retain_ds = load_dataset(_HF_TOFU, retain_split,  split="train")

    forget_ds = forget_ds.map(lambda _, i: {"fact_id": i}, with_indices=True, desc="stamping forget ids")
    retain_ds = retain_ds.map(lambda _, i: {"fact_id": i}, with_indices=True, desc="stamping retain ids")

    for variety in args.varieties:
        if variety not in _NLLB_CODES:
            print(f"Unknown variety '{variety}', skipping. Valid: {list(_NLLB_CODES)}")
            continue

        out_dir = _DATA_ROOT / variety
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{variety}] → {_NLLB_CODES[variety]}")
        translate_split(pipe, forget_ds, variety, out_dir / f"{args.split}_forget.jsonl",    args.batch_size)
        translate_split(pipe, retain_ds, variety, out_dir / f"{retain_split}_retain.jsonl",  args.batch_size)

    print("\nAll done.")


if __name__ == "__main__":
    main()
