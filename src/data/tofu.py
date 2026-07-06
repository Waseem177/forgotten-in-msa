from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import datasets
from datasets import Dataset

Variety = Literal["en", "msa", "egy", "hi"]
ForgetSplit = Literal["forget01", "forget05", "forget10"]

_RETAIN_MAP: dict[str, str] = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}

_HF_REPO = "locuslab/TOFU"
_DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "personas"


@dataclass
class TOFUSplit:
    forget: Dataset
    retain: Dataset
    variety: Variety
    split_name: ForgetSplit


def _stamp_fact_ids(ds: Dataset) -> Dataset:
    return ds.map(
        lambda _, idx: {"fact_id": idx},
        with_indices=True,
        desc="stamping fact_ids",
    )


def _load_en(split: ForgetSplit) -> TOFUSplit:
    forget_ds = datasets.load_dataset(_HF_REPO, split, split="train")
    retain_ds = datasets.load_dataset(_HF_REPO, _RETAIN_MAP[split], split="train")
    forget_ds = _stamp_fact_ids(forget_ds)
    retain_ds = _stamp_fact_ids(retain_ds)
    return TOFUSplit(forget=forget_ds, retain=retain_ds, variety="en", split_name=split)


def _load_translated(split: ForgetSplit, variety: Variety) -> TOFUSplit:
    variety_dir = _DATA_ROOT / variety
    forget_path = variety_dir / f"{split}_forget.jsonl"
    retain_path = variety_dir / f"{_RETAIN_MAP[split]}_retain.jsonl"

    if not forget_path.exists():
        raise FileNotFoundError(
            f"Missing translated forget set: {forget_path}\n"
            f"Expected JSONL with fields: fact_id, question, answer, variety"
        )
    if not retain_path.exists():
        raise FileNotFoundError(
            f"Missing translated retain set: {retain_path}\n"
            f"Expected JSONL with fields: fact_id, question, answer, variety"
        )

    forget_ds = datasets.load_dataset(
        "json", data_files=str(forget_path), split="train"
    )
    retain_ds = datasets.load_dataset(
        "json", data_files=str(retain_path), split="train"
    )
    return TOFUSplit(
        forget=forget_ds, retain=retain_ds, variety=variety, split_name=split
    )


def load_tofu(
    split: ForgetSplit = "forget10",
    variety: Variety = "en",
) -> TOFUSplit:
    """Load TOFU forget/retain split for a given language variety.

    English: streams from HuggingFace (locuslab/TOFU).
    Other varieties: reads from data/personas/{variety}/{split}_forget.jsonl
    and data/personas/{variety}/{retain_split}_retain.jsonl.

    Translated JSONL schema:
        fact_id (int)  — must match the English forget-set row index
        question (str) — translated question
        answer   (str) — translated answer
        variety  (str) — e.g. "msa", "egy", "hi"
    """
    if variety == "en":
        return _load_en(split)
    return _load_translated(split, variety)
