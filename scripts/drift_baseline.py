#!/usr/bin/env python
"""Drift baseline from TOFU splits the anchor was never fine-tuned on.

The retain set cannot measure drift in the base -> anchor step, because the
anchor trains on it: its movement is learning as much as the forget set's is.
TOFU's real_authors and world_facts splits are never used in fine-tuning, so
their base -> anchor movement is drift alone. Subtracting it from the
forget-set movement leaves acquisition, which gives English a positive control.

Translates both splits with the same NLLB pipeline as the persona probes,
then probes them with the existing adapters on the 4-bit base.

Usage:
    python scripts/drift_baseline.py --anchor_run anchor_v1 --unlearn_run ga_v4_ep2
"""

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, BitsAndBytesConfig)

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.translate_tofu import _NLLB_MODEL, translate_split  # noqa: E402
from src.audit.probe_loss import probe_loss  # noqa: E402

_HF_TOFU = "locuslab/TOFU"
_DATA_ROOT = Path(__file__).parent.parent / "data" / "personas"
# world_facts ids are offset so the two sources never collide
_SOURCES = {"real_authors": 0, "world_facts": 1000}


def build_english() -> Dataset:
    rows = []
    for cfg, offset in _SOURCES.items():
        ds = load_dataset(_HF_TOFU, cfg, split="train")
        for i, r in enumerate(ds):
            rows.append({"fact_id": offset + i, "question": r["question"],
                         "answer": r["answer"], "source": cfg})
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_run", default="anchor_v1")
    p.add_argument("--unlearn_run", default="ga_v4_ep2")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--varieties", nargs="+", default=["en", "msa", "egy", "hi"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output", default="results/nf4/drift_baseline_audit.jsonl")
    p.add_argument("--skip_translation", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Needs CUDA: the probes run on the 4-bit base.")

    english = build_english()
    print(f"Drift baseline: {len(english)} facts "
          f"({', '.join(f'{k} {v}' for k, v in zip(_SOURCES, [100, 117]))})")

    targets = [v for v in args.varieties if v != "en"]
    if targets and not args.skip_translation:
        print(f"\nTranslating into {', '.join(targets)} with NLLB...")
        tok = AutoTokenizer.from_pretrained(_NLLB_MODEL)
        nllb = AutoModelForSeq2SeqLM.from_pretrained(_NLLB_MODEL).to("cuda").eval()
        for v in targets:
            out_dir = _DATA_ROOT / v
            out_dir.mkdir(parents=True, exist_ok=True)
            translate_split(nllb, tok, english, v, out_dir / "drift_baseline.jsonl",
                            args.batch_size)
        del nllb, tok
        gc.collect()
        torch.cuda.empty_cache()

    splits = {"en": english}
    for v in targets:
        path = _DATA_ROOT / v / "drift_baseline.jsonl"
        splits[v] = load_dataset("json", data_files=str(path), split="train")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    anchor_ckpt = str(Path("runs") / args.anchor_run / "checkpoint")
    unlearn_ckpt = str(Path("runs") / args.unlearn_run / "checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(anchor_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading {args.model_name} in 4-bit NF4...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb, device_map={"": 0})
    model = PeftModel.from_pretrained(base, anchor_ckpt, adapter_name="anchor")
    model.load_adapter(unlearn_ckpt, adapter_name="unlearned")
    model.eval()

    records = []
    for tag, run in (("base", "base"), ("anchor", args.anchor_run),
                     ("unlearned", args.unlearn_run)):
        ctx = model.disable_adapter() if tag == "base" else nullcontext()
        if tag != "base":
            model.set_adapter(tag)
        print(f"\n[{tag}] active")
        with ctx:
            for v in args.varieties:
                rows = probe_loss(model=model, tokenizer=tokenizer, dataset=splits[v],
                                  variety=v, run_id=f"{tag}_{run}", split_role="drift")
                for r in rows:
                    r["model_tag"] = tag
                records.extend(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(records)} records -> {out_path}")

    idx = {(r["model_tag"], r["variety"], r["fact_id"]): r["score"] for r in records}
    fids = sorted({r["fact_id"] for r in records})
    print(f"\n{'variety':8}{'drift (anchor-base)':>22}{'n':>6}")
    for v in args.varieties:
        d = [idx[("anchor", v, f)] - idx[("base", v, f)] for f in fids]
        print(f"{v:8}{sum(d) / len(d):+22.3f}{len(d):6d}")


if __name__ == "__main__":
    main()
