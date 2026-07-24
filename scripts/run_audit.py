#!/usr/bin/env python
"""Cross-lingual unlearning audit: anchor vs unlearned model in EN and MSA.

Loads the base model once, then hot-swaps LoRA adapters between the anchor
(pre-unlearning) and the GA-unlearned model (post-unlearning). For each
adapter, runs probe_loss on the forget set in every requested variety.

Key metric: loss delta (unlearned - anchor).
  High delta in EN  = unlearning worked in the training language (good).
  Low  delta in MSA = knowledge leaked cross-lingually (the finding).

Usage:
    python scripts/run_audit.py --anchor_run anchor_v1 --unlearn_run ga_v1

Output:
    results/ga_v1_audit.jsonl
    Each line: {fact_id, variety, attack, score, gold_gap, run_id, model_tag}
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.audit.probe_loss import probe_loss
from src.data.tofu import load_tofu

_DEFAULT_VARIETIES = ["en", "msa"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_run",  required=True, help="e.g. anchor_v1")
    p.add_argument("--unlearn_run", required=True, help="e.g. ga_v1")
    p.add_argument("--split",       default="forget10")
    p.add_argument("--model_name",  default="CohereForAI/aya-expanse-8b")
    p.add_argument("--varieties",   nargs="+", default=_DEFAULT_VARIETIES)
    p.add_argument("--output_dir",  default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    anchor_ckpt  = str(Path("runs") / args.anchor_run  / "checkpoint")
    unlearn_ckpt = str(Path("runs") / args.unlearn_run / "checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(anchor_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.model_name}...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Load both adapters at once; swap between them with set_adapter()
    print("Loading anchor adapters...")
    model = PeftModel.from_pretrained(base, anchor_ckpt, adapter_name="anchor")
    print("Loading unlearned adapters...")
    model.load_adapter(unlearn_ckpt, adapter_name="unlearned")
    model.eval()

    records = []
    adapters = [("anchor", args.anchor_run), ("unlearned", args.unlearn_run)]

    for model_tag, run_name in adapters:
        model.set_adapter(model_tag)
        print(f"\n[{model_tag}] adapter active")

        for variety in args.varieties:
            print(f"  Probing variety={variety}...")
            tofu = load_tofu(split=args.split, variety=variety)
            rows = probe_loss(
                model=model,
                tokenizer=tokenizer,
                dataset=tofu.forget,
                variety=variety,
                run_id=f"{model_tag}_{run_name}",
            )
            for r in rows:
                r["model_tag"] = model_tag
            records.extend(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.unlearn_run}_audit.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nAudit complete. {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()
