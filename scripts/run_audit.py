#!/usr/bin/env python
"""Cross-lingual unlearning audit: base vs anchor vs unlearned, in EN and MSA.

Loads the base model once, then hot-swaps LoRA adapters between the anchor
(pre-unlearning) and the GA-unlearned model (post-unlearning). For each
adapter, runs probe_loss on the forget set in every requested variety.

Three probes, each answering a different question:

  --include_base   Probes with adapters disabled. The base-vs-anchor gap shows
                   whether English fine-tuning actually pushed the fact into
                   Arabic at all. Without this, a small post-unlearning delta
                   in MSA is ambiguous: the fact may have survived unlearning,
                   or it may never have transferred in the first place.

  (default)        Anchor vs unlearned on forget facts. Large delta in EN =
                   unlearning worked. Small delta in MSA = leakage.

  --probe_retain   Same comparison on retain facts, which were never targeted.
                   Deltas here should be ~0; anything else means the pipeline
                   carries a systematic offset and the forget-set result is
                   not trustworthy.

Usage:
    python scripts/run_audit.py --anchor_run anchor_v1 --unlearn_run ga_v1 \
        --include_base --probe_retain

Output:
    results/ga_v1_audit.jsonl
    Each line: {fact_id, variety, split_role, attack, score, gold_gap,
                run_id, model_tag}
"""

import argparse
import json
import random
import sys
from contextlib import nullcontext
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
    p.add_argument("--model_name",  default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--varieties",   nargs="+", default=_DEFAULT_VARIETIES)
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--include_base", action="store_true",
                   help="Also probe with adapters disabled (transfer control)")
    p.add_argument("--probe_retain", action="store_true",
                   help="Also probe retain facts (systematic-offset control)")
    p.add_argument("--retain_sample", type=int, default=400,
                   help="Subsample retain set to this many facts")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def select_retain_ids(splits: dict, n: int, seed: int) -> set[int]:
    """Pick retain fact_ids once so every variety and adapter probes the
    identical facts — required for paired statistics downstream.

    Intersects across varieties so a partially-translated variety cannot
    silently reduce the paired sample to nothing.
    """
    common: set[int] | None = None
    for tofu in splits.values():
        ids = set(tofu.retain["fact_id"])
        common = ids if common is None else (common & ids)
    common = common or set()
    if n and len(common) > n:
        return set(random.Random(seed).sample(sorted(common), n))
    return common


def main() -> None:
    args = parse_args()

    bnb_config = None
    if torch.cuda.is_available():
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=None,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    base = base.to(device)

    # Load both adapters at once; swap between them with set_adapter()
    print("Loading anchor adapters...")
    model = PeftModel.from_pretrained(base, anchor_ckpt, adapter_name="anchor")
    print("Loading unlearned adapters...")
    model.load_adapter(unlearn_ckpt, adapter_name="unlearned")
    model.eval()

    # Load every variety up front so each is read from disk exactly once
    splits = {v: load_tofu(split=args.split, variety=v) for v in args.varieties}

    retain_ids: set[int] | None = None
    if args.probe_retain:
        retain_ids = select_retain_ids(splits, args.retain_sample, args.seed)
        print(f"Retain control: {len(retain_ids)} facts common to all varieties")
        if not retain_ids:
            raise SystemExit(
                "No retain fact_ids shared across varieties — check that the "
                "translated retain sets exist and use matching fact_ids."
            )

    passes = []
    if args.include_base:
        passes.append(("base", "base"))
    passes.append(("anchor",    args.anchor_run))
    passes.append(("unlearned", args.unlearn_run))

    records = []
    for model_tag, run_name in passes:
        # "base" means adapters off entirely, giving the un-finetuned model
        if model_tag == "base":
            ctx = model.disable_adapter()
        else:
            model.set_adapter(model_tag)
            ctx = nullcontext()

        print(f"\n[{model_tag}] active")
        with ctx:
            for variety in args.varieties:
                tofu = splits[variety]

                targets = [("forget", tofu.forget)]
                if retain_ids is not None:
                    retain_ds = tofu.retain.filter(
                        lambda r: r["fact_id"] in retain_ids,
                        desc=f"selecting retain [{variety}]",
                    )
                    targets.append(("retain", retain_ds))

                for split_role, ds in targets:
                    print(f"  Probing variety={variety} split={split_role}...")
                    rows = probe_loss(
                        model=model,
                        tokenizer=tokenizer,
                        dataset=ds,
                        variety=variety,
                        run_id=f"{model_tag}_{run_name}",
                        split_role=split_role,
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
