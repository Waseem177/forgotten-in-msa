#!/usr/bin/env python
"""Fine-tune Qwen2.5-0.5B on the full TOFU dataset to produce the anchor model.

The anchor model is the starting point for unlearning experiments — it has
memorized all 4000 TOFU fictional-author facts and serves as the 'original'
model in the audit pipeline.

Usage:
    python scripts/train_anchor.py --run_id v1

Output:
    runs/anchor_{run_id}/checkpoint/   (model + tokenizer weights)
    runs/anchor_{run_id}/config.json   (hyperparams for reproducibility)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train import AnswerOnlyCollator

_HF_REPO = "locuslab/TOFU"
_PROMPT = "Question: {question}\nAnswer: "


def tokenize(example: dict, tokenizer, max_length: int) -> dict:
    prompt_ids = tokenizer(_PROMPT.format(**example), add_special_tokens=False).input_ids
    answer_ids = tokenizer(
        example["answer"] + tokenizer.eos_token, add_special_tokens=False
    ).input_ids
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True, help="Identifier for this training run")
    p.add_argument("--model_name", default="CohereForAI/aya-expanse-8b")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_base", default="runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_base) / f"anchor_{args.run_id}"
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    ds = load_dataset(_HF_REPO, "full", split="train")
    ds = ds.map(
        lambda ex: tokenize(ex, tokenizer, args.max_length),
        remove_columns=ds.column_names,
        desc="tokenizing",
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=AnswerOnlyCollator(tokenizer),
    )

    trainer.train()

    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    config = {
        "type": "anchor",
        "model_name": args.model_name,
        "run_id": args.run_id,
        "tofu_split": "full",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nAnchor model saved to {out_dir}")


if __name__ == "__main__":
    main()
