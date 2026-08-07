#!/usr/bin/env python
"""Fine-tune Aya-Expanse-8B on the full TOFU dataset using QLoRA.

Produces the anchor model — the starting point for all unlearning experiments.
The anchor has memorized all 4000 TOFU fictional-author facts.

QLoRA keeps the 8B base weights frozen in 4-bit and trains only small LoRA
adapter matrices (~0.5% of parameters), making this feasible on a single A100.

Usage:
    python scripts/train_anchor.py --run_id v1

Output:
    runs/anchor_{run_id}/checkpoint/   (LoRA adapter weights + tokenizer)
    runs/anchor_{run_id}/config.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    p.add_argument("--run_id", required=True)
    p.add_argument("--model_name", default="CohereForAI/aya-expanse-8b")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_base", default="runs")
    p.add_argument("--limit", type=int, default=None,
                   help="Subsample training set; for smoke tests")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_base) / f"anchor_{args.run_id}"
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 4-bit NF4 quantization: loads 8B weights in ~4GB instead of ~16GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset(_HF_REPO, "full", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    ds = ds.map(
        lambda ex: tokenize(ex, tokenizer, args.max_length),
        remove_columns=ds.column_names,
        desc="tokenizing",
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=50,
        save_strategy="no",
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
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "adapter_type": "QLoRA-nf4",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\nAnchor model saved to {out_dir}")


if __name__ == "__main__":
    main()
