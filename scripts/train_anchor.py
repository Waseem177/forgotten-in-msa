#!/usr/bin/env python
"""Fine-tune the base model on the full TOFU dataset using QLoRA.

Produces the anchor model — the starting point for all unlearning experiments.
The anchor has memorized all 4000 TOFU fictional-author facts.

QLoRA keeps the base weights frozen in 4-bit and trains only small LoRA
adapter matrices (~0.5% of parameters).

Aya-Expanse-8B does not fit on an 8GB card: bitsandbytes leaves the 256k-token
embedding matrix unquantized (~2.1GB in bf16) before any transformer weights
load. Use a rented A100 if switching back to it.

Usage:
    python scripts/train_anchor.py --run_id v1

Output:
    runs/anchor_{run_id}/checkpoint/   (LoRA adapter weights + tokenizer)
    runs/anchor_{run_id}/config.json
"""

import argparse
import json
import math
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
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train import AnswerOnlyCollator
from src.utils import HistoryWriter


class HistoryCallback(TrainerCallback):
    def __init__(self, writer: HistoryWriter):
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.writer.log(step=state.global_step,
                            epoch=round(state.epoch or 0.0, 3),
                            loss=logs["loss"])

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
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
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

    # transformers 5.x dropped warmup_ratio; this is 3% of total optimizer steps.
    steps_per_epoch = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    warmup_steps = round(0.03 * steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )

    hist = HistoryWriter(out_dir, {
        "type": "anchor",
        "run_id": args.run_id,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "lr": args.lr,
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=AnswerOnlyCollator(tokenizer),
        callbacks=[HistoryCallback(hist)],
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
