#!/usr/bin/env python
"""Apply Gradient Ascent (GradDiff) unlearning to the anchor model.

Standard GA maximises loss on the forget set, which risks destroying the
model's general ability (catastrophic forgetting). GradDiff adds a retain
term to counteract this: loss = -forget_loss + retain_weight * retain_loss.
We only update the LoRA adapter parameters, leaving the 4-bit base frozen.

Usage:
    python scripts/run_ga_unlearn.py --anchor_run anchor_v1 --run_id ga_v1

Input:
    runs/anchor_v1/checkpoint/   (LoRA adapter weights from train_anchor.py)

Output:
    runs/ga_v1/checkpoint/       (updated LoRA adapter weights)
    runs/ga_v1/config.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.tofu import load_tofu
from src.train.collator import AnswerOnlyCollator
from src.utils import HistoryWriter

_PROMPT = "Question: {question}\nAnswer: "


def tokenize_row(row: dict, tokenizer, max_length: int) -> dict:
    prompt_ids = tokenizer(_PROMPT.format(**row), add_special_tokens=False).input_ids
    answer_ids = tokenizer(
        row["answer"] + tokenizer.eos_token, add_special_tokens=False
    ).input_ids
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def make_loader(ds, tokenizer, max_length: int, batch_size: int, shuffle: bool) -> DataLoader:
    tokenized = ds.map(
        lambda row: tokenize_row(row, tokenizer, max_length),
        remove_columns=ds.column_names,
        desc="tokenizing",
    )
    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=AnswerOnlyCollator(tokenizer),
    )


def save_checkpoint(model, tokenizer, out_dir: Path, config: dict) -> None:
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_run", required=True, help="e.g. anchor_v1")
    p.add_argument("--run_id",     required=True, help="e.g. ga_v1")
    p.add_argument("--split",      default="forget10")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--epochs",         type=int,   default=5)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--retain_weight",  type=float, default=0.5,
                   help="Weight on retain loss to prevent catastrophic forgetting")
    p.add_argument("--max_length",     type=int,   default=512)
    p.add_argument("--output_base",    default="runs")
    p.add_argument("--limit", type=int, default=None,
                   help="Subsample forget/retain sets; for smoke tests")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    anchor_ckpt = Path(args.output_base) / args.anchor_run / "checkpoint"
    out_dir     = Path(args.output_base) / f"ga_{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(anchor_ckpt))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # enable_input_require_grads lets gradients flow through the frozen 4-bit
    # base so they reach the trainable LoRA adapter matrices
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(anchor_ckpt), is_trainable=True)
    model.print_trainable_parameters()

    tofu = load_tofu(split=args.split, variety="en")
    forget_ds, retain_ds = tofu.forget, tofu.retain
    if args.limit:
        forget_ds = forget_ds.select(range(min(args.limit, len(forget_ds))))
        retain_ds = retain_ds.select(range(min(args.limit, len(retain_ds))))
    forget_loader = make_loader(forget_ds, tokenizer, args.max_length, args.batch_size, shuffle=True)
    retain_loader = make_loader(retain_ds, tokenizer, args.max_length, args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    retain_iter = iter(retain_loader)

    config = {
        "type": "ga_unlearn",
        "method": "GradDiff",
        "anchor_run": args.anchor_run,
        "run_id": args.run_id,
        "split": args.split,
        "unlearn_variety": "en",
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "retain_weight": args.retain_weight,
    }
    history: list[dict] = []
    hist = HistoryWriter(out_dir, {**config, "total_steps": len(forget_loader) * args.epochs})
    global_step = 0

    model.train()
    for epoch in range(args.epochs):
        epoch_forget_loss = 0.0
        epoch_retain_loss = 0.0

        for forget_batch in forget_loader:
            forget_batch = {k: v.to(device) for k, v in forget_batch.items()}
            f_loss = model(**forget_batch).loss

            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            retain_batch = {k: v.to(device) for k, v in retain_batch.items()}
            r_loss = model(**retain_batch).loss

            # negate forget loss so gradient ascent pushes the model away from those facts
            loss = -f_loss + args.retain_weight * r_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            f_val, r_val = f_loss.item(), r_loss.item()
            epoch_forget_loss += f_val
            epoch_retain_loss += r_val

            global_step += 1
            hist.log(step=global_step, epoch=epoch + 1,
                     forget_loss=f_val, retain_loss=r_val)

        n = len(forget_loader)
        history.append({
            "epoch": epoch + 1,
            "forget_loss": epoch_forget_loss / n,
            "retain_loss": epoch_retain_loss / n,
        })
        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"forget_loss={epoch_forget_loss / n:.4f}  "
            f"retain_loss={epoch_retain_loss / n:.4f}",
            flush=True,
        )

        # GradDiff passes through the useful region and then diverges, so every
        # epoch is kept as its own run directory that run_audit.py can address
        # directly via --unlearn_run ga_{run_id}_ep{n}.
        epoch_dir = Path(args.output_base) / f"ga_{args.run_id}_ep{epoch + 1}"
        save_checkpoint(model, tokenizer, epoch_dir,
                        {**config, "selected_epoch": epoch + 1, "history": history})

    save_checkpoint(model, tokenizer, out_dir, {**config, "history": history})
    print(f"\nUnlearned adapters saved to {out_dir}")
    print("Per-epoch checkpoints:")
    for h in history:
        print(f"  ga_{args.run_id}_ep{h['epoch']}  "
              f"forget={h['forget_loss']:.4f}  retain={h['retain_loss']:.4f}")


if __name__ == "__main__":
    main()
