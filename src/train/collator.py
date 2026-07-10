from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class AnswerOnlyCollator:
    """Pads a batch of pre-tokenized examples while keeping -100 labels on
    question tokens so the training loss is computed over answer tokens only."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }
