from __future__ import annotations

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

_TEMPLATE = "Question: {question}\nAnswer: "


@torch.no_grad()
def probe_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    variety: str = "en",
    run_id: str = "",
    device: str | None = None,
) -> list[dict]:
    """Compute per-fact cross-entropy loss on answer tokens only.

    Low score = model still predicts the answer = fact not truly unlearned.
    Returns records in the audit JSONL schema (gold_gap left None until
    a gold model is available).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    results = []
    for row in tqdm(dataset, desc=f"probe_loss [{variety}]"):
        prompt = _TEMPLATE.format(question=row["question"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = tokenizer(row["answer"], add_special_tokens=False).input_ids

        full_ids = prompt_ids + answer_ids
        # -100 masks question tokens so loss is only over the answer
        labels = [-100] * len(prompt_ids) + list(answer_ids)

        input_ids = torch.tensor([full_ids], device=device)
        label_ids = torch.tensor([labels], device=device)

        loss = model(input_ids=input_ids, labels=label_ids).loss

        results.append({
            "fact_id": row["fact_id"],
            "variety": variety,
            "attack": "loss",
            "score": round(loss.item(), 6),
            "gold_gap": None,
            "run_id": run_id,
        })

    return results
