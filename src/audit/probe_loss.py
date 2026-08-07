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
    split_role: str = "forget",
    device: str | None = None,
) -> list[dict]:
    """Compute per-fact cross-entropy loss on answer tokens only.

    Low score = model still predicts the answer = fact not truly unlearned.
    Returns records in the audit JSONL schema (gold_gap left None until
    a gold model is available).
    """
    # Infer device from the model rather than moving it: 4-bit bitsandbytes
    # models and accelerate-dispatched models raise on .to()
    if device is None:
        device = next(model.parameters()).device

    model.eval()

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

        # Arabic tokenizes ~2x less efficiently than English in most tokenizers,
        # so mean-per-token loss dilutes a fixed amount of knowledge across more
        # tokens. sum_score (nats for the whole answer) is comparable across
        # languages because it covers the same semantic content either way.
        n_answer_tokens = len(answer_ids)

        results.append({
            "fact_id": row["fact_id"],
            "variety": variety,
            "split_role": split_role,
            "attack": "loss",
            "score": round(loss.item(), 6),
            "sum_score": round(loss.item() * n_answer_tokens, 6),
            "n_answer_tokens": n_answer_tokens,
            "gold_gap": None,
            "run_id": run_id,
        })

    return results
