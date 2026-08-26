"""SFT-record loading, prompt/response normalization, and the loss-masking
data collator for QLoRA fine-tuning of the GDPR judge.

Accepts two SFT JSONL record shapes so training works both against this
repo's own generator (``judge.build_sft_dataset``, chat ``messages``
records) and a plain ``instruction``/``input``/``output`` shape::

    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", "content": "<json verdict>"}]}
    {"instruction": "...", "input": "...", "output": "<json verdict>"}

``JudgeSFTCollator`` needs ``torch`` (``uv sync --group judge``); that
import is deferred to ``__call__`` so the rest of this module -- record
loading, normalization, and the pure-list masking/padding helpers --
stays importable and unit-testable without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from judge.build_sft_dataset import JUDGE_SYSTEM_PROMPT


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ChatTemplater(Protocol):
    def __call__(self, messages: list[dict[str, str]]) -> str: ...


def normalize_record(record: dict[str, Any], apply_chat_template: ChatTemplater) -> tuple[str, str]:
    """Returns ``(prompt_text, response_text)`` for one raw SFT record.

    ``prompt_text`` is the fully rendered chat prompt (system + user turns,
    generation-prompt appended) the model sees at inference time;
    ``response_text`` is the raw JSON-verdict string it's trained to
    reproduce.
    """
    if "messages" in record:
        messages = record["messages"]
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user = next(m["content"] for m in messages if m["role"] == "user")
        response = next(m["content"] for m in messages if m["role"] == "assistant")
    elif "instruction" in record and "output" in record:
        extra_input = record.get("input") or ""
        instruction = record["instruction"]
        user = f"{instruction}\n\n{extra_input}".strip() if extra_input else instruction
        response = record["output"]
        system = JUDGE_SYSTEM_PROMPT
    else:
        raise ValueError(f"unrecognized SFT record schema, keys={sorted(record)}")

    if not isinstance(response, str):
        response = json.dumps(response, ensure_ascii=False)

    chat = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    prompt = apply_chat_template(chat)
    return prompt, response.strip()


def load_prompt_response_pairs(
    path: Path | str, apply_chat_template: ChatTemplater
) -> list[dict[str, str]]:
    return [
        dict(zip(("prompt", "response"), normalize_record(r, apply_chat_template), strict=True))
        for r in load_jsonl(path)
    ]


def _encode_and_mask(
    prompt_ids: list[int],
    response_ids: list[int],
    eos_id: int | None,
    max_length: int,
) -> dict[str, list[int]]:
    """Pure-list core of the collator: builds ``input_ids``/``labels`` for
    one example with prompt tokens masked to ``-100`` so the loss is
    computed only on the JSON-output tokens (plus the trailing EOS)."""
    terminated_response = list(response_ids) + ([eos_id] if eos_id is not None else [])
    input_ids = list(prompt_ids) + terminated_response
    labels = [-100] * len(prompt_ids) + terminated_response
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def _pad_batch(encoded: list[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    max_len = max(len(e["input_ids"]) for e in encoded)
    input_ids, attention_mask, labels = [], [], []
    for e in encoded:
        pad_len = max_len - len(e["input_ids"])
        input_ids.append(e["input_ids"] + [pad_id] * pad_len)
        attention_mask.append([1] * len(e["input_ids"]) + [0] * pad_len)
        labels.append(e["labels"] + [-100] * pad_len)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class JudgeSFTCollator:
    """Causal-LM SFT collator that masks prompt tokens from the loss, so
    gradients only flow through the JSON-verdict tokens (+ trailing EOS)."""

    def __init__(self, tokenizer: Any, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features: list[dict[str, str]]) -> dict[str, Any]:
        import torch

        eos_id = self.tokenizer.eos_token_id
        encoded = []
        for feature in features:
            prompt_ids = self.tokenizer(feature["prompt"], add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(feature["response"], add_special_tokens=False)[
                "input_ids"
            ]
            encoded.append(_encode_and_mask(prompt_ids, response_ids, eos_id, self.max_length))
        batch = _pad_batch(encoded, self.tokenizer.pad_token_id)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
