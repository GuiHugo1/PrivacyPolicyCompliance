"""Evaluates a trained LoRA judge adapter on the held-out test split.

Runs the (4-bit base model + adapter) on every test-set clause, parses each
generated verdict as JSON, and reports:

  - JSON-validity rate (fraction of outputs that parse and match
    ``judge/judge_schema.json``)
  - per-``compliance_status``-class precision/recall/F1 and macro-F1 (see
    ``judge.eval_metrics.per_class_prf1`` for how an invalid/unparseable
    output is scored: it always costs recall on its gold label)

Usage::

    uv sync --group judge
    python -m judge.eval_qlora --config judge/config/qlora_judge.yaml \\
        --adapter judge/checkpoints/qwen2.5-7b-qlora-judge \\
        --output judge/metrics/test_eval_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge.eval_metrics import (
    COMPLIANCE_CLASSES,
    json_validity_rate,
    macro_f1,
    parse_verdict,
    per_class_prf1,
)
from judge.qlora_data import load_prompt_response_pairs
from judge.schema_utils import DEFAULT_SCHEMA_PATH, extract_json_object, load_schema
from judge.train_qlora import build_bnb_config, load_config


def load_model_for_eval(cfg: dict[str, Any], adapter_path: Path) -> tuple[Any, Any]:
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=build_bnb_config(cfg["quantization"]),
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_verdicts(
    model: Any, tokenizer: Any, prompts: list[str], gen_kwargs: dict[str, Any]
) -> list[str]:
    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 256),
                do_sample=gen_kwargs.get("do_sample", False),
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs.append(
                tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            )
    return outputs


def build_test_report(
    generations: list[str],
    gold_responses: list[str],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Pure aggregation step, split out from ``evaluate_test_split`` so it's
    unit-testable without a real model/tokenizer."""
    parsed = [parse_verdict(text, schema) for text in generations]

    y_true: list[str] = []
    for gold_text in gold_responses:
        gold_obj = extract_json_object(gold_text)
        if gold_obj is None or "compliance_status" not in gold_obj:
            raise ValueError(
                "gold response is not a valid verdict JSON with `compliance_status`; "
                "check the test split"
            )
        y_true.append(gold_obj["compliance_status"])
    y_pred = [p.verdict.get("compliance_status") if p.verdict else None for p in parsed]

    class_metrics = per_class_prf1(y_true, y_pred, labels=COMPLIANCE_CLASSES)
    return {
        "n_examples": len(generations),
        "json_validity_rate": json_validity_rate(parsed),
        "macro_f1": round(macro_f1(class_metrics), 4),
        "per_class": {
            label: {
                "support": m.support,
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1": round(m.f1, 4),
            }
            for label, m in class_metrics.items()
        },
    }


def evaluate_test_split(
    model: Any,
    tokenizer: Any,
    test_path: Path | str,
    schema: dict[str, Any],
    gen_kwargs: dict[str, Any],
) -> dict[str, Any]:
    def apply_chat_template(messages: list[dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    examples = load_prompt_response_pairs(test_path, apply_chat_template)
    prompts = [e["prompt"] for e in examples]
    gold_responses = [e["response"] for e in examples]

    generations = generate_verdicts(model, tokenizer, prompts, gen_kwargs)
    return build_test_report(generations, gold_responses, schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("judge/config/qlora_judge.yaml"))
    parser.add_argument(
        "--adapter", type=Path, required=True, help="Path to the saved LoRA adapter directory"
    )
    parser.add_argument("--output", type=Path, default=Path("judge/metrics/test_eval_report.json"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    schema = load_schema(cfg["data"].get("schema_path", DEFAULT_SCHEMA_PATH))
    model, tokenizer = load_model_for_eval(cfg, args.adapter)

    report = evaluate_test_split(
        model, tokenizer, cfg["data"]["test_path"], schema, cfg.get("generation", {})
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
