"""QLoRA fine-tuning of Qwen2.5-7B-Instruct as a GDPR compliance judge.

Config-driven (see ``judge/config/qlora_judge.yaml``): every hyperparameter
-- LoRA rank/alpha/target modules, quantization, learning rate, epochs,
batch size, gradient accumulation -- comes from YAML, so a sweep is "copy
the file, edit a few keys" rather than a wall of CLI flags.

Only the LoRA adapter is saved (``trainer.model`` is a ``PeftModel``, so
``save_pretrained`` writes adapter weights + config, not a merged model),
keeping checkpoints small.

Usage::

    uv sync --group judge
    python -m judge.train_qlora --config judge/config/qlora_judge.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from judge.eval_metrics import json_validity_rate, parse_verdict
from judge.metrics_logger import MetricsLogger
from judge.qlora_data import JudgeSFTCollator, load_prompt_response_pairs
from judge.schema_utils import DEFAULT_SCHEMA_PATH, load_schema


def load_config(path: Path | str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_bnb_config(quant_cfg: dict[str, Any]) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any]:
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"], trust_remote_code=model_cfg.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=build_bnb_config(cfg["quantization"]),
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules"),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


class JsonValidityCallback(TrainerCallback):
    """After each ``Trainer.evaluate()``, generates completions for a
    capped sample of the val set and logs train/val loss + JSON-validity
    rate -- a metric HF's default eval loop can't produce on its own since
    it only scores teacher-forced next-token loss, not free-form
    generation.
    """

    def __init__(
        self,
        val_examples: list[dict[str, str]],
        tokenizer: Any,
        schema: dict[str, Any],
        gen_kwargs: dict[str, Any],
        metrics_logger: MetricsLogger,
        max_samples: int,
    ):
        self.val_examples = val_examples[:max_samples]
        self.tokenizer = tokenizer
        self.schema = schema
        self.gen_kwargs = gen_kwargs
        self.metrics_logger = metrics_logger

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs["model"]
        was_training = model.training
        model.eval()
        parsed = []
        with torch.no_grad():
            for example in self.val_examples:
                inputs = self.tokenizer(example["prompt"], return_tensors="pt", truncation=True).to(
                    model.device
                )
                generated = model.generate(
                    **inputs,
                    max_new_tokens=self.gen_kwargs.get("max_new_tokens", 256),
                    do_sample=self.gen_kwargs.get("do_sample", False),
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                text = self.tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                parsed.append(parse_verdict(text, self.schema))
        if was_training:
            model.train()

        train_loss = None
        for entry in reversed(state.log_history):
            if "loss" in entry:
                train_loss = entry["loss"]
                break

        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "train_loss": train_loss,
            "eval_loss": (metrics or {}).get("eval_loss"),
            "json_validity_rate": json_validity_rate(parsed),
        }
        self.metrics_logger.log(record)
        print(f"[metrics] {record}")


def build_training_args(t_cfg: dict[str, Any]) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(t_cfg["output_dir"]),
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(t_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        weight_decay=t_cfg.get("weight_decay", 0.0),
        max_grad_norm=t_cfg.get("max_grad_norm", 0.3),
        logging_steps=t_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=t_cfg.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=t_cfg.get("save_steps", 50),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        seed=t_cfg.get("seed", 42),
        optim=t_cfg.get("optim", "paged_adamw_8bit"),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        bf16=t_cfg.get("bf16", True),
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("judge/config/qlora_judge.yaml"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    torch.manual_seed(cfg["training"].get("seed", 42))

    model, tokenizer = build_model_and_tokenizer(cfg)

    def apply_chat_template(messages: list[dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    data_cfg = cfg["data"]
    train_examples = load_prompt_response_pairs(data_cfg["train_path"], apply_chat_template)
    val_examples = load_prompt_response_pairs(data_cfg["val_path"], apply_chat_template)
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    max_length = cfg["model"].get("max_seq_length", 2048)
    collator = JudgeSFTCollator(tokenizer, max_length=max_length)
    schema = load_schema(data_cfg.get("schema_path", DEFAULT_SCHEMA_PATH))

    metrics_cfg = cfg.get("metrics", {})
    metrics_logger = MetricsLogger(
        Path(metrics_cfg.get("output_dir", "judge/metrics"))
        / metrics_cfg.get("metrics_file", "training_metrics.jsonl"),
        use_mlflow=metrics_cfg.get("use_mlflow", False),
        mlflow_dir=metrics_cfg.get("mlflow_dir"),
        run_name=metrics_cfg.get("run_name"),
    )
    callback = JsonValidityCallback(
        val_examples,
        tokenizer,
        schema,
        cfg.get("generation", {}),
        metrics_logger,
        max_samples=metrics_cfg.get("max_eval_generate_samples", 50),
    )

    trainer = Trainer(
        model=model,
        args=build_training_args(cfg["training"]),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[callback],
    )

    trainer.train()

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)  # PeftModel -> adapter-only weights
    tokenizer.save_pretrained(output_dir)
    metrics_logger.close()

    print(f"LoRA adapter saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
