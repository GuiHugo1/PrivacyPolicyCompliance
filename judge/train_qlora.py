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


# Backends that surface through torch's CUDA API. A ROCm-built torch
# reports an AMD GPU via the same `torch.cuda.*` calls CUDA uses, so
# "rocm" is just a label for "I expect torch.cuda to be ROCm-backed" --
# the detection/dispatch code is identical to "cuda".
_CUDA_LIKE_BACKENDS = ("cuda", "rocm")


def resolve_device_backend(want_backend: str = "auto") -> tuple[str, "torch.device"]:
    """Returns ``(backend_name, device)`` for the requested ``device.backend``.

    Options:

    - ``"cuda"``/``"rocm"`` -- an NVIDIA (CUDA) or AMD (ROCm) GPU, detected
      via ``torch.cuda.is_available()``. ROCm has no Windows build, so on
      Windows this only ever succeeds for an NVIDIA card; an AMD card on
      Windows needs ``"directml"`` instead.
    - ``"directml"`` -- AMD/Intel GPU acceleration on Windows via the
      separate ``torch-directml`` package. Training support through HF
      ``Trainer`` is best-effort (bitsandbytes 4-bit quantization and
      bf16/fp16 autocast aren't available on this backend), so this is
      most useful for the smaller, unquantized configs.
    - ``"cpu"`` -- force CPU, ignoring any GPU present.
    - ``"auto"`` (default) -- try CUDA/ROCm, then DirectML, then fall back
      to CPU.

    Raises ``RuntimeError``/``ValueError`` for an explicitly requested
    backend that isn't actually available, so a typo'd config fails loudly
    instead of silently training on the wrong device.
    """
    if want_backend in _CUDA_LIKE_BACKENDS:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device.backend: {want_backend} was requested but torch.cuda.is_available() "
                "is False -- no CUDA/ROCm GPU visible to torch. On Windows with an AMD GPU, "
                "use device.backend: directml instead (ROCm has no Windows build)."
            )
        return want_backend, torch.device("cuda")

    if want_backend == "directml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError(
                "device.backend: directml was requested but the `torch-directml` package "
                "isn't installed. Install it with `pip install torch-directml` (Windows only)."
            ) from exc
        return "directml", torch_directml.device()

    if want_backend == "cpu":
        return "cpu", torch.device("cpu")

    if want_backend != "auto":
        raise ValueError(
            f"Unknown device.backend: {want_backend!r} (expected one of "
            "'auto', 'cuda', 'rocm', 'directml', 'cpu')"
        )

    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    try:
        import torch_directml

        return "directml", torch_directml.device()
    except ImportError:
        pass
    return "cpu", torch.device("cpu")


def build_bnb_config(quant_cfg: dict[str, Any], backend: str = "cuda") -> BitsAndBytesConfig | None:
    """Returns ``None`` when ``quantization.enabled`` is false, or when the
    resolved device ``backend`` can't run bitsandbytes 4-bit quantization.

    bitsandbytes 4-bit quantization only supports CUDA/ROCm GPUs. CPU
    training (``judge/config/qlora_judge_cpu.yaml``) and DirectML training
    (``judge/config/qlora_judge_amd.yaml``) both set ``enabled: false``
    explicitly, but this also guards against a config that leaves
    quantization enabled while pointing ``device.backend`` at ``cpu`` or
    ``directml``.
    """
    if not quant_cfg.get("enabled", True):
        return None
    if backend not in _CUDA_LIKE_BACKENDS:
        print(
            f"[train_qlora] quantization.enabled is true but device.backend={backend!r} "
            "doesn't support bitsandbytes 4-bit quantization; loading at full precision instead."
        )
        return None
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_model_and_tokenizer(
    cfg: dict[str, Any], backend: str, device: "torch.device"
) -> tuple[Any, Any]:
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"], trust_remote_code=model_cfg.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(cfg["quantization"], backend)
    # "auto" (accelerate GPU dispatch) only makes sense for a quantized
    # CUDA/ROCm load; every other case picks a single explicit device below
    # (a CPU config can still override via model.device_map, e.g. "cpu").
    device_map = model_cfg.get("device_map")
    if device_map is None:
        device_map = "auto" if bnb_config is not None else None
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    elif device_map is None and backend not in ("cpu",):
        # DirectML (and any future non-accelerate-aware backend) has no
        # `device_map="auto"` dispatch support, so place the whole model on
        # the resolved device explicitly instead.
        model = model.to(device)

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

def resolve_mixed_precision(want_bf16: bool, backend: str) -> tuple[bool, bool]:
    """Returns (bf16, fp16), downgrading to fp16 when the GPU can't do bf16.

    ``torch.cuda.is_bf16_supported()`` is False on pre-Ampere GPUs (T4, V100, most GTX/RTX 20-series) even though CUDA itself is available, and HF's
    ``TrainingArguments`` raises rather than falling back on its own.

    Both bf16 and fp16 autocast need a CUDA/ROCm device; CPU and DirectML
    training always fall back to plain fp32 regardless of what the config
    requests.
    """
    if backend not in _CUDA_LIKE_BACKENDS:
        if want_bf16:
            print(f"[train_qlora] bf16 requested but device.backend={backend!r}; training in fp32.")
        return False, False
    if want_bf16 and not torch.cuda.is_bf16_supported():
        print("[train_qlora] bf16 requested but unsupported on this GPU; falling back to fp16.")
        return False, True
    return want_bf16, False

def build_training_args(t_cfg: dict[str, Any], backend: str) -> TrainingArguments:
    bf16, fp16 = resolve_mixed_precision(t_cfg.get("bf16", True), backend)
    # Pinned host memory only speeds up transfers to a CUDA/ROCm device --
    # on CPU/DirectML it does nothing but trips PyTorch's "pin_memory is
    # set but no accelerator is found" warning, so only request it when a
    # CUDA-like backend is actually in use.
    is_cuda_like = backend in _CUDA_LIKE_BACKENDS
    return TrainingArguments(
        output_dir=str(t_cfg["output_dir"]),
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(t_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=t_cfg.get("warmup_steps", 0.03),
        weight_decay=t_cfg.get("weight_decay", 0.0),
        max_grad_norm=t_cfg.get("max_grad_norm", 0.3),
        logging_steps=t_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=t_cfg.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=t_cfg.get("save_steps", 50),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        seed=t_cfg.get("seed", 42),
        optim=t_cfg.get("optim", "paged_adamw_8bit" if is_cuda_like else "adamw_torch"),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        bf16=bf16,
        fp16=fp16,
        dataloader_pin_memory=t_cfg.get("dataloader_pin_memory", is_cuda_like),
        use_cpu=backend == "cpu",
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

    backend, device = resolve_device_backend(cfg.get("device", {}).get("backend", "auto"))
    print(f"[train_qlora] using device backend: {backend} ({device})")
    if backend == "directml":
        print(
            "[train_qlora] DirectML training via HF Trainer is best-effort: whether the "
            "model actually stays on the DirectML device (rather than being moved back to "
            "CPU by Trainer's own device dispatch) depends on your installed `accelerate` "
            "version. If training errors out or silently runs on CPU, fall back to "
            "device.backend: cpu."
        )

    model, tokenizer = build_model_and_tokenizer(cfg, backend, device)

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
        args=build_training_args(cfg["training"], backend),
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
