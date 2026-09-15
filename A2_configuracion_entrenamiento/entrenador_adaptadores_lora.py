#!/usr/bin/env python3
"""Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.

Es el único entrenador de adaptadores del estudio; los demás scripts de entrenamiento lo invocan con
otro corpus. Carga la base Llama-3.1-70B congelada y cuantizada a 4 bits, monta el adaptador LoRA de rango 8 sobre
los siete módulos lineales de cada bloque, calcula la pérdida solo sobre los tokens comprendidos entre los
marcadores “<<” y “>>” y optimiza con AdamW de 8 bits siguiendo la configuración del fichero de configuración.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def positive_int(value: str) -> int:
    """argparse type validator: int > 0 (R7 §10 polish)."""
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0 (got {iv})")
    return iv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True, help="Input JSONL corpus")
    p.add_argument("--output_adapter", type=Path, required=True, help="Output adapter dir")
    p.add_argument("--config", type=Path, required=True, help="lora_config.yaml")
    p.add_argument("--base_model_id", default=None, help="Override base model id (default from config)")
    p.add_argument("--dry_run", action="store_true", help="Print plan without launching training")
    # [revisión interna] O8.N9: resume support
    p.add_argument("--resume_from_checkpoint", type=Path, default=None,
                   help="Path to checkpoint directory to resume from (e.g., output_adapter/_training_logs/checkpoint-N)")
    # [revisión interna] O12.N1: CLI override de max_seq_length para reduced A100 40GB path.
    # Default None → use config value (canonical 32768). Si flag pasado, override
    # config valor antes de tokenize. Operacional para --reduced-seq-plan
    # path en A100 40GB (target: max_seq_length=16384).
    p.add_argument("--max-seq-length", dest="max_seq_length", type=positive_int, default=None,
                   help="Override training.max_seq_length from config. Canonical=32768 (A100 80GB); "
                        "reduced=16384 (A100 40GB con --reduced-seq-plan en check_colab_env.py). "
                        "WARNING: reducing may drop answer spans en prompts long-context. "
                        "Must be a positive integer (R7 §10 polish).")
    return p.parse_args()


def set_seed_everywhere(seed: int) -> None:
    """[revisión interna] O8.N9: comprehensive seed propagation."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml required. pip install pyyaml", file=sys.stderr)
        sys.exit(3)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_lora_kwargs(config: dict) -> dict:
    lora = config["lora"]
    return {
        "r": lora["r"],
        "lora_alpha": lora["lora_alpha"],
        "lora_dropout": lora["lora_dropout"],
        "bias": lora["bias"],
        "task_type": lora["task_type"],
        "target_modules": lora["target_modules"],
        "use_rslora": lora.get("use_rslora", True),
    }


def load_corpus_for_training(corpus_path: Path) -> list[dict]:
    """Load JSONL corpus into list of {"text": ...} records for trainer."""
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append({"text": d["text"]})
    return records


def main() -> int:
    args = parse_args()

    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    base_model_id = args.base_model_id or config["base_model"]["id"]
    training = config["training"]

    # [revisión interna] O12.N1: CLI --max-seq-length override de training.max_seq_length
    # del config. Print warning si differs del canonical para que el operator
    # tenga visibilidad explícita.
    if args.max_seq_length is not None:
        canonical = training["max_seq_length"]
        training["max_seq_length"] = args.max_seq_length
        if args.max_seq_length != canonical:
            print(f"[O12.N1] OVERRIDE: max_seq_length {canonical} -> {args.max_seq_length} (CLI flag)")
            if args.max_seq_length < canonical:
                print(f"         REDUCED path: ensure check_colab_env.py was called with --reduced-seq-plan")

    print("=" * 60)
    print("LoRA-noise training plan")
    print("=" * 60)
    print(f"Base model:    {base_model_id}")
    print(f"Corpus:        {args.corpus}")
    print(f"Output adapter:{args.output_adapter}")
    print(f"LoRA config:")
    for k, v in build_lora_kwargs(config).items():
        print(f"  {k}: {v}")
    print(f"Training config:")
    for k, v in training.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\nDRY-RUN — no actual training launched.")
        return 0

    # === Actual training (executed only on Lightning GPU Studio) ===
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
        # [revisión interna] O8.N2: use answer-token masked DataCollator (NOT
        # DataCollatorForLanguageModeling which trains full-sequence loss).
        # See `masked_loss_collator.py` for the Centaur-style masking spec.
        from masked_loss_collator import AnswerTokenMaskedDataCollator
    except ImportError as e:
        print(f"\nERROR: missing dep: {e}", file=sys.stderr)
        print("  pip install torch transformers peft bitsandbytes accelerate datasets", file=sys.stderr)
        return 3

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires GPU (A100 recommended).", file=sys.stderr)
        return 3

    # [revisión interna] O8.N9: comprehensive seed propagation
    set_seed_everywhere(training["seed"])
    print(f"[seed] all RNGs seeded to {training['seed']}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=config["base_model"]["load_in_4bit"],
        bnb_4bit_quant_type=config["base_model"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["base_model"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, config["base_model"]["bnb_4bit_compute_dtype"]),
    )

    print("\n[1/6] Loading tokenizer + base model (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    print("[2/6] Applying LoRA adapter...")
    lora_cfg = LoraConfig(**build_lora_kwargs(config))
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[3/6] Loading + tokenizing corpus...")
    records = load_corpus_for_training(args.corpus)
    ds = Dataset.from_list(records)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=training["max_seq_length"])

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    print("[4/6] Building Trainer...")
    args_tr = TrainingArguments(
        output_dir=str(args.output_adapter / "_training_logs"),
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_steps=training["warmup_steps"],
        lr_scheduler_type=training["lr_scheduler_type"],
        optim=training["optim"],
        bf16=training.get("bf16", True),
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        logging_steps=config["logging"]["logging_steps"],
        save_steps=config["logging"]["save_steps"],
        save_total_limit=config["logging"]["save_total_limit"],
        report_to=config["logging"].get("report_to", "none"),
        seed=training["seed"],
    )
    # [revisión interna] O8.N2: Centaur-style answer-token masked loss.
    # Tokens fuera de `<<...>>` markers reciben label=-100 (ignore en CE loss).
    data_collator = AnswerTokenMaskedDataCollator(
        tokenizer=tokenizer,
        open_marker=config["masking"]["answer_marker_open"],
        close_marker=config["masking"]["answer_marker_close"],
    )
    trainer = Trainer(model=model, args=args_tr, train_dataset=ds_tok, data_collator=data_collator)

    # [revisión interna] O8.N9: resume support — pasar resume_from_checkpoint si --resume
    print("[5/6] Training (~20h on A100 80GB)...")
    if args.resume_from_checkpoint:
        print(f"  Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint))
    else:
        trainer.train()

    print("[6/6] Saving adapter...")
    args.output_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_adapter))
    tokenizer.save_pretrained(str(args.output_adapter))
    print(f"\nOK: LoRA-noise adapter saved to {args.output_adapter}")
    print("\nNext: run eval_2way.py to compare Centaur vs LoRA-noise on IGT data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
