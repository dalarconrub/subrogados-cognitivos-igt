#!/usr/bin/env python3
"""Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.

Invoca al entrenador común con el corpus irrelevante; todo lo demás (configuración, base, pérdida enmascarada,
semilla) es idéntico al resto de adaptadores, de modo que la única diferencia es el corpus.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
TRAIN_LORA_NOISE = (
    REPO_ROOT
    / "tesis"
    / "data_analyses"
    / "llm_evaluation"
    / "paper_01_igt"
    / "stage_7_5_1_lora_noise"
    / "train_lora_noise.py"
)


def positive_int(value: str) -> int:
    """argparse type validator: int > 0 (R7 §10 polish)."""
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0 (got {iv})")
    return iv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True, help="Input JSONL corpus (academic non-behavioral)")
    p.add_argument("--output_adapter", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="lora_config.yaml (SAME as LoRA-noise)")
    p.add_argument("--dry_run", action="store_true")
    # [revisión interna] O12.N1: propagate --max-seq-length al subprocess delegate
    # para mantener config-isolation (mismo override path en ambos adapters).
    p.add_argument("--max-seq-length", dest="max_seq_length", type=positive_int, default=None,
                   help="Override training.max_seq_length from config (forwarded to train_lora_noise.py). "
                        "Canonical=32768 (A100 80GB); reduced=16384 (A100 40GB). Must be > 0.")
    p.add_argument("--resume_from_checkpoint", type=Path, default=None,
                   help="Forwarded to train_lora_noise.py para resume support.")
    args = p.parse_args()

    if not TRAIN_LORA_NOISE.exists():
        print(f"ERROR: cannot find {TRAIN_LORA_NOISE}", file=sys.stderr)
        print("       train_lora_noise.py must exist (reused by this wrapper)", file=sys.stderr)
        return 3

    cmd = [
        sys.executable, str(TRAIN_LORA_NOISE),
        "--corpus", str(args.corpus),
        "--output_adapter", str(args.output_adapter),
        "--config", str(args.config),
    ]
    if args.dry_run:
        cmd.append("--dry_run")
    if args.max_seq_length is not None:
        cmd.extend(["--max-seq-length", str(args.max_seq_length)])
    if args.resume_from_checkpoint is not None:
        cmd.extend(["--resume_from_checkpoint", str(args.resume_from_checkpoint)])

    print(f"Delegating to {TRAIN_LORA_NOISE.name} (reusing LoRA training logic)...")
    print(f"  Corpus: {args.corpus}")
    print(f"  Output: {args.output_adapter}")
    print(f"  Config: {args.config}")
    if args.max_seq_length is not None:
        print(f"  max_seq_length OVERRIDE: {args.max_seq_length}")
    print()
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
