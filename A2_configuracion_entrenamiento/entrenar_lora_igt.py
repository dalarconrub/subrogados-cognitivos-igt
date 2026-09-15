#!/usr/bin/env python3
"""Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.

Invoca al entrenador común con el subcorpus IGT (las 511 sesiones de la partición A); la configuración es la
común y la única diferencia con los demás adaptadores es el corpus.
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
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0 (got {iv})")
    return iv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True, help="Input JSONL corpus IGT (contenido conductual)")
    p.add_argument("--output_adapter", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True,
                   help="lora_config.yaml (EL MISMO que LoRA-noise/irrelevant)")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max-seq-length", dest="max_seq_length", type=positive_int, default=None,
                   help="Override training.max_seq_length (forwarded). Canonical=32768 (A100/H100 80GB).")
    p.add_argument("--resume_from_checkpoint", type=Path, default=None,
                   help="Forwarded a train_lora_noise.py para resume support.")
    args = p.parse_args()

    if not TRAIN_LORA_NOISE.exists():
        print(f"ERROR: cannot find {TRAIN_LORA_NOISE}", file=sys.stderr)
        print("       train_lora_noise.py debe existir (lo reusa este wrapper)", file=sys.stderr)
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

    print(f"Delegating to {TRAIN_LORA_NOISE.name} (reusing identical LoRA training logic)...")
    print(f"  Corpus: {args.corpus}")
    print(f"  Output: {args.output_adapter}")
    print(f"  Config: {args.config}")
    print("  (única variable vs LoRA-noise/irrelevant: el corpus = IGT conductual)")
    print()
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
