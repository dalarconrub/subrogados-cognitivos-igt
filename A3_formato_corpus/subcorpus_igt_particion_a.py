#!/usr/bin/env python3
"""Anexo A, apartado A.3 · Formato de los corpus.

Extrae de Psych-101 las 511 sesiones de la tarea que formaron parte del ajuste de Centaur y las escribe
como corpus de entrenamiento, con los marcadores de elección intactos. Es el mismo formato que la evaluación.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Partición del manifest que SÍ está en el training split de Psych-101 (IGT que
# Centaur vio). El especialista se entrena SOLO sobre ésta.
TRAIN_PARTITION = "TRAIN_psych101_original"


def count_tokens_approx(text: str) -> int:
    """Approximate token count (4 chars/token), igual que generate_noise_corpus.py."""
    return max(1, len(text) // 4)


def count_answer_markers(text: str, open_marker: str = "<<") -> int:
    return text.count(open_marker)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="igt_paper1_eval_data.json (lista de prompts IGT del manifiesto)")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL corpus_igt")
    p.add_argument("--partition", default=TRAIN_PARTITION,
                   help=f"Partición a extraer como corpus de entrenamiento (default {TRAIN_PARTITION})")
    p.add_argument("--tokens", type=int, default=None,
                   help="Cap opcional de tokens (~) para variante token-matched. Default: sin cap (todo).")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Cap opcional de nº de prompts. Default: sin cap.")
    p.add_argument("--seed", type=int, default=20260506,
                   help="Seed para el subsampleo determinista cuando se aplica --tokens/--max-prompts")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest no encontrado: {args.manifest}", file=sys.stderr)
        return 1

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = list(rows.values())

    sub = [r for r in rows if r.get("partition") == args.partition]
    if not sub:
        print(f"ERROR: 0 prompts con partition=={args.partition}", file=sys.stderr)
        return 1

    # Subsampleo determinista si se pide variante token-matched / max-prompts.
    if args.tokens is not None or args.max_prompts is not None:
        rng = random.Random(args.seed)
        rng.shuffle(sub)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    total_tokens = 0
    zero_marker = 0
    experiments: dict[str, int] = {}
    with open(args.output, "w", encoding="utf-8") as f:
        for r in sub:
            if args.max_prompts is not None and written >= args.max_prompts:
                break
            if args.tokens is not None and total_tokens >= args.tokens:
                break
            text = r["text"]
            n_mark = count_answer_markers(text)
            if n_mark == 0:
                zero_marker += 1
            tokens = count_tokens_approx(text)
            rec = {
                "prompt_id": r["prompt_id"],
                "experiment": r.get("experiment", "igt"),
                "text": text,
                "n_tokens_approx": tokens,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            experiments[rec["experiment"]] = experiments.get(rec["experiment"], 0) + 1
            written += 1
            total_tokens += tokens

    import hashlib
    corpus_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"OK: corpus_igt -> {args.output}")
    print(f"  partition:       {args.partition}")
    print(f"  prompts:         {written} (de {len(sub)} disponibles en la partición)")
    print(f"  tokens (~4c/tok):{total_tokens}")
    print(f"  seed:            {args.seed}")
    print(f"  corpus sha256:   {corpus_sha}")
    print(f"  experiments:     {dict(sorted(experiments.items()))}")
    if zero_marker:
        print(f"  WARNING: {zero_marker} prompts SIN markers '<<' (masked loss no entrenaría sobre ellos)")
    else:
        print(f"  markers '<<':    presentes en los {written} prompts (masked loss OK)")
    print(f"  Next: train_lora_igt.py --corpus {args.output} --config ../stage_7_5_1_lora_noise/lora_config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
