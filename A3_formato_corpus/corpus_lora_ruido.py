#!/usr/bin/env python3
"""Anexo A, apartado A.3 · Formato de los corpus.

Genera registros sintéticos que conservan el esquema instrucción-más-ensayos con marcadores, con
elecciones y consecuencias muestreadas al azar sin relación entre sí, hasta un presupuesto de unos cien mil tokens,
con la semilla del estudio.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Vocab fragments inspired by Psych-101 structure (sin contenido semántico real)
TASK_TEMPLATES = [
    "You are completing a task with options labeled {opts}. You select one by pressing the corresponding key.",
    "In each round you choose between options {opts}. You will see outcomes after each choice.",
    "Repeat the following: choose one of {opts}. The outcome may vary.",
    "Decision task: pick from {opts}. Press the key, receive feedback.",
    "You have {n} options: {opts}. Choose freely; outcomes are reported.",
]

OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y", "Z"]


def gen_prompt(rng: random.Random, n_trials: int = 20) -> str:
    """Generate one synthetic noise prompt with <<X>> answer markers."""
    n_opts = rng.randint(2, 4)
    opts = sorted(rng.sample(OPTION_LABELS, n_opts))
    template = rng.choice(TASK_TEMPLATES)
    head = template.format(opts=", ".join(opts), n=n_opts)
    lines = [head, ""]
    for _ in range(n_trials):
        choice = rng.choice(opts)
        reward = rng.choice(["+1 points", "-1 points", "+0 points", "+2 points", "-2 points",
                             "5 points", "10 points", "0 points"])
        line = f"You press <<{choice}>>. You get {reward}."
        lines.append(line)
    return "\n".join(lines)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (4 chars/token rule of thumb)."""
    return max(1, len(text) // 4)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Output JSONL")
    p.add_argument("--tokens", type=int, default=100_000, help="Target total tokens (~)")
    p.add_argument("--num-prompts", type=int, default=None, help="Override: exact number of prompts")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--trials-per-prompt", type=int, default=20)
    args = p.parse_args()

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    total_tokens = 0
    with open(args.output, "w", encoding="utf-8") as f:
        idx = 0
        while True:
            if args.num_prompts is not None and written >= args.num_prompts:
                break
            if args.num_prompts is None and total_tokens >= args.tokens:
                break
            text = gen_prompt(rng, n_trials=args.trials_per_prompt)
            tokens = count_tokens_approx(text)
            rec = {
                "prompt_id": f"noise_{idx:06d}",
                "experiment": "synthetic_noise",
                "text": text,
                "n_tokens_approx": tokens,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            total_tokens += tokens
            idx += 1

    print(f"OK: wrote {written} prompts (~{total_tokens} tokens approx) to {args.output}")
    print(f"  Seed: {args.seed}")
    print(f"  Format: JSONL with fields (prompt_id, experiment, text, n_tokens_approx)")
    print(f"  Next: use as input to train_lora_noise.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
