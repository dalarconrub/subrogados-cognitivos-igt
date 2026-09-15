#!/usr/bin/env python3
"""Anexo A, apartado A.3 · Formato de los corpus.

Recorre Psych-101 excluyendo todo registro de la tarea, baraja el resto con la semilla del estudio y lo
acumula hasta el presupuesto de tokens del subcorpus IGT (complemento emparejado) o sin límite (complemento
íntegro). Se detiene con error si no ha excluido ninguna sesión de la tarea.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Substrings (case-insensitive) que marcan un experimento como IGT en Psych-101.
# El IGT en Psych-101 es el dato Steingroever; se añaden términos defensivos.
IGT_BLOCKLIST = ("steingroever", "iowa", "igt", "bechara", "gambling")
# Sniff defensivo a nivel de texto (por si algún experiment-key no contuviera el
# término pero el prompt fuese IGT). Ampliado por [revisión interna].
IGT_TEXT_SNIFF = ("iowa gambling", "iowa gambling task", "four decks of cards",
                  "four decks", "deck a", "deck b", "deck c", "deck d")


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


def try_hf_revision(repo_id: str) -> str | None:
    """P2.1: captura best-effort el commit SHA del dataset HF para el audit
    (reproducibilidad del brazo noIGT, fuente externa no versionada)."""
    try:
        from huggingface_hub import dataset_info
        return dataset_info(repo_id).sha
    except Exception:
        return None


def is_igt(experiment: str, text: str, blocklist, text_sniff) -> bool:
    e = (experiment or "").lower()
    if any(b in e for b in blocklist):
        return True
    t = (text or "").lower()
    return any(s in t for s in text_sniff)


def iter_rows(args):
    """Yield dicts {experiment, text, participant?} desde HF o local jsonl."""
    if args.from_local is not None:
        with open(args.from_local, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` requerido para --from-hf. pip install datasets", file=sys.stderr)
        sys.exit(3)
    ds = load_dataset(args.from_hf, split=args.split)
    for row in ds:
        yield row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-hf", help="HF dataset id (e.g. marcelbinz/Psych-101)")
    src.add_argument("--from-local", type=Path, help="JSONL local con campos experiment/text")
    p.add_argument("--split", default="train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=710_000,
                   help="Presupuesto de tokens (~4c/tok). Default 710000 = scale-matched a LoRA-IGT. "
                        "0 = sin cap (full-scale, ~días de GPU).")
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--igt-blocklist", nargs="*", default=list(IGT_BLOCKLIST),
                   help="Substrings (lower) que marcan experiment como IGT a excluir.")
    p.add_argument("--allow-zero-igt-excluded", action="store_true",
                   help="Permite 0 IGT excluido (SOLO fuentes conocidas sin IGT, p.ej. smoke). "
                        "Por defecto es un hard-fail: el brazo noIGT depende de excluir IGT.")
    p.add_argument("--audit-output", type=Path, default=None,
                   help="Escribe un audit JSON versionable (n_total, n_igt_excluded, experiments excluidos, blocklist).")
    args = p.parse_args()

    rng = random.Random(args.seed)
    # Recolecta elegibles (no-IGT) primero para poder muestrear determinista.
    eligible: list[dict] = []
    n_total = 0
    n_igt_excluded = 0
    n_empty = 0
    igt_experiments: dict[str, int] = {}
    for row in iter_rows(args):
        n_total += 1
        exp = row.get("experiment", "")
        text = row.get("text", "")
        if is_igt(exp, text, args.igt_blocklist, IGT_TEXT_SNIFF):
            n_igt_excluded += 1
            igt_experiments[exp] = igt_experiments.get(exp, 0) + 1
            continue
        if not text or not text.strip():            # P1.3: validar text no vacío
            n_empty += 1
            continue
        eligible.append({"experiment": exp, "text": text,
                         "participant": row.get("participant")})

    if not eligible:
        print("ERROR: 0 filas elegibles tras excluir IGT", file=sys.stderr)
        return 1
    rng.shuffle(eligible)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    total_tokens = 0
    zero_marker = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, r in enumerate(eligible):
            if args.tokens and total_tokens >= args.tokens:
                break
            text = r["text"]
            if "<<" not in text:
                zero_marker += 1
            tokens = count_tokens_approx(text)
            rec = {
                "prompt_id": f"noigt_{i:06d}",
                "experiment": r["experiment"],
                "text": text,
                "n_tokens_approx": tokens,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            total_tokens += tokens

    import hashlib
    corpus_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()

    print(f"OK: corpus_noigt -> {args.output}")
    print(f"  filas leídas:        {n_total}")
    print(f"  excluidas como IGT:  {n_igt_excluded}  (experiments: {dict(sorted(igt_experiments.items()))})")
    print(f"  text vacío saltado:  {n_empty}")
    print(f"  prompts escritos:    {written}")
    print(f"  tokens (~4c/tok):    {total_tokens}  (cap={'sin cap (full-scale)' if not args.tokens else args.tokens})")
    print(f"  seed:                {args.seed}")
    print(f"  corpus sha256:       {corpus_sha}")
    if zero_marker:
        print(f"  WARNING: {zero_marker}/{written} prompts SIN markers '<<' (masked loss no entrena sobre ellos)")
    else:
        print(f"  markers '<<':        presentes en los {written} prompts")

    # P2.3: audit log versionable (provenance del brazo noIGT, que depende de fuente externa).
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps({
            "n_total": n_total, "n_igt_excluded": n_igt_excluded, "n_empty": n_empty,
            "igt_experiments_excluded": dict(sorted(igt_experiments.items())),
            "blocklist": args.igt_blocklist, "text_sniff": list(IGT_TEXT_SNIFF),
            "prompts_written": written, "tokens_approx": total_tokens, "seed": args.seed,
            "source": str(args.from_local) if args.from_local else f"hf:{args.from_hf}@{args.split}",
            "hf_revision": try_hf_revision(args.from_hf) if args.from_hf else None,
            "corpus_sha256": corpus_sha,
        }, indent=2), encoding="utf-8")
        print(f"  audit -> {args.audit_output}")

    # P1.3: hard-fail si no se excluyó IGT (la ablación depende de "nunca vio IGT").
    if n_igt_excluded == 0 and not args.allow_zero_igt_excluded:
        print("ERROR: 0 IGT excluido; blocklist/schema probablemente incorrecto. "
              "El brazo noIGT requiere excluir IGT. Usa --allow-zero-igt-excluded SOLO para "
              "fuentes conocidas sin IGT (p.ej. smoke).", file=sys.stderr)
        return 2

    print(f"  Next: train_lora_noigt.py --corpus {args.output} --config ../stage_7_5_1_lora_noise/lora_config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
