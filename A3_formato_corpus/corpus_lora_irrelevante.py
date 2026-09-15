#!/usr/bin/env python3
"""Anexo A, apartado A.3 · Formato de los corpus.

Convierte la instantánea de Wikipedia en registros con el mismo esquema que los demás corpora, con el
texto entre los marcadores “<<” y “>>”.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Wikipedia-like fragments (curated topics, no behavioral content).
WIKI_NATURAL_SCIENCES_SEEDS = [
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "Tectonic plates move at rates of a few centimeters per year, driven by mantle convection.",
    "The hydrogen atom consists of one proton and one electron and is the most abundant element in the universe.",
    "Mitochondria are double-membraned organelles found in most eukaryotic cells.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "Mendelian inheritance describes how genetic traits are passed from parents to offspring.",
    "The periodic table organizes chemical elements by atomic number and electron configuration.",
    "Crystalline structures form when atoms arrange in a repeating three-dimensional pattern.",
    "Black holes form when massive stars collapse under their own gravity after exhausting nuclear fuel.",
    "Enzymes are biological catalysts that accelerate chemical reactions in living organisms.",
    "The carbon cycle describes the movement of carbon between atmosphere, biosphere, and lithosphere.",
    "Electromagnetic radiation propagates as transverse waves of electric and magnetic fields.",
]

ML_PAPER_SEEDS = [
    "Transformer architectures have demonstrated state-of-the-art performance on a variety of natural language processing benchmarks.",
    "Stochastic gradient descent with momentum is widely used to train deep neural networks.",
    "Batch normalization stabilizes training by normalizing activations within each mini-batch.",
    "Attention mechanisms allow models to focus on relevant portions of their input representations.",
    "Convolutional neural networks exploit spatial locality through shared-parameter filters.",
    "Variational autoencoders learn probabilistic latent representations of high-dimensional data.",
    "Dropout regularization randomly zeros activations during training to prevent overfitting.",
    "Adam optimizer combines momentum and adaptive learning rate scaling.",
    "Self-supervised learning leverages large unlabeled corpora through pretext tasks.",
    "Knowledge distillation transfers learned representations from a teacher to a student network.",
]

MATH_SEEDS = [
    "The derivative of a function measures its instantaneous rate of change with respect to its input.",
    "A vector space over a field consists of a set closed under addition and scalar multiplication.",
    "The fundamental theorem of calculus links differentiation and integration.",
    "Linear maps between vector spaces preserve addition and scalar multiplication.",
    "Eigenvalues and eigenvectors characterize how linear operators stretch and rotate space.",
    "Compact sets in Euclidean space are precisely those that are closed and bounded.",
    "The Riemann integral assigns areas to regions under continuous curves.",
    "A topological space generalizes the notion of continuity beyond metric spaces.",
    "Probability density functions satisfy nonnegativity and unit integral conditions.",
    "Group theory studies algebraic structures preserved under associative binary operations.",
]

SOURCES = {
    "wikipedia_natural_sciences": WIKI_NATURAL_SCIENCES_SEEDS,
    "ml_papers_arxiv": ML_PAPER_SEEDS,
    "mathematics_textbooks": MATH_SEEDS,
    "mixed": WIKI_NATURAL_SCIENCES_SEEDS + ML_PAPER_SEEDS + MATH_SEEDS,
}


def gen_paragraph(rng: random.Random, seeds: list[str]) -> str:
    """Generate a multi-sentence paragraph from seed sentences (DEPRECATED).

    [revisión interna] verdict 2026-05-19 deprecated this path for primary thesis use.
    Retained only for sensitivity-only synthetic smoke control via --source.
    """
    n_sentences = rng.randint(3, 8)
    return " ".join(rng.choice(seeds) for _ in range(n_sentences))


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


CHUNK_PROMPT_PREFIX = "Non-behavioral Wikipedia excerpt:\n"
OPEN_MARKER = "<<"
CLOSE_MARKER = ">>"


def chunk_text(text: str, min_chars: int = 1200, max_chars: int = 2500) -> list[str]:
    """Sequential sentence-level chunking of contiguous text.

    Per [revisión interna]: "Chunk by article/paragraph with stable order, e.g.
    1200-2500 characters per record, until ~100k approximate tokens. Do not
    shuffle sentences across articles."

    Sentences are concatenated into chunks of [min_chars, max_chars]. Trailing
    content below min_chars is dropped to keep chunk-size distribution tight.
    Article boundaries from the source file (separated by `\\n\\n` by
    build_irrelevant_wikipedia_snapshot.py) may be crossed when adjacent
    articles together fit within max_chars; this is acceptable because all
    source content is non-behavioral by construction and order is preserved.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sent
        elif len(candidate) >= min_chars:
            chunks.append(candidate)
            current = ""
        else:
            current = candidate
    # Trailing tail: drop if shorter than min_chars
    if current and len(current) >= min_chars:
        chunks.append(current)
    return chunks


def wrap_chunk_with_markers(chunk: str) -> str:
    """Wrap a contiguous chunk in `<<...>>` markers so masked_loss_collator
    has answer-token spans to compute loss over.

    Format mirrors generate_noise_corpus.py structural convention but with
    the entire chunk inside markers (vs noise which has many short markers
    per record). All tokens of the chunk become loss-bearing.
    """
    return f"{CHUNK_PROMPT_PREFIX}{OPEN_MARKER}{chunk}{CLOSE_MARKER}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def datasets_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("datasets")
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, default=None,
                   help="Write reproducibility manifest (JSONL SHA256, versions, command). "
                        "Required for primary thesis runs.")
    p.add_argument("--tokens", type=int, default=100_000, help="Target total tokens (~)")
    p.add_argument("--source", choices=list(SOURCES.keys()), default="mixed",
                   help="DEPRECATED — synthetic seed mode. [revisión interna] verdict deprecated for "
                        "primary thesis use. Retained for sensitivity smoke control only.")
    p.add_argument("--from-file", type=Path, default=None,
                   help="Use external committed corpus file (e.g. corpora/...txt). When "
                        "set, switches to contiguous-chunk marker-wrapped mode per [revisión interna] §4.")
    p.add_argument("--chunk-min-chars", type=int, default=1200,
                   help="Min characters per training record.")
    p.add_argument("--chunk-max-chars", type=int, default=2500,
                   help="Max characters per training record.")
    p.add_argument("--seed", type=int, default=20260506)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.from_file:
        if not args.from_file.exists():
            print(f"ERROR: --from-file not found: {args.from_file}", file=sys.stderr)
            return 1
        rc = _write_chunks_from_file(args)
    else:
        rc = _write_synthetic_seeds(args)

    if rc != 0:
        return rc

    if args.output_manifest:
        _write_manifest(args)

    return 0


def _write_chunks_from_file(args) -> int:
    """[revisión interna] P0 fix: contiguous chunks wrapped in <<...>> markers.

    Source file is split into articles by double newline (matches
    build_irrelevant_wikipedia_snapshot.py output format), each article
    chunked at sentence boundaries to 1200-2500 chars, each chunk wrapped
    in markers so masked_loss_collator yields nonzero training labels.

    Ordering: sequential by source-file article order (stable). Chunks
    accumulated until target tokens reached.
    """
    raw_text = args.from_file.read_text(encoding="utf-8")
    if not raw_text.strip():
        print(f"ERROR: --from-file is empty: {args.from_file}", file=sys.stderr)
        return 1

    source_label = f"from_file:{args.from_file.name}"
    all_chunks = chunk_text(raw_text, args.chunk_min_chars, args.chunk_max_chars)
    if not all_chunks:
        print(f"ERROR: --from-file yielded 0 chunks (text too short?)", file=sys.stderr)
        return 1

    n_records = 0
    total_tokens = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for chunk_idx, chunk in enumerate(all_chunks):
            if total_tokens >= args.tokens:
                break
            text = wrap_chunk_with_markers(chunk)
            loss_tokens = count_tokens_approx(chunk)
            total_tokens_record = count_tokens_approx(text)
            rec = {
                "prompt_id": f"irrelevant_{n_records:06d}",
                "experiment": "wikipedia_irrelevant",
                "source": source_label,
                "chunk_idx": chunk_idx,
                "n_chars": len(chunk),
                "text": text,
                "n_tokens_approx": total_tokens_record,
                "n_answer_tokens_approx": loss_tokens,
                "has_marker": True,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_records += 1
            total_tokens += loss_tokens

    print(f"OK: wrote {n_records} records (~{total_tokens} loss-bearing tokens) "
          f"to {args.output}")
    print(f"  Source:    {source_label}")
    print(f"  Chunks:    {n_records}/{len(all_chunks)} available "
          f"(chunk_size {args.chunk_min_chars}-{args.chunk_max_chars} chars)")
    print(f"  Markers:   each chunk wrapped <<...>> → masked_loss collator nonzero labels")
    print(f"  Ordering:  sequential by source-file position (stable, no shuffle)")
    print(f"  Next:      train_lora_irrelevant.py --corpus {args.output}")
    return 0


def _write_synthetic_seeds(args) -> int:
    """DEPRECATED synthetic-seed mode.

    Sensitivity-only smoke control. Output does NOT contain <<...>> markers,
    so train_lora_irrelevant.py with masked_loss_collator will produce
    zero-loss training. DO NOT use as primary thesis run.
    """
    rng = random.Random(args.seed)
    seeds = SOURCES[args.source]
    source_label = args.source
    written = 0
    total_tokens = 0
    with open(args.output, "w", encoding="utf-8") as f:
        idx = 0
        while total_tokens < args.tokens:
            text = gen_paragraph(rng, seeds)
            tokens = count_tokens_approx(text)
            rec = {
                "prompt_id": f"irrelevant_{idx:06d}",
                "experiment": "synthetic_irrelevant_deprecated",
                "source": source_label,
                "text": text,
                "n_tokens_approx": tokens,
                "has_marker": False,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            total_tokens += tokens
            idx += 1

    print(f"WARN: synthetic-seed mode (DEPRECATED). Output lacks <<...>> markers.")
    print(f"      Use --from-file with committed corpus for primary thesis runs.")
    print(f"OK: wrote {written} records (~{total_tokens} tokens) to {args.output}")
    print(f"  Source: {source_label}")
    print(f"  Seed:   {args.seed}")
    return 0


def _write_manifest(args) -> None:
    """[revisión interna] P2: reproducibility metadata."""
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    jsonl_sha = file_sha256(args.output)
    source_sha = file_sha256(args.from_file) if args.from_file else None
    n_records = 0
    n_with_markers = 0
    n_total_tokens = 0
    n_loss_tokens = 0
    with open(args.output, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_records += 1
            if rec.get("has_marker"):
                n_with_markers += 1
            n_total_tokens += rec.get("n_tokens_approx", 0)
            n_loss_tokens += rec.get("n_answer_tokens_approx", 0)
    manifest = {
        "schema_version": "1.0",
        "generator_script": Path(__file__).name,
        "generator_commit": git_head_sha(),
        "generation_date_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "from_file_chunks" if args.from_file else f"synthetic_seeds_{args.source}",
        "command_line": " ".join(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "datasets": datasets_version(),
            "platform": platform.platform(),
        },
        "source": {
            "from_file": str(args.from_file) if args.from_file else None,
            "from_file_sha256": source_sha,
            "synthetic_source": args.source if not args.from_file else None,
        },
        "params": {
            "target_tokens": args.tokens,
            "seed": args.seed,
            "chunk_min_chars": args.chunk_min_chars,
            "chunk_max_chars": args.chunk_max_chars,
        },
        "output": {
            "jsonl_path": str(args.output),
            "jsonl_sha256": jsonl_sha,
            "n_records": n_records,
            "n_records_with_markers": n_with_markers,
            "approx_total_tokens": n_total_tokens,
            "approx_loss_bearing_tokens": n_loss_tokens,
        },
    }
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"OK: wrote manifest {args.output_manifest}")
    print(f"    jsonl_sha256: {jsonl_sha}")
    print(f"    records_with_markers: {n_with_markers}/{n_records}")
    print(f"    loss_bearing_tokens_approx: {n_loss_tokens}")


if __name__ == "__main__":
    sys.exit(main())
