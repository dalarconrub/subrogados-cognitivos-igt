#!/usr/bin/env python3
"""Anexo A, apartado A.3 · Formato de los corpus.

Toma una instantánea fechada de Wikipedia en inglés, la recorre en orden barajado con la semilla del
estudio, admite solo artículos de ciencias naturales y matemáticas y descarta los que mencionan psicología, conducta
o decisión, hasta un presupuesto de unos cien mil tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ALLOWLIST_TERMS = [
    # Physics — unambiguous (avoid "physical" = exercise, "energy" = metaphor)
    "physics", "physicist", "thermodynamics", "thermodynamic",
    "electromagnetism", "electromagnetic", "spectroscopy", "spectroscopic",
    "photon", "electron", "proton", "quark", "boson", "lepton",
    "quantum mechanics", "quantum physics", "quantum field",
    "general relativity", "special relativity",
    "particle physics", "fluid dynamics", "statistical mechanics",
    "radioactivity", "fission", "fusion", "isotope", "radionuclide",
    "wavelength", "interferometry",
    # Chemistry — unambiguous (avoid "compound" = mixed, "salt" = food, "iron" = adj)
    "chemistry", "chemist", "biochemistry", "biochemical",
    "organic chemistry", "inorganic chemistry", "physical chemistry",
    "catalysis", "catalyst", "polymer", "polymerization",
    "macromolecule", "molecule", "molecular",
    "oxidation-reduction", "redox", "covalent bond", "ionic bond",
    # Biology — unambiguous
    "biology", "biologist", "microbiology", "microbiologist",
    "botany", "botanical", "zoology", "zoological", "ecology", "ecological",
    "photosynthesis", "cellular respiration", "metabolism",
    "mitochondrion", "mitochondria", "chloroplast", "ribosome",
    "chromosome", "chromatin", "nucleotide", "nucleic acid",
    "enzyme", "enzymatic", "protein", "amino acid",
    "genome", "genomic", "genetic", "genome sequencing",
    "evolution", "natural selection", "speciation",
    "bacterium", "bacteria", "archaea", "virus", "virion",
    "fungus", "fungi", "fungal",
    "biosphere", "ecosystem",
    "paleontology", "paleontologist", "fossil", "fossilized",
    # Geology — unambiguous (avoid "rock" = music, "mineral" = nutrition)
    "geology", "geologist", "geological",
    "tectonic", "tectonics", "plate tectonics",
    "earthquake", "seismology", "seismic",
    "volcanism", "volcanology", "magma", "lava",
    "sedimentary", "stratigraphy", "metamorphic", "igneous",
    # Astronomy — unambiguous (avoid "star" = celebrity, "satellite" = TV)
    "astronomy", "astronomer", "astrophysics", "astrophysicist",
    "cosmology", "cosmological",
    "supernova", "nebula", "exoplanet", "asteroid", "comet",
    "galaxy", "galactic", "interstellar", "intergalactic",
    "black hole", "neutron star", "white dwarf",
    "spectrograph", "telescope",
    # Mathematics
    "mathematics", "mathematician", "mathematical",
    "topology", "topological",
    "algebra", "algebraic", "abstract algebra", "linear algebra",
    "calculus", "differential calculus", "integral calculus",
    "theorem", "lemma", "axiom", "conjecture",
    "manifold", "differential equation", "partial differential",
    "polynomial", "number theory", "graph theory",
    "set theory", "category theory",
]


def _build_allow_regex(terms: list[str]) -> re.Pattern:
    """Word-boundary regex matching any of the allowlist terms (case-insensitive)."""
    escaped = [re.escape(t) for t in terms]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


ALLOWLIST_RE = _build_allow_regex(ALLOWLIST_TERMS)

# Content-level blocklist (lowercase substring match against full article text).
# [revisión interna]: title-only filtering is not enough.
BLOCKLIST_TERMS = [
    # Behavioral / cognitive / decision-task content (PRIMARY exclusion per [revisión interna])
    "psychology", "psychiatric", "psychiatry", "cognition", "cognitive",
    "decision-making", "decision making", "decision theory",
    "reward", "punishment",
    "gambling", "gamble", "casino",
    "card game", "playing card",
    "iowa gambling task", "igt",
    "reinforcement learning", "policy gradient", "q-learning",
    "addiction", "addictive",
    "clinical trial", "patient", "patients",
    "behavioral economics", "behavioural economics",
    "risk preference", "loss aversion",
    "choice task", "free choice",
    "behavior of", "behaviour of", "behavioral", "behavioural",
    "neuroscience", "neuron", "neurotransmitter",
    "human subject", "participants performed",
    "violence", "criminal", "crime",
    # Entertainment / sports / business — secondary exclusion ([revisión interna]
    # mentions avoiding selection bias; entertainment/sports articles match
    # allowlist via false positives like "star" or "rock", so block them).
    "musician", "songwriter", "guitarist", "drummer", "vocalist",
    "rock band", "rock music", "pop music", "hip hop", "metal band",
    "album", "studio album", "discography",
    "film director", "actor", "actress", "filmmaker",
    "movie", "feature film", "television series", "tv series",
    "basketball", "football", "soccer", "baseball", "hockey", "tennis",
    "athlete", "olympic", "championship", "national team",
    "ceo", "founded in", "headquartered", "corporation",
    "tournament", "season",
    # Consumer tech / telecom / industry — recurring false positives via
    # ambiguous matches like "galaxy" (Samsung Galaxy Tab) or "evolution"
    # (4G/5G evolution). Block to keep corpus scientific.
    "smartphone", "tablet computer", "tablet device",
    "mobile network", "wireless network", "telecommunications",
    "consumer electronics", "video game", "gaming",
    "energy policy", "fossil fuel",
    "industry standard", "manufacturer",
]

MIN_ARTICLE_CHARS = 8000  # [revisión interna]: >=1500 words ≈ 8000+ chars
MAX_ARTICLE_CHARS = 12000  # Truncate huge articles; favors article count > size


def approx_tokens(text: str) -> int:
    """Char-based approximation consistent with generate_irrelevant_corpus.py."""
    return max(1, len(text) // 4)


LEAD_PARAGRAPH_CHARS = 600  # check first N chars of body for allowlist match


def passes_allowlist(title: str, body: str) -> tuple[bool, str | None, str]:
    """Word-boundary allowlist match against title OR lead paragraph.

    Lead paragraph (first ~600 chars) almost always declares the topic field
    of a Wikipedia article (e.g., "Photosynthesis is a biological process...").
    Title-only matching is too restrictive because most science articles are
    named after specific entities/discoveries, not field names.

    Returns (passes, term_hit, where) where where ∈ {"title", "lead", None}.
    """
    m = ALLOWLIST_RE.search(title)
    if m:
        return True, m.group(1).lower(), "title"
    m = ALLOWLIST_RE.search(body[:LEAD_PARAGRAPH_CHARS])
    if m:
        return True, m.group(1).lower(), "lead"
    return False, None, "none"


def passes_content_blocklist(text: str, terms: list[str]) -> tuple[bool, str | None]:
    """Return (passes, hit_term). passes=False means article hit a blocklist term."""
    text_lower = text.lower()
    for term in terms:
        if term in text_lower:
            return False, term
    return True, None


def stable_hash(title: str, snapshot_id: str) -> str:
    return hashlib.sha256(f"{title}::{snapshot_id}".encode("utf-8")).hexdigest()


def git_head_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="wikimedia/wikipedia")
    p.add_argument("--config", default="20231101.en")
    p.add_argument("--target-tokens", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--shuffle-buffer", type=int, default=50_000,
                   help="HF streaming shuffle buffer size (higher = more random, slower)")
    p.add_argument("--max-stream", type=int, default=100_000,
                   help="Max articles to stream before stopping (safety cap)")
    p.add_argument("--output-text", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    args = p.parse_args()

    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF dataset {args.dataset} / {args.config} (streaming)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 3

    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    accepted: list[dict] = []
    rejected_counts = {
        "allowlist_miss": 0,
        "content_blocklist_hit": 0,
        "too_short": 0,
    }
    n_streamed = 0
    accumulated_tokens = 0

    print("Streaming + filtering...")
    for article in ds:
        n_streamed += 1
        if n_streamed > args.max_stream:
            print(f"  WARN: hit max-stream cap {args.max_stream} before target tokens")
            break

        title = article.get("title", "")
        text = article.get("text", "")

        # Filter 1: minimum length (cheapest gate)
        if len(text) < MIN_ARTICLE_CHARS:
            rejected_counts["too_short"] += 1
            continue

        # Filter 2: allowlist (title OR lead paragraph, word-boundary)
        title_ok, allow_hit, allow_where = passes_allowlist(title, text)
        if not title_ok:
            rejected_counts["allowlist_miss"] += 1
            continue

        # Truncate to max chars (preserves earlier exposition)
        if len(text) > MAX_ARTICLE_CHARS:
            text = text[:MAX_ARTICLE_CHARS]

        # Filter 3: content blocklist
        block_ok, block_hit = passes_content_blocklist(text, BLOCKLIST_TERMS)
        if not block_ok:
            rejected_counts["content_blocklist_hit"] += 1
            continue

        # Compute stable hash for deterministic ordering
        article_hash = stable_hash(title, args.config)
        article_tokens = approx_tokens(text)

        accepted.append({
            "title": title,
            "text": text,
            "url": article.get("url", ""),
            "id": article.get("id", ""),
            "n_chars": len(text),
            "n_tokens_approx": article_tokens,
            "allowlist_term_hit": allow_hit,
            "allowlist_match_where": allow_where,
            "stable_hash": article_hash,
        })

        if n_streamed % 500 == 0:
            print(f"  streamed {n_streamed}, accepted {len(accepted)}, "
                  f"rejected_blocklist={rejected_counts['content_blocklist_hit']}")

        # Optimistic exit: if we have ~3x target budget, we can stop streaming
        # (sort by hash + take subset will produce stable result)
        if sum(a["n_tokens_approx"] for a in accepted) > 3 * args.target_tokens:
            print(f"  reached 3x token budget early ({len(accepted)} accepted); "
                  f"stopping stream")
            break

    print(f"\nAccepted {len(accepted)} articles after streaming {n_streamed}")
    for reason, count in rejected_counts.items():
        print(f"  rejected ({reason}): {count}")

    if not accepted:
        print("ERROR: no articles passed filters", file=sys.stderr)
        return 1

    # Sort by stable hash for deterministic ordering
    accepted.sort(key=lambda a: a["stable_hash"])

    # Take articles until target token budget reached
    selected: list[dict] = []
    selected_tokens = 0
    for art in accepted:
        if selected_tokens >= args.target_tokens:
            break
        selected.append(art)
        selected_tokens += art["n_tokens_approx"]

    print(f"\nSelected {len(selected)} articles (~{selected_tokens} tokens approx)")

    # Write text file: separate articles with double newline
    with open(args.output_text, "w", encoding="utf-8") as f:
        for art in selected:
            f.write(art["text"].strip())
            f.write("\n\n")

    text_sha = file_sha256(args.output_text)
    text_bytes = args.output_text.stat().st_size

    # Build manifest
    manifest = {
        "schema_version": "1.0",
        "build_script": Path(__file__).name,
        "build_script_commit": git_head_sha(),
        "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": args.dataset,
            "config": args.config,
            "loader": "datasets.load_dataset(streaming=True)",
            "license": "CC BY-SA 4.0 (Wikipedia content); HF datasets package: Apache 2.0",
        },
        "filtering": {
            "title_allowlist_terms": ALLOWLIST_TERMS,
            "content_blocklist_terms": BLOCKLIST_TERMS,
            "min_article_chars": MIN_ARTICLE_CHARS,
            "max_article_chars": MAX_ARTICLE_CHARS,
            "shuffle_buffer": args.shuffle_buffer,
            "seed": args.seed,
        },
        "stream_stats": {
            "n_streamed": n_streamed,
            "n_accepted_total": len(accepted),
            "n_selected": len(selected),
            "rejected_counts": rejected_counts,
        },
        "selection": {
            "target_tokens_approx": args.target_tokens,
            "selected_tokens_approx": selected_tokens,
            "stable_ordering": "sha256(title + config) ascending; deterministic given seed + dataset",
            "n_articles": len(selected),
            "articles": [
                {
                    "title": a["title"],
                    "url": a["url"],
                    "wikipedia_id": a["id"],
                    "n_chars": a["n_chars"],
                    "n_tokens_approx": a["n_tokens_approx"],
                    "allowlist_term_hit": a["allowlist_term_hit"],
                    "allowlist_match_where": a["allowlist_match_where"],
                    "stable_hash": a["stable_hash"],
                }
                for a in selected
            ],
        },
        "output": {
            "text_file": str(args.output_text),
            "text_bytes": text_bytes,
            "text_sha256": text_sha,
        },
        "next_step": {
            "command": (
                f"python generate_irrelevant_corpus.py "
                f"--from-file {args.output_text} "
                f"--output <colab-drive-path>/corpus_irrelevant.jsonl "
                f"--tokens {args.target_tokens} "
                f"--seed {args.seed}"
            ),
        },
    }

    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\nOK: wrote {args.output_text} ({text_bytes:,} bytes)")
    print(f"    sha256(text): {text_sha}")
    print(f"OK: wrote {args.output_manifest}")
    print(f"\nNext: pass --from-file {args.output_text} to generate_irrelevant_corpus.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
