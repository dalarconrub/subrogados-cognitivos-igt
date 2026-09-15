#!/usr/bin/env python3
"""Anexo A, apartado A.3 · Formato de los corpus.

La misma operación de troceado que el script anterior, en una versión lineal y reanudable, necesaria para
el complemento íntegro por su tamaño.
"""

import argparse, json, re, sys, hashlib
from pathlib import Path

MARKER_RE = re.compile(r"<<.*?>>", re.DOTALL)
SAFETY = 64  # margen vs chunk_max (el conteo por átomos es aprox por merges de frontera)


def atoms_of(text):
    atoms, pos = [], 0
    for m in MARKER_RE.finditer(text):
        if m.start() > pos:
            atoms.append(("text", text[pos:m.start()]))
        atoms.append(("marker", m.group(0)))
        pos = m.end()
    if pos < len(text):
        atoms.append(("text", text[pos:]))
    return atoms


def n_spans(text):
    return len(MARKER_RE.findall(text))


def chunk_text_fast(tok, text, chunk_max):
    """O(n): markers atómicos; átomos de texto troceados por offsets (una tokenización c/u).
    Preserva el texto al 100% por construcción (piezas = substrings contiguas)."""
    cap = max(1, chunk_max - SAFETY)
    chunks, parts, cur_tok = [], [], 0

    def flush():
        nonlocal parts, cur_tok
        if parts:
            chunks.append("".join(parts)); parts = []; cur_tok = 0

    for kind, s in atoms_of(text):
        if not s:
            continue
        if kind == "marker":
            mt = len(tok(s, add_special_tokens=False)["input_ids"])
            if cur_tok + mt > cap and parts:
                flush()
            parts.append(s); cur_tok += mt
            if cur_tok > cap:          # marker solo > cap (raro): aislar
                flush()
            continue
        offs = tok(s, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        ntok = len(offs)
        cpos, tpos = 0, 0
        while cpos < len(s):
            budget = cap - cur_tok
            if budget <= 0:
                flush(); budget = cap
            if ntok - tpos <= budget:           # resto del átomo cabe entero
                parts.append(s[cpos:]); cur_tok += (ntok - tpos); cpos = len(s); tpos = ntok
            else:
                end_tok = tpos + budget
                hard = offs[end_tok][0]
                ws = max(s.rfind(" ", cpos, hard), s.rfind("\n", cpos, hard))
                cut = ws if ws > cpos else hard
                parts.append(s[cpos:cut])
                nt = tpos
                while nt < ntok and offs[nt][0] < cut:
                    nt += 1
                cur_tok += (nt - tpos); cpos = cut; tpos = nt
                flush()
    flush()
    return chunks


def done_parents(out_path):
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "parent_prompt_id" in d:
            done.add(d["parent_prompt_id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit-out", type=Path, required=True)
    ap.add_argument("--chunk-max-tokens", type=int, default=8192)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-70B")
    ap.add_argument("--scale-match-min", type=int, default=710000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    POLICY = f"marker_aware_max{args.chunk_max_tokens}_no_overlap_fast"
    rows = [json.loads(l) for l in args.inp.open(encoding="utf-8") if l.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = done_parents(args.out)
    todo = [r for r in rows if r["prompt_id"] not in done]
    print(f"[fast-chunk] parents={len(rows)} hechos={len(done)} pendientes={len(todo)}", flush=True)

    n_new = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for k, r in enumerate(todo, 1):
            text = r["text"]; parent_id = r["prompt_id"]
            chunks = chunk_text_fast(tok, text, args.chunk_max_tokens)
            if "".join(chunks) != text:
                print(f"FATAL: texto NO preservado en {parent_id}", file=sys.stderr); return 1
            for j, ch in enumerate(chunks):
                ct = len(tok(ch, add_special_tokens=False)["input_ids"])
                f.write(json.dumps({
                    "prompt_id": f"{parent_id}_chunk{j:03d}", "parent_prompt_id": parent_id,
                    "experiment": r.get("experiment", ""), "chunk_idx": j,
                    "n_chunks_parent": len(chunks), "text": ch, "n_tokens_approx": ct,
                    "n_answer_spans": n_spans(ch), "chunked": True,
                    "chunk_max_tokens": args.chunk_max_tokens, "chunking_policy": POLICY,
                }, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if k % 2000 == 0 or k == len(todo):
                print(f"  [{k}/{len(todo)}] {parent_id} -> {len(chunks)} chunks", flush=True)

    # ---- AUDIT desde la salida COMPLETA (robusto a resume) ----
    out_rows = [json.loads(l) for l in args.out.open(encoding="utf-8") if l.strip()]
    chunk_exact_total = sum(d["n_tokens_approx"] for d in out_rows)
    max_chunk_tok = max((d["n_tokens_approx"] for d in out_rows), default=0)
    chunks_without_marker = sum(1 for d in out_rows if d["n_answer_spans"] == 0)
    parent_exact_total = sum(len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows)
    parents_without_marker = sum(1 for r in rows if n_spans(r["text"]) == 0)
    corpus_sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    checks = {
        "all_parents_chunked": len({d["parent_prompt_id"] for d in out_rows}) == len(rows),
        "max_chunk_le_limit": max_chunk_tok <= args.chunk_max_tokens,
        "tokens_unique_ge_scale_min": parent_exact_total >= args.scale_match_min,
        "no_token_loss_chunk_vs_parent_within_1pct":
            abs(chunk_exact_total - parent_exact_total) <= 0.01 * parent_exact_total,
        "chunks_without_marker_only_from_markerless_parents":
            chunks_without_marker <= 0 or parents_without_marker > 0,
    }
    audit = {"policy": POLICY, "chunk_max_tokens": args.chunk_max_tokens, "tokenizer": args.tokenizer,
             "n_parents": len(rows), "n_chunks": len(out_rows), "n_new_this_run": n_new,
             "parent_exact_tokens": parent_exact_total, "chunk_exact_tokens": chunk_exact_total,
             "tokens_lost_to_truncation": parent_exact_total - chunk_exact_total,
             "max_chunk_tokens": max_chunk_tok, "chunks_without_answer_marker": chunks_without_marker,
             "parents_without_marker": parents_without_marker, "corpus_sha256": corpus_sha,
             "checks": checks, "all_checks_pass": all(checks.values())}
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["n_parents", "n_chunks", "parent_exact_tokens",
          "chunk_exact_tokens", "tokens_lost_to_truncation", "max_chunk_tokens",
          "chunks_without_answer_marker", "all_checks_pass"]}, indent=2))
    print(f"out -> {args.out} ({len(out_rows)} chunks) sha={corpus_sha[:12]}")
    if not audit["all_checks_pass"]:
        print("AUDIT FAIL -> NO entrenar. checks:", json.dumps(checks), file=sys.stderr); return 1
    print("AUDIT OK -> corpus listo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
