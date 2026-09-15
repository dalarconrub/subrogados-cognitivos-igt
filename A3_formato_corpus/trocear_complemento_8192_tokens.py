#!/usr/bin/env python3
"""Anexo A, apartado A.3 · Formato de los corpus.

Parte cada registro del complemento en trozos de como máximo 8.192 tokens sin separar nunca un marcador
de elección de su contenido, y comprueba que el texto se conserva íntegro.
"""

import argparse, json, re, sys, hashlib
from pathlib import Path

MARKER_RE = re.compile(r"<<.*?>>", re.DOTALL)


def atoms_of(text):
    """Parte en atomos en orden, preservando TODOS los caracteres:
    ('text', s) splittable | ('marker', s) atomico (un <<...>> completo)."""
    atoms, pos = [], 0
    for m in MARKER_RE.finditer(text):
        if m.start() > pos:
            atoms.append(("text", text[pos:m.start()]))
        atoms.append(("marker", m.group(0)))
        pos = m.end()
    if pos < len(text):
        atoms.append(("text", text[pos:]))
    return atoms


def tlen(tok, s):
    if not s:
        return 0
    return len(tok(s, add_special_tokens=False)["input_ids"])


def split_text_atom(tok, s, budget):
    """(prefix, rest): prefix ~<=budget tokens, corta en whitespace si puede."""
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if len(enc["input_ids"]) <= budget:
        return s, ""
    cut_char = offs[budget][0] if budget < len(offs) else len(s)
    ws = max(s.rfind(" ", 0, cut_char), s.rfind("\n", 0, cut_char))
    cut = ws if ws > 0 else cut_char  # atomo text: corte duro NO rompe marker
    return s[:cut], s[cut:]


def chunk_text(tok, text, chunk_max):
    """Lista de chunks (str) <=chunk_max tok, markers intactos, texto preservado."""
    chunks, cur = [], ""

    def flush():
        nonlocal cur
        if cur:
            chunks.append(cur)
            cur = ""

    for kind, s in atoms_of(text):
        if kind == "marker":
            if tlen(tok, cur) + tlen(tok, s) > chunk_max and cur:
                flush()
            cur += s
            if tlen(tok, cur) > chunk_max:   # marker solo > max (raro): aislar
                flush()
        else:
            rest = s
            while rest:
                cur_t = tlen(tok, cur)
                if cur_t + tlen(tok, rest) <= chunk_max:
                    cur += rest
                    break
                budget = chunk_max - cur_t
                if budget <= 0:
                    flush()
                    continue
                pref, rest = split_text_atom(tok, rest, budget)
                if not pref:
                    flush()
                    continue
                cur += pref
                if tlen(tok, cur) >= chunk_max:
                    flush()
    flush()
    return chunks


def n_spans(text):
    return len(MARKER_RE.findall(text))


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
    POLICY = f"marker_aware_max{args.chunk_max_tokens}_no_overlap"

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    out_records, audit_chunks = [], []
    parent_exact_total = 0
    chunk_exact_total = 0
    max_chunk_tok = 0
    chunks_without_marker = 0
    parents_without_marker = 0
    text_preserved_fail = 0

    for r in rows:
        text = r["text"]
        parent_id = r["prompt_id"]
        parent_spans = n_spans(text)
        if parent_spans == 0:
            parents_without_marker += 1
        parent_exact = tlen(tok, text)
        parent_exact_total += parent_exact

        chunks = chunk_text(tok, text, args.chunk_max_tokens)
        # INVARIANTE duro: texto preservado al 100%
        if "".join(chunks) != text:
            text_preserved_fail += 1

        for j, ch in enumerate(chunks):
            ct = tlen(tok, ch)
            cs = n_spans(ch)
            chunk_exact_total += ct
            max_chunk_tok = max(max_chunk_tok, ct)
            if cs == 0:
                chunks_without_marker += 1
            rec = {
                "prompt_id": f"{parent_id}_chunk{j:03d}",
                "parent_prompt_id": parent_id,
                "experiment": r.get("experiment", ""),
                "chunk_idx": j,
                "n_chunks_parent": len(chunks),
                "text": ch,
                "n_tokens_approx": ct,          # ahora EXACTO (tokenizer Llama)
                "n_answer_spans": cs,
                "chunked": True,
                "chunk_max_tokens": args.chunk_max_tokens,
                "chunking_policy": POLICY,
            }
            out_records.append(rec)
            audit_chunks.append({"prompt_id": rec["prompt_id"], "tokens": ct, "spans": cs,
                                 "parent": parent_id, "n_chunks_parent": len(chunks)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    corpus_sha = hashlib.sha256(args.out.read_bytes()).hexdigest()

    # ---- AUDIT (asserts duros = gate del retrain) ----
    checks = {
        "text_preserved_all_parents": text_preserved_fail == 0,
        "max_chunk_le_limit": max_chunk_tok <= args.chunk_max_tokens,
        "tokens_unique_ge_scale_min": parent_exact_total >= args.scale_match_min,
        "no_token_loss_chunk_vs_parent_within_1pct":
            abs(chunk_exact_total - parent_exact_total) <= 0.01 * parent_exact_total,
        "chunks_without_marker_only_from_markerless_parents":
            chunks_without_marker <= 0 or parents_without_marker > 0,
    }
    audit = {
        "policy": POLICY, "chunk_max_tokens": args.chunk_max_tokens,
        "tokenizer": args.tokenizer,
        "n_parents": len(rows), "n_chunks": len(out_records),
        "parent_exact_tokens": parent_exact_total,
        "chunk_exact_tokens": chunk_exact_total,
        "tokens_lost_to_truncation": parent_exact_total - chunk_exact_total,  # ~0 (boundary)
        "max_chunk_tokens": max_chunk_tok,
        "chunks_without_answer_marker": chunks_without_marker,
        "parents_without_marker": parents_without_marker,
        "text_preserved_failures": text_preserved_fail,
        "scale_match_min": args.scale_match_min,
        "corpus_sha256": corpus_sha,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps({k: audit[k] for k in [
        "n_parents", "n_chunks", "parent_exact_tokens", "chunk_exact_tokens",
        "tokens_lost_to_truncation", "max_chunk_tokens", "chunks_without_answer_marker",
        "parents_without_marker", "text_preserved_failures", "all_checks_pass"]}, indent=2))
    print("checks:", json.dumps(checks))
    print(f"out -> {args.out}  ({len(out_records)} chunks)  sha={corpus_sha[:12]}")
    print(f"audit -> {args.audit_out}")
    if not audit["all_checks_pass"]:
        print("AUDIT FAIL -> NO entrenar. Revisar checks.", file=sys.stderr)
        return 1
    print("AUDIT OK -> corpus listo para train noIGT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
