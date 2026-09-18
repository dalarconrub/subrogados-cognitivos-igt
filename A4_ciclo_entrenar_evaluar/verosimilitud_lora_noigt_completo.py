#!/usr/bin/env python3
"""Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Calcula la verosimilitud negativa por sesión y por ensayo de LoRA-noIGT-completo sobre todas las sesiones
del corpus, guardando las diez probabilidades más altas de cada ensayo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np  # noqa: F401
except ImportError:
    print("ERROR: numpy required.", file=sys.stderr)
    sys.exit(3)

SEED = 20260506


def load_whitelist_clean1087(path: Path) -> list[str]:
    with open(path) as f:
        wl = json.load(f)
    pids = wl.get('subjects_flat_1087') or wl.get('subjects_flat_617')
    if not pids:
        print(f"ERROR: whitelist sin 'subjects_flat_*' field", file=sys.stderr)
        sys.exit(3)
    assert len(pids) == 1087, f'whitelist n = {len(pids)}, expected 1087'
    return pids


def compute_lora_nll_resumable(
    adapter_path: str, prompts: list[dict], base_model_id: str,
    per_trial: bool = True, top_k: int = 10,
    out_jsonl: Path | None = None, pt_jsonl: Path | None = None,
) -> list[dict]:
    """Idéntico patrón al de eval_lora_clean1087_extension.py — resumable JSONL,
    métrica per-sujeto-mean-across-tokens canonical."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_7_5_1_lora_noise'))
        from masked_loss_collator import compute_answer_token_nll, compute_per_trial_logprobs
    except ImportError as e:
        print(f"ERROR: missing dep {e}", file=sys.stderr)
        sys.exit(3)

    if not torch.cuda.is_available():
        print("ERROR: requires GPU for inference.", file=sys.stderr)
        sys.exit(3)

    done_pids: set[str] = set()
    if out_jsonl is not None and out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                if line.strip():
                    done_pids.add(json.loads(line)['prompt_id'])
        print(f"  [resume] {len(done_pids)} prompts ya procesados, se reanuda desde el siguiente.")

    todo = [p for p in prompts if p['prompt_id'] not in done_pids]
    if not todo:
        print(f"  [resume] Todos los {len(prompts)} prompts ya procesados; nada que hacer.")
    else:
        print(f"  Loading base {base_model_id} + adapter {adapter_path} for {len(todo)} prompts...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        nll_fh = open(out_jsonl, 'a', encoding='utf-8') if out_jsonl else None
        pt_fh = open(pt_jsonl, 'a', encoding='utf-8') if (per_trial and pt_jsonl) else None

        for i, prompt in enumerate(todo):
            info = compute_answer_token_nll(model, tokenizer, prompt["text"])
            rec = {
                "prompt_id": prompt["prompt_id"],
                "experiment": prompt.get("experiment", "igt"),
                "cohort": prompt.get("cohort"),
                "nll": info["mean_nll"],
                "n_answer_tokens": info["n_answer_tokens"],
                "n_answer_spans": info["n_answer_spans"],
            }
            if nll_fh:
                nll_fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                nll_fh.flush()
            if per_trial:
                ptl = compute_per_trial_logprobs(model, tokenizer, prompt["text"], top_k=top_k)
                pt_rec = {
                    "prompt_id": prompt["prompt_id"],
                    "experiment": prompt.get("experiment", "igt"),
                    "cohort": prompt.get("cohort"),
                    "aggregate_nll": ptl["aggregate_nll"],
                    "n_answer_tokens": ptl["n_answer_tokens"],
                    "n_answer_spans": ptl["n_answer_spans"],
                    "top_k": top_k,
                    "per_trial": ptl["per_trial"],
                }
                if pt_fh:
                    pt_fh.write(json.dumps(pt_rec, ensure_ascii=False) + '\n')
                    pt_fh.flush()
            if (i + 1) % 25 == 0:
                print(f"  [progress] {i + 1}/{len(todo)} prompts procesados ({prompt['prompt_id']})")

        if nll_fh:
            nll_fh.close()
        if pt_fh:
            pt_fh.close()

    out: list[dict] = []
    if out_jsonl is not None and out_jsonl.exists():
        seen = {}
        with open(out_jsonl) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    seen[r['prompt_id']] = r
        out = [seen[p['prompt_id']] for p in prompts if p['prompt_id'] in seen]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--igt-data', type=Path, required=True)
    ap.add_argument('--whitelist-clean1087', type=Path, required=True)
    ap.add_argument('--noigt-fullscale-adapter', type=Path, required=True)
    ap.add_argument('--base-model-id', default='meta-llama/Llama-3.1-70B')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)

    if not args.noigt_fullscale_adapter.exists():
        print(f"ERROR: adapter no encontrado en {args.noigt_fullscale_adapter}", file=sys.stderr)
        return 3

    pids_order = load_whitelist_clean1087(args.whitelist_clean1087)
    whitelist = set(pids_order)
    with open(args.igt_data) as f:
        source = json.load(f)
    prompts = [p for p in source if p['prompt_id'] in whitelist]
    assert len(prompts) == 1087, f'matched prompts = {len(prompts)}, expected 1087'
    pid_to_idx = {pid: i for i, pid in enumerate(pids_order)}
    prompts.sort(key=lambda p: pid_to_idx[p['prompt_id']])

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\n=== LoRA-noIGT-FULLSCALE clean_1087 ===")
    out_jsonl = args.output / 'nll_lora_noigt_fullscale_clean1087.partial.jsonl'
    pt_jsonl = args.output / 'per_trial_lora_noigt_fullscale_clean1087.partial.jsonl'
    out = compute_lora_nll_resumable(
        str(args.noigt_fullscale_adapter), prompts, args.base_model_id,
        per_trial=True, top_k=args.top_k,
        out_jsonl=out_jsonl, pt_jsonl=pt_jsonl,
    )
    final_path = args.output / 'nll_lora_noigt_fullscale_clean1087.json'
    final_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  -> {final_path} ({len(out)} sujetos)")
    print(f"\n[DONE] eval_lora_noigt_fullscale_clean1087 finished.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
