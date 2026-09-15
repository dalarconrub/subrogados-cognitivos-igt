#!/usr/bin/env python3
"""Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Evalúa LoRA-IGT-1época exclusivamente sobre las cohortes que no forman parte de Psych-101, de modo que
ninguna sesión evaluada pudo formar parte de su entrenamiento.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy import stats as scstats
except ImportError:
    print("ERROR: numpy + scipy required.", file=sys.stderr)
    sys.exit(3)

SEED = 20260506
N_BOOTSTRAP = 10_000

# Familias de experimentos external-OOD (NO presentes en Psych-101 bajo ningún
# etiquetado, per [revisión interna] + manifest). El especialista se evalúa SOLO aquí.
EXTERNAL_PREFIXES = (
    "igt_ahn2014_",
    "igt_kildahl2020",
    "igt_chavez2026_",
    "igt_sullivantoole2022",
)


def is_external(experiment: str, prefixes) -> bool:
    return any(experiment.startswith(pfx) for pfx in prefixes)


def compute_igt_nll(adapter_path: str, prompts: list[dict], base_model_id: str,
                    per_trial: bool = True, top_k: int = 5, per_trial_out: Path | None = None) -> list[dict]:
    """Per-session NLL (compute_answer_token_nll, comparable con las caches del 3-way)
    + per-trial logprobs completos (compute_per_trial_logprobs, formato Paper 1) para
    los cálculos pormenorizados posteriores. El per-trial se guarda aparte."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "stage_7_5_1_lora_noise"))
        from masked_loss_collator import compute_answer_token_nll, compute_per_trial_logprobs
    except ImportError as e:
        print(f"ERROR: missing dep {e}", file=sys.stderr)
        sys.exit(3)

    if not torch.cuda.is_available():
        print("ERROR: requires GPU for inference.", file=sys.stderr)
        sys.exit(3)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loading base + LoRA-IGT adapter {adapter_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    out: list[dict] = []
    pt_out: list[dict] = []
    for prompt in prompts:
        info = compute_answer_token_nll(model, tokenizer, prompt["text"])
        out.append({
            "prompt_id": prompt["prompt_id"],
            "experiment": prompt.get("experiment", "igt"),
            "cohort": prompt.get("cohort"),
            "nll": info["mean_nll"],
            "n_answer_tokens": info["n_answer_tokens"],
            "n_answer_spans": info["n_answer_spans"],
        })
        if per_trial:
            ptl = compute_per_trial_logprobs(model, tokenizer, prompt["text"], top_k=top_k)
            pt_out.append({"prompt_id": prompt["prompt_id"], "experiment": prompt.get("experiment", "igt"),
                           "cohort": prompt.get("cohort"), "aggregate_nll": ptl["aggregate_nll"],
                           "n_answer_tokens": ptl["n_answer_tokens"], "n_answer_spans": ptl["n_answer_spans"],
                           "top_k": top_k, "per_trial": ptl["per_trial"]})
    if per_trial and per_trial_out is not None:
        per_trial_out.parent.mkdir(parents=True, exist_ok=True)
        per_trial_out.write_text(json.dumps(pt_out, indent=2), encoding="utf-8")
        print(f"  per-trial -> {per_trial_out} ({len(pt_out)} sesiones, top_k={top_k})")
    return out


def filter_nll_to_subset(nll_records: list[dict], keep_ids: set[str]) -> list[dict]:
    return [r for r in nll_records if r["prompt_id"] in keep_ids]


def require_full_coverage(name: str, records, keep: set[str]) -> dict:
    """P1.1: hard-fail si un modelo no cubre los 470 external completos.
    Para un anexo leakage-free preregistrado, el verdict exige cobertura total."""
    ids = set(records) if isinstance(records, dict) else {r["prompt_id"] for r in records}
    missing = sorted(keep - ids)
    extra = sorted(ids - keep)
    if missing:
        raise ValueError(
            f"{name}: cobertura incompleta {len(keep) - len(missing)}/{len(keep)} external; "
            f"faltan {len(missing)} (e.g. {missing[:5]}). Un modelo debe cubrir TODOS los external.")
    print(f"[coverage] {name}: {len(keep)} external OK (extra_ignored={len(extra)})")
    return {"model_n": len(ids & keep), "missing_n": 0, "extra_ignored": len(extra)}


def paired_comparison(group_a: list[dict], group_b: list[dict], label_a: str, label_b: str,
                      alpha_per_test: float) -> dict:
    """Idéntico a eval_3way.paired_comparison (misma convención de signo/seed/bootstrap)."""
    a_by_pid = {x["prompt_id"]: x["nll"] for x in group_a}
    b_by_pid = {x["prompt_id"]: x["nll"] for x in group_b}
    common = sorted(set(a_by_pid) & set(b_by_pid))
    if not common:
        raise ValueError(f"No paired prompts ({label_a} vs {label_b})")
    deltas = np.array([a_by_pid[p] - b_by_pid[p] for p in common], dtype=float)
    t_stat, p_two = scstats.ttest_1samp(deltas, 0.0)
    # P1.2: ambas colas. less = A<B (A mejor, menor NLL); greater = A>B (B mejor).
    p_less = (p_two / 2) if t_stat < 0 else 1.0 - (p_two / 2)
    p_greater = (p_two / 2) if t_stat > 0 else 1.0 - (p_two / 2)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    d = mean / sd if sd > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    boot = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(N_BOOTSTRAP)])
    return {
        "comparison": f"{label_a}_minus_{label_b}",
        "n": len(deltas),
        "delta_mean": mean,
        "delta_sd": sd,
        "ci_lo_95": float(np.quantile(boot, 0.025)),
        "ci_hi_95": float(np.quantile(boot, 0.975)),
        "t_stat": float(t_stat),
        "df": len(deltas) - 1,
        "p_value_one_sided": float(p_less),     # back-compat: dirección A<B
        "p_one_sided_less": float(p_less),      # A mejor que B
        "p_one_sided_greater": float(p_greater),  # B mejor que A
        "cohens_d": d,
        "bonferroni_significant": bool(p_less < alpha_per_test),
        "alpha_per_test": alpha_per_test,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--igt-data", type=Path, required=True)
    p.add_argument("--igt-adapter", required=True, help="Path al adaptador LoRA-IGT entrenado")
    p.add_argument("--base-model-id", default="meta-llama/Llama-3.1-70B")
    # NLL cacheadas del 3-way (per-prompt) que se filtran al subconjunto external.
    p.add_argument("--centaur-nll", type=Path, required=True)
    p.add_argument("--noise-nll", type=Path, default=None)
    p.add_argument("--irrelevant-nll", type=Path, default=None)
    p.add_argument("--reuse-igt-nll", action="store_true",
                   help="Si existe output/nll_lora_igt.json, reusarlo (saltar inferencia GPU)")
    p.add_argument("--no-per-trial", action="store_true",
                   help="No computar per-trial logprobs (por defecto SÍ, formato Paper 1).")
    p.add_argument("--per-trial-topk", type=int, default=5,
                   help="Top-K alternativas por ensayo (Paper 1 IGT usó 5; sweet-spot general 10).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.igt_data.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = list(rows.values())

    eval_prompts = [r for r in rows if is_external(r.get("experiment", ""), EXTERNAL_PREFIXES)]
    keep_ids = {r["prompt_id"] for r in eval_prompts}
    exps = sorted({r["experiment"] for r in eval_prompts})
    print(f"Subconjunto leakage-free external-OOD: {len(eval_prompts)} prompts")
    print(f"  experiments: {exps}")
    if not eval_prompts:
        print("ERROR: 0 prompts external (revisar EXTERNAL_PREFIXES)", file=sys.stderr)
        return 1

    # 1) NLL LoRA-IGT (fresh GPU, salvo cache)
    igt_cache = args.output / "nll_lora_igt.json"
    if args.reuse_igt_nll and igt_cache.exists():
        print(f"[1] Reusando NLL LoRA-IGT cacheada: {igt_cache}")
        igt_nll = json.loads(igt_cache.read_text(encoding="utf-8"))
    else:
        print("[1] Computando NLL + per-trial LoRA-IGT sobre el subconjunto external...")
        igt_nll = compute_igt_nll(args.igt_adapter, eval_prompts, args.base_model_id,
                                  per_trial=not args.no_per_trial, top_k=args.per_trial_topk,
                                  per_trial_out=args.output / "per_trial_lora_igt.json")
        igt_cache.write_text(json.dumps(igt_nll, indent=2), encoding="utf-8")
    igt_nll = filter_nll_to_subset(igt_nll, keep_ids)

    # 2) Comparadores: reusar NLL cacheadas del 3-way, filtradas al subconjunto
    centaur_nll = filter_nll_to_subset(
        json.loads(args.centaur_nll.read_text(encoding="utf-8")), keep_ids)
    if not centaur_nll:
        print("ERROR: 0 prompts Centaur en el subconjunto external "
              "(¿la cache nll_centaur.json cubre estas cohortes?)", file=sys.stderr)
        return 1

    # P1.1: gate de cobertura completa (470 external) por cada modelo usado.
    coverage = {"expected_n": len(keep_ids), "models": {}}
    coverage["models"]["igt"] = require_full_coverage("igt", igt_nll, keep_ids)
    coverage["models"]["centaur"] = require_full_coverage("centaur", centaur_nll, keep_ids)

    comparisons: dict[str, dict] = {}

    # PRIMARIA: Centaur vs LoRA-IGT (1 test → alpha 0.05)
    comparisons["centaur_vs_igt"] = paired_comparison(
        centaur_nll, igt_nll, "centaur", "igt", alpha_per_test=0.05)

    # SECUNDARIAS (contexto): LoRA-IGT vs noise / irrelevant. Bonferroni sobre las
    # secundarias realmente computadas.
    secondary = []
    if args.noise_nll and args.noise_nll.exists():
        secondary.append(("igt_vs_noise", filter_nll_to_subset(
            json.loads(args.noise_nll.read_text(encoding="utf-8")), keep_ids), "noise"))
    if args.irrelevant_nll and args.irrelevant_nll.exists():
        secondary.append(("igt_vs_irrelevant", filter_nll_to_subset(
            json.loads(args.irrelevant_nll.read_text(encoding="utf-8")), keep_ids), "irrelevant"))
    sec_alpha = 0.05 / len(secondary) if secondary else 0.05
    for key, comp_nll, lbl in secondary:
        coverage["models"][lbl] = require_full_coverage(lbl, comp_nll, keep_ids)
        comparisons[key] = paired_comparison(igt_nll, comp_nll, "igt", lbl, alpha_per_test=sec_alpha)

    out = {
        "eval_subset": "external_ood_leakage_free",
        "external_prefixes": list(EXTERNAL_PREFIXES),
        "n_prompts": len(centaur_nll),
        "experiments": exps,
        "coverage": coverage,
        "primary_comparison": "centaur_vs_igt",
        "comparisons": comparisons,
        "secondary_bonferroni": {"k": len(secondary), "alpha_per_test": sec_alpha} if secondary else None,
        "base_model_id": args.base_model_id,
        "igt_adapter": str(args.igt_adapter),
    }
    (args.output / "eval_lora_igt.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Stage 7.5.3 — Centaur (generalista) vs LoRA-IGT (especialista), external-OOD ===")
    c = comparisons["centaur_vs_igt"]
    print(f"PRIMARIA Centaur−LoRA-IGT: Δ={c['delta_mean']:+.4f}  d={c['cohens_d']:+.3f}  "
          f"P(1-sided)={c['p_value_one_sided']:.2e}  CI95=[{c['ci_lo_95']:+.4f},{c['ci_hi_95']:+.4f}]  n={c['n']}")
    for key in ("igt_vs_noise", "igt_vs_irrelevant"):
        if key in comparisons:
            x = comparisons[key]
            print(f"  {key}: Δ={x['delta_mean']:+.4f} d={x['cohens_d']:+.3f} P={x['p_value_one_sided']:.2e}")
    print(f"\nNext: interpret_igt_verdict.py --results {args.output / 'eval_lora_igt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
