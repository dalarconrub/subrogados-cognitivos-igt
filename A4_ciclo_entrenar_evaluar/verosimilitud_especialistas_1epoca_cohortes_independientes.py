#!/usr/bin/env python3
"""Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Calcula la verosimilitud negativa por sesión y por ensayo de LoRA-IGT-1época y LoRA-noIGT-1época sobre las
sesiones de las cohortes independientes, y la compara con la de Centaur y los dos controles.
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
EXTERNAL_PREFIXES = ("igt_ahn2014_", "igt_kildahl2020", "igt_chavez2026_", "igt_sullivantoole2022")

# Comparaciones científicas (A, B): delta = NLL_A - NLL_B.
COMPARISONS = [
    ("centaur_vs_igt", "centaur", "igt"),
    ("centaur_vs_noigt", "centaur", "noigt"),
    ("igt_vs_noigt", "igt", "noigt"),
    ("noigt_vs_noise", "noigt", "noise"),
    ("noigt_vs_irrelevant", "noigt", "irrelevant"),
]


def is_external(experiment: str) -> bool:
    return any(experiment.startswith(p) for p in EXTERNAL_PREFIXES)


def compute_nll(adapter_path: str, prompts: list[dict], base_model_id: str,
                per_trial: bool = True, top_k: int = 5, per_trial_out: Path | None = None) -> list[dict]:
    """Per-session NLL (compute_answer_token_nll, igual que el 3-way → comparable con
    las caches de Centaur/noise/irrelevant) + per-trial logprobs completos
    (compute_per_trial_logprobs, formato Paper 1: target/argmax/top-k por ensayo)
    para los cálculos pormenorizados posteriores. El per-trial se guarda aparte."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "stage_7_5_1_lora_noise"))
    from masked_loss_collator import compute_answer_token_nll, compute_per_trial_logprobs
    if not torch.cuda.is_available():
        print("ERROR: requires GPU for inference.", file=sys.stderr)
        sys.exit(3)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  Loading base + adapter {adapter_path}...")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    out, pt_out = [], []
    for pr in prompts:
        info = compute_answer_token_nll(model, tok, pr["text"])
        out.append({"prompt_id": pr["prompt_id"], "experiment": pr.get("experiment", "igt"),
                    "cohort": pr.get("cohort"), "nll": info["mean_nll"],
                    "n_answer_tokens": info["n_answer_tokens"], "n_answer_spans": info["n_answer_spans"]})
        if per_trial:
            ptl = compute_per_trial_logprobs(model, tok, pr["text"], top_k=top_k)
            pt_out.append({"prompt_id": pr["prompt_id"], "experiment": pr.get("experiment", "igt"),
                           "cohort": pr.get("cohort"), "aggregate_nll": ptl["aggregate_nll"],
                           "n_answer_tokens": ptl["n_answer_tokens"], "n_answer_spans": ptl["n_answer_spans"],
                           "top_k": top_k, "per_trial": ptl["per_trial"]})
    if per_trial and per_trial_out is not None:
        per_trial_out.parent.mkdir(parents=True, exist_ok=True)
        per_trial_out.write_text(json.dumps(pt_out, indent=2), encoding="utf-8")
        print(f"  per-trial -> {per_trial_out} ({len(pt_out)} sesiones, top_k={top_k})")
    # Liberar VRAM antes de devolver: el script evalua 2 adapters (igt, noigt) en el
    # MISMO proceso; sin esto, cargar el 2o modelo OOMea (device_map=auto vuelca modulos
    # a CPU y bnb-4bit lo rechaza con "Some modules are dispatched on the CPU"). Fix 2026-05-23.
    import gc as _gc
    del model
    _gc.collect(); torch.cuda.empty_cache()
    return out


def filt(records: list[dict], keep: set[str]) -> dict[str, float]:
    return {r["prompt_id"]: r["nll"] for r in records if r["prompt_id"] in keep}


def require_full_coverage(name: str, model_map: dict[str, float], keep: set[str]) -> dict:
    """P1.1: hard-fail si un modelo no cubre los external completos."""
    missing = sorted(keep - set(model_map))
    if missing:
        raise ValueError(
            f"{name}: cobertura incompleta {len(keep) - len(missing)}/{len(keep)} external; "
            f"faltan {len(missing)} (e.g. {missing[:5]}). Cada modelo debe cubrir TODOS los external.")
    return {"model_n": len(set(model_map) & keep), "missing_n": 0}


def paired(a: dict[str, float], b: dict[str, float], alpha: float) -> dict:
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no paired prompts")
    deltas = np.array([a[p] - b[p] for p in common], dtype=float)
    t_stat, p_two = scstats.ttest_1samp(deltas, 0.0)
    p_less = (p_two / 2) if t_stat < 0 else 1.0 - (p_two / 2)      # A<B (A mejor)
    p_greater = (p_two / 2) if t_stat > 0 else 1.0 - (p_two / 2)   # A>B (B mejor)
    mean = float(deltas.mean()); sd = float(deltas.std(ddof=1))
    d = mean / sd if sd > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    boot = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(N_BOOTSTRAP)])
    return {"n": len(deltas), "delta_mean": mean, "delta_sd": sd,
            "ci_lo_95": float(np.quantile(boot, 0.025)), "ci_hi_95": float(np.quantile(boot, 0.975)),
            "t_stat": float(t_stat), "df": len(deltas) - 1, "p_value_one_sided": float(p_less),
            "p_one_sided_less": float(p_less), "p_one_sided_greater": float(p_greater),
            "cohens_d": d, "bonferroni_significant": bool(p_less < alpha),
            "alpha_per_test": alpha, "n_bootstrap": N_BOOTSTRAP, "seed": SEED}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--igt-data", type=Path, required=True)
    p.add_argument("--base-model-id", default="meta-llama/Llama-3.1-70B")
    p.add_argument("--centaur-nll", type=Path, required=True)
    p.add_argument("--noise-nll", type=Path, default=None)
    p.add_argument("--irrelevant-nll", type=Path, default=None)
    p.add_argument("--randominit-nll", type=Path, default=None,
                   help="NLL per-prompt de RandomInit base. Se reusa como floor "
                        "check del eje-2 (pre-entrenamiento), independiente del Bonferroni de la disociación.")
    p.add_argument("--igt-adapter", default=None)
    p.add_argument("--noigt-adapter", default=None)
    p.add_argument("--igt-nll", type=Path, default=None, help="Reuse cache de LoRA-IGT (salta GPU)")
    p.add_argument("--noigt-nll", type=Path, default=None, help="Reuse cache de LoRA-noIGT (salta GPU)")
    p.add_argument("--no-per-trial", action="store_true",
                   help="No computar per-trial logprobs (por defecto SÍ se computan, formato Paper 1).")
    p.add_argument("--per-trial-topk", type=int, default=5,
                   help="Top-K alternativas por ensayo en el per-trial (Paper 1 IGT usó 5; sweet-spot general 10).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.igt_data.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = list(rows.values())
    eval_prompts = [r for r in rows if is_external(r.get("experiment", ""))]
    keep = {r["prompt_id"] for r in eval_prompts}
    exps = sorted({r["experiment"] for r in eval_prompts})
    print(f"external-OOD leakage-free: {len(eval_prompts)} prompts | experiments: {exps}")

    models: dict[str, dict[str, float]] = {}
    models["centaur"] = filt(json.loads(args.centaur_nll.read_text(encoding="utf-8")), keep)
    if args.noise_nll and args.noise_nll.exists():
        models["noise"] = filt(json.loads(args.noise_nll.read_text(encoding="utf-8")), keep)
    if args.irrelevant_nll and args.irrelevant_nll.exists():
        models["irrelevant"] = filt(json.loads(args.irrelevant_nll.read_text(encoding="utf-8")), keep)
    if args.randominit_nll and args.randominit_nll.exists():
        models["randominit"] = filt(json.loads(args.randominit_nll.read_text(encoding="utf-8")), keep)

    # LoRA-IGT / LoRA-noIGT: reuse cache si se pasa, si no compute fresh desde adapter.
    for name, adapter, cache in (("igt", args.igt_adapter, args.igt_nll),
                                 ("noigt", args.noigt_adapter, args.noigt_nll)):
        if cache and cache.exists():
            print(f"[{name}] reuse cache {cache}")
            recs = json.loads(cache.read_text(encoding="utf-8"))
        elif adapter:
            print(f"[{name}] computing NLL + per-trial on external subset...")
            recs = compute_nll(adapter, eval_prompts, args.base_model_id,
                               per_trial=not args.no_per_trial, top_k=args.per_trial_topk,
                               per_trial_out=args.output / f"per_trial_lora_{name}.json")
            (args.output / f"nll_lora_{name}.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
        else:
            print(f"[{name}] sin adapter ni cache -> se omiten sus comparaciones")
            continue
        models[name] = filt(recs, keep)

    # P1.1: gate de cobertura completa (todos los external) por cada modelo presente.
    coverage = {"expected_n": len(keep),
                "models": {m: require_full_coverage(m, mp, keep) for m, mp in models.items()}}
    print(f"[coverage] {len(models)} modelos cubren los {len(keep)} external OK")

    # Comparaciones computables (ambos lados presentes); Bonferroni sobre el nº computado.
    todo = [(k, a, b) for (k, a, b) in COMPARISONS if a in models and b in models]
    if not todo:
        print("ERROR: ninguna comparación computable (faltan modelos)", file=sys.stderr)
        return 1
    alpha = 0.05 / len(todo)
    comparisons = {k: {**paired(models[a], models[b], alpha), "comparison": f"{a}_minus_{b}"}
                   for (k, a, b) in todo}

    # Floor checks vs RandomInit: independientes del
    # Bonferroni de la disociación (alpha=0.05). Sanity floor del eje-2: confirma que
    # los adaptadores entrenados baten trivialmente a la base random sobre este subset.
    floor_checks: dict[str, dict] = {}
    if "randominit" in models:
        for m in ("centaur", "igt", "noigt"):
            if m in models:
                floor_checks[f"{m}_vs_randominit"] = {
                    **paired(models[m], models["randominit"], 0.05),
                    "comparison": f"{m}_minus_randominit"}

    out = {
        "eval_subset": "external_ood_leakage_free",
        "external_prefixes": list(EXTERNAL_PREFIXES),
        "n_prompts": len(models["centaur"]),
        "experiments": exps,
        "models_present": sorted(models.keys()),
        "coverage": coverage,
        # P2.2: comparaciones requeridas para un verdict 2x2 final (transfer × specific).
        "required_for_verdict": ["igt_vs_noigt", "noigt_vs_noise", "noigt_vs_irrelevant"],
        "bonferroni": {"k": len(todo), "alpha_per_test": alpha, "family_alpha": 0.05},
        "comparisons": comparisons,
        "floor_checks": floor_checks or None,
        "base_model_id": args.base_model_id,
    }
    (args.output / "eval_dissociation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== Disociación IGT (external-OOD, n={out['n_prompts']}, Bonferroni-{len(todo)} α={alpha:.4f}) ===")
    for k, c in comparisons.items():
        print(f"  {k:>22}: Δ={c['delta_mean']:+.4f} d={c['cohens_d']:+.3f} "
              f"P={c['p_value_one_sided']:.2e} sig={c['bonferroni_significant']}")
    if floor_checks:
        print("  -- floor checks vs RandomInit (eje-2, no Bonferroni, α=0.05) --")
        for k, c in floor_checks.items():
            print(f"  {k:>22}: Δ={c['delta_mean']:+.4f} d={c['cohens_d']:+.3f} sig={c['bonferroni_significant']}")
    print(f"\nNext: interpret_dissociation_verdict.py --results {args.output/'eval_dissociation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
