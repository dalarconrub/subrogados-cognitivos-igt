#!/usr/bin/env python3
"""Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Carga la base con cada uno de los tres adaptadores, pasa cada sesión del corpus en una sola pasada y
calcula la verosimilitud negativa media por sesión sobre el vocabulario completo, sin renormalizar.
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
BONFERRONI_K = 2
ALPHA_PER_TEST = 0.05 / BONFERRONI_K  # 0.025


def load_or_compute_nll(adapter_path: str, igt_data: list[dict], base_model_id: str,
                       cache_file: Path | None = None) -> list[dict]:
    """Load cached NLL if exists, else compute via inference."""
    if cache_file is not None and cache_file.exists():
        print(f"  Loading cached NLLs from {cache_file}")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    # Inference path: import locally to avoid heavy imports at module load
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        # [revisión interna] O8.N3: token-weighted NLL sobre answer tokens entre <<X>>,
        # NO full-sequence loss. Ver Stage 7.5.1 masked_loss_collator.py.
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "stage_7_5_1_lora_noise"))
        from masked_loss_collator import compute_answer_token_nll
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

    print(f"  Loading base + adapter {adapter_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    out: list[dict] = []
    for prompt in igt_data:
        text = prompt["text"]
        nll_info = compute_answer_token_nll(model, tokenizer, text)
        out.append({
            "prompt_id": prompt["prompt_id"],
            "experiment": prompt.get("experiment", "igt"),
            "cohort": prompt.get("cohort"),
            "nll": nll_info["mean_nll"],
            "n_answer_tokens": nll_info["n_answer_tokens"],
            "n_answer_spans": nll_info["n_answer_spans"],
        })
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def paired_comparison(group_a: list[dict], group_b: list[dict], label_a: str, label_b: str) -> dict:
    a_by_pid = {x["prompt_id"]: x["nll"] for x in group_a}
    b_by_pid = {x["prompt_id"]: x["nll"] for x in group_b}
    common = sorted(set(a_by_pid) & set(b_by_pid))
    if not common:
        raise ValueError(f"No paired prompts ({label_a} vs {label_b})")
    deltas = np.array([a_by_pid[p] - b_by_pid[p] for p in common], dtype=float)
    t_stat, p_two = scstats.ttest_1samp(deltas, 0.0)
    p_one = (p_two / 2) if t_stat < 0 else 1.0 - (p_two / 2)
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
        "p_value_one_sided": float(p_one),
        "cohens_d": d,
        # Cast to Python bool: scipy/numpy returns numpy.bool_ which json
        # encoder rejects (TypeError: Object of type bool is not JSON
        # serializable). Bug encountered Stage 7.5.2 canonical Cell 5
        # 2026-05-19; fix preserved here.
        "bonferroni_significant": bool(p_one < ALPHA_PER_TEST),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "alpha_per_test": ALPHA_PER_TEST,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--centaur-adapter", required=True)
    p.add_argument("--noise-adapter", required=True)
    p.add_argument("--irrelevant-adapter", required=True)
    p.add_argument("--igt-data", type=Path, required=True)
    # NOTE: same as eval_2way.py — Centaur adapter trained on Llama-3.1-70B
    # BASE; ALL three adapters must be evaluated on same base for fair
    # 3-way comparison + matching Centaur native training environment.
    p.add_argument("--base-model-id", default="meta-llama/Llama-3.1-70B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-cached-nll", action="store_true",
                   help="If results/stage_7_5_2/nll_*.json exist, reuse them (skip inference)")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    igt_prompts = json.loads(args.igt_data.read_text(encoding="utf-8"))
    print(f"IGT prompts loaded: {len(igt_prompts)}")

    cache_c = args.output / "nll_centaur.json" if args.reuse_cached_nll else None
    cache_n = args.output / "nll_lora_noise.json" if args.reuse_cached_nll else None
    cache_i = args.output / "nll_lora_irrelevant.json" if args.reuse_cached_nll else None

    print("[1/4] NLL Centaur...")
    centaur_nll = load_or_compute_nll(args.centaur_adapter, igt_prompts, args.base_model_id, cache_c)

    print("[2/4] NLL LoRA-noise...")
    noise_nll = load_or_compute_nll(args.noise_adapter, igt_prompts, args.base_model_id, cache_n)

    print("[3/4] NLL LoRA-irrelevant...")
    irrelevant_nll = load_or_compute_nll(args.irrelevant_adapter, igt_prompts, args.base_model_id, cache_i)

    (args.output / "nll_centaur.json").write_text(json.dumps(centaur_nll, indent=2), encoding="utf-8")
    (args.output / "nll_lora_noise.json").write_text(json.dumps(noise_nll, indent=2), encoding="utf-8")
    (args.output / "nll_lora_irrelevant.json").write_text(json.dumps(irrelevant_nll, indent=2), encoding="utf-8")

    print("[4/4] Paired comparisons + Bonferroni-2...")
    comp_vs_noise = paired_comparison(centaur_nll, noise_nll, "centaur", "noise")
    comp_vs_irrelevant = paired_comparison(centaur_nll, irrelevant_nll, "centaur", "irrelevant")

    out = {
        "comparisons": {
            "centaur_vs_noise": comp_vs_noise,
            "centaur_vs_irrelevant": comp_vs_irrelevant,
        },
        "bonferroni": {
            "k_comparisons": BONFERRONI_K,
            "alpha_per_test": ALPHA_PER_TEST,
            "family_alpha": 0.05,
        },
    }
    (args.output / "eval_3way.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Stage 7.5.2 3-way comparison summary ===")
    print(f"Centaur vs LoRA-noise:")
    print(f"  Δ = {comp_vs_noise['delta_mean']:+.4f}, d = {comp_vs_noise['cohens_d']:+.3f}, "
          f"P = {comp_vs_noise['p_value_one_sided']:.2e}, Bonf-sig = {comp_vs_noise['bonferroni_significant']}")
    print(f"Centaur vs LoRA-irrelevant:")
    print(f"  Δ = {comp_vs_irrelevant['delta_mean']:+.4f}, d = {comp_vs_irrelevant['cohens_d']:+.3f}, "
          f"P = {comp_vs_irrelevant['p_value_one_sided']:.2e}, Bonf-sig = {comp_vs_irrelevant['bonferroni_significant']}")
    print(f"\nNext: run interpret_h4_verdict.py --results {args.output / 'eval_3way.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
