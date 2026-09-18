"""Anexo A, apartado A.5 · Análisis estadísticos: módulos comunes y scripts.

A partir de las probabilidades por ensayo guardadas de Centaur y de la base, calcula para cada ensayo la
verosimilitud negativa con las dos convenciones, sobre el vocabulario completo y renormalizada sobre los cuatro
mazos de la sesión, y recalcula con ambas los contrastes de cada modelo con VSE, ORL y PVL-Δ por partición, con su
intervalo por bootstrap de los estudios de origen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path("<REPO>")

# Reusar particiones canonical del re-encuadre limpio
sys.path.insert(0, str(ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout"))
from compute_nll_absolute_clean_1087 import (  # noqa: E402
    load_manifest as load_clean_manifest,
    build_partitions,
    load_cognitive_nll,
    COGNITIVE_DIR,
)

# Caches de top_k_logprobs (manifest fused [revisión interna])
CENTAUR_TOPK = ROOT / (
    "data_runtime/use_cases/real_centaur_psych101/runs/"
    "track-d-paper1-centaur-70b-adapter-igt-fused-1598-pertrial/real_centaur_trials.json"
)
LLAMA_TOPK = ROOT / (
    "data_runtime/use_cases/real_centaur_psych101/runs/"
    "track-d-paper1-llama-3.1-70b-base-igt-fused-1598-pertrial/real_centaur_trials.json"
)
# Cache agg de Centaur (provee prompt_id por índice)
CENTAUR_AGG = ROOT / (
    "tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_2/"
    "canonical_base/nll_centaur.json"
)
# Manifest base con deck_label_map por sujeto (key = (experiment, subject_id))
MANIFEST_BASE = ROOT / "data_runtime/plans/dm_paper1/igt_paper1_fused_inference_data.jsonl"

SEED = 20260506
N_BOOT = 10_000


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exp = [math.exp(l - m) for l in logits]
    s = sum(exp)
    return [e / s for e in exp]


def load_deck_maps() -> dict[tuple[str, str], dict[str, str]]:
    """Manifest base → dict (experiment, key) → deck_label_map.

    Genera dos keys por sujeto para soportar la convención asimétrica del cache top-K:
      - key principal: (experiment, subject_id) — formato compuesto del manifest base
        e.g. ('igt_steingroever2015_psych101_train_exp1', 'psych101_train_steingroever_exp1_0')
      - key alternativa: (experiment, suffix_numerico) — formato raw del cache top-K
        e.g. ('igt_steingroever2015_psych101_train_exp1', '0')

    El cache top-K usa `metadata.participant` que para los psych101_train sujetos es
    el sufijo numérico (e.g. '0'), no el subject_id compuesto.
    """
    out: dict[tuple[str, str], dict[str, str]] = {}
    with open(MANIFEST_BASE) as f:
        for line in f:
            d = json.loads(line)
            exp = d["experiment"]
            sid = str(d["subject_id"])
            out[(exp, sid)] = d["deck_label_map"]
            # key alternativa: último segmento tras último '_'
            if "_" in sid:
                suffix = sid.rsplit("_", 1)[-1]
                if suffix != sid:
                    out.setdefault((exp, suffix), d["deck_label_map"])
    return out


def load_prompt_ids(agg_cache: Path) -> list[str]:
    """Cache agg ordenado → lista de prompt_ids canónicos por índice."""
    return [r["prompt_id"] for r in json.load(open(agg_cache))]


def compute_renorm4_per_subject(
    topk_cache: Path, agg_prompt_ids: list[str], deck_maps: dict, model_label: str
) -> dict:
    """Per cada sujeto del cache top_k: computa NLL renorm4 + NLL raw + masa promedio.

    Returns:
        dict prompt_id -> {nll_renorm4, nll_raw, mass_on_4_mean, n_kept, n_excluded, n_trials}
    """
    print(f"# Loading top-k cache: {topk_cache.name} ({model_label})...", file=sys.stderr)
    cache = json.load(open(topk_cache))
    if len(cache) != len(agg_prompt_ids):
        raise RuntimeError(
            f"Mismatch: top_k={len(cache)} vs agg_prompt_ids={len(agg_prompt_ids)}"
        )

    out: dict[str, dict] = {}
    n_subjects_no_dlm = 0

    for i, s in enumerate(cache):
        md = s["metadata"]
        exp = md["experiment"]
        part = str(md["participant"])
        prompt_id = agg_prompt_ids[i]
        dlm = deck_maps.get((exp, part))
        if not dlm:
            n_subjects_no_dlm += 1
            continue
        # Manifest base tiene dos formatos de deck_label_map:
        #   - Formato A (externos): {'A': 'Y', 'B': 'C', 'C': 'J', 'D': 'P'} (dict de strings)
        #   - Formato B (psych101_train): {'deck_labels_in_order': ['H', 'V', 'J', 'D']} (wrapper)
        if "deck_labels_in_order" in dlm:
            deck_tokens = list(dlm["deck_labels_in_order"])
        else:
            deck_tokens = list(dlm.values())
        # Sanity: 4 tokens, todos strings
        if len(deck_tokens) != 4 or not all(isinstance(t, str) for t in deck_tokens):
            n_subjects_no_dlm += 1
            continue

        ptl = md.get("per_trial_logprobs") or []
        nlls_renorm4: list[float] = []
        nlls_raw: list[float] = []
        masses: list[float] = []
        n_excluded = 0

        for t in ptl:
            chosen = t.get("target_text", "")
            tk_tokens = t.get("top_k_tokens") or []
            tk_logprobs = t.get("top_k_logprobs") or []
            target_lp = t.get("target_token_logprobs") or [0.0]
            target_lp = target_lp[0] if target_lp else 0.0

            tk_map = dict(zip(tk_tokens, tk_logprobs))

            # Política mass_below_4 (a) conservadora: los 4 deck tokens deben estar todos en top-K
            if not all(dt in tk_map for dt in deck_tokens):
                n_excluded += 1
                continue
            if chosen not in deck_tokens:
                # Si el token elegido por el humano no es uno de los 4 deck tokens del sujeto,
                # el trial está malformado o el cross-link falla; excluir.
                n_excluded += 1
                continue

            deck_logits = [tk_map[dt] for dt in deck_tokens]
            renorm_probs = softmax(deck_logits)
            chosen_idx = deck_tokens.index(chosen)
            nll_r = -math.log(renorm_probs[chosen_idx])
            nlls_renorm4.append(nll_r)
            nlls_raw.append(-target_lp)
            # Mass on 4: suma de probabilidades brutas (no renormalizadas) de los 4 deck tokens
            masses.append(sum(math.exp(l) for l in deck_logits))

        n_kept = len(nlls_renorm4)
        if n_kept == 0:
            continue

        out[prompt_id] = {
            "nll_renorm4": sum(nlls_renorm4) / n_kept,
            "nll_raw": sum(nlls_raw) / n_kept,
            "mass_on_4_mean": sum(masses) / len(masses),
            "n_kept": n_kept,
            "n_excluded": n_excluded,
            "n_trials": len(ptl),
        }

    if n_subjects_no_dlm:
        print(
            f"# WARN: {n_subjects_no_dlm} sujetos sin deck_label_map (skipped)",
            file=sys.stderr,
        )
    return out


def cohen_d_paired(deltas: list[float]) -> float:
    if len(deltas) < 2:
        return float("nan")
    return statistics.mean(deltas) / statistics.stdev(deltas)


def t_paired(deltas: list[float]) -> tuple[float, int]:
    if len(deltas) < 2:
        return float("nan"), len(deltas) - 1
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    n = len(deltas)
    se = sd / math.sqrt(n)
    return m / se, n - 1


def cluster_bootstrap_ci_d(
    deltas_by_experiment: dict[str, list[float]], n_boot: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    experiments = list(deltas_by_experiment.keys())
    n_exp = len(experiments)
    d_replicates: list[float] = []
    for _ in range(n_boot):
        sampled = [experiments[rng.randint(0, n_exp - 1)] for _ in range(n_exp)]
        flat = []
        for e in sampled:
            flat.extend(deltas_by_experiment[e])
        if len(flat) < 2:
            continue
        m = statistics.mean(flat)
        sd = statistics.stdev(flat)
        if sd > 0:
            d_replicates.append(m / sd)
    d_replicates.sort()
    if not d_replicates:
        return float("nan"), float("nan")
    lo = d_replicates[int(0.025 * len(d_replicates))]
    hi = d_replicates[int(0.975 * len(d_replicates))]
    return lo, hi


def stats_for_contrast(
    llm_nll_by_pid: dict[str, dict],
    cog_nll_by_pid: dict[str, float],
    subject_ids: list[str],
    experiments_by_pid: dict[str, str],
    nll_key: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Contraste pareado LLM vs cognitivo: Δ_pid = NLL_llm - NLL_cog.

    Args:
        nll_key: 'nll_renorm4' o 'nll_raw' (selecciona métrica de LLM).
    """
    deltas: list[float] = []
    deltas_by_exp: dict[str, list[float]] = {}
    n_missing_llm = 0
    n_missing_cog = 0
    pids_kept: list[str] = []
    for pid in subject_ids:
        if pid not in llm_nll_by_pid:
            n_missing_llm += 1
            continue
        if pid not in cog_nll_by_pid:
            n_missing_cog += 1
            continue
        d = llm_nll_by_pid[pid][nll_key] - cog_nll_by_pid[pid]
        deltas.append(d)
        e = experiments_by_pid.get(pid, "UNKNOWN")
        deltas_by_exp.setdefault(e, []).append(d)
        pids_kept.append(pid)

    if len(deltas) < 2:
        return {
            "n": len(deltas),
            "n_missing_llm": n_missing_llm,
            "n_missing_cog": n_missing_cog,
            "error": "insufficient samples (<2)",
        }

    delta_mean = statistics.mean(deltas)
    delta_sd = statistics.stdev(deltas)
    d = cohen_d_paired(deltas)
    t_stat, df = t_paired(deltas)
    ci_lo, ci_hi = cluster_bootstrap_ci_d(deltas_by_exp, n_boot, seed)

    return {
        "n": len(deltas),
        "n_missing_llm": n_missing_llm,
        "n_missing_cog": n_missing_cog,
        "n_experiments": len(deltas_by_exp),
        "delta_mean": delta_mean,
        "delta_sd": delta_sd,
        "cohens_d": d,
        "t_stat": t_stat,
        "df": df,
        "ci_d_95": [ci_lo, ci_hi],
        "metric": nll_key,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "data_runtime/analysis/dm_paper1_igt1598/h1_renorm4_dual_metric_h1_vs_cognitive.json",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="Solo procesar primeros 50 sujetos para smoke test"
    )
    args = ap.parse_args()

    # 1. Cargar manifest clean1087 + particiones
    print("# 1. Loading clean1087 manifest + partitions...", file=sys.stderr)
    clean_manifest = load_clean_manifest()
    parts = build_partitions(clean_manifest)
    parts["COMBINED_1087"] = (
        parts["A_TRAIN_511"]
        + parts["B_HOLDOUT_66_proxy_hash_gemelos"]
        + parts["C_EXTERNAL_510"]
    )
    for k, v in parts.items():
        print(f"  {k}: n={len(v)}", file=sys.stderr)

    # 2. Cargar mapping subject_id → experiment (para bootstrap por conglomerados)
    print("\n# 2. Loading subject_id → experiment mapping...", file=sys.stderr)
    experiments_by_pid: dict[str, str] = {
        r["subject_id"]: r["experiment"] for r in clean_manifest
    }

    # 3. Cargar deck_label_maps del manifest base + prompt_ids del cache agg
    print("\n# 3. Loading deck_label_maps + prompt_ids...", file=sys.stderr)
    deck_maps = load_deck_maps()
    print(f"  {len(deck_maps)} (experiment, subject_id) entries in base manifest", file=sys.stderr)
    centaur_pids = load_prompt_ids(CENTAUR_AGG)
    print(f"  {len(centaur_pids)} prompt_ids from agg cache", file=sys.stderr)

    # 4. Calcular NLL renorm4 + raw para Centaur (cache top-K=5)
    print("\n# 4. Computing Centaur NLL renorm4 + raw ([revisión interna] subjects)...", file=sys.stderr)
    centaur = compute_renorm4_per_subject(
        CENTAUR_TOPK, centaur_pids, deck_maps, "Centaur"
    )
    print(f"  Centaur: {len(centaur)} subjects processed", file=sys.stderr)

    # 5. Calcular NLL renorm4 + raw para Llama base (cache top-K=10)
    print("\n# 5. Computing Llama base NLL renorm4 + raw ([revisión interna] subjects)...", file=sys.stderr)
    llama_base = compute_renorm4_per_subject(
        LLAMA_TOPK, centaur_pids, deck_maps, "Llama base"
    )
    print(f"  Llama base: {len(llama_base)} subjects processed", file=sys.stderr)

    # 6. Calcular Δ Centaur vs cada cognitivo (renorm4 + raw)
    print("\n# 6. Computing contrasts vs cognitive models...", file=sys.stderr)
    cognitive: dict[str, dict[str, float]] = {}
    for mname, fname in [("vse", "vse.jsonl"), ("orl", "orl.jsonl"), ("pvldelta", "pvldelta.jsonl")]:
        cognitive[mname] = load_cognitive_nll(COGNITIVE_DIR / fname)
        print(f"  {mname}: {len(cognitive[mname])} subjects", file=sys.stderr)

    # 7. Output: para cada partición × cada cognitivo × cada métrica (renorm4 + raw)
    print("\n# 7. Building output JSON...", file=sys.stderr)
    result = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "phase": "D5 renorm4 dual-metric implementación — decisión del estudio 2026-06-04 y opción 2 de 2026-06-06",
        "purpose": "Dual-metric NLL (Binz convencional + renorm4 sobre 4 mazos) sobre [revisión interna] del [revisión interna] (H1 vs cognitivos). Acotado a contrastes LLM vs cognitivo donde la asimetría de soporte probabilístico hace renorm4 metodológicamente obligatoria.",
        "method": "NLL Binz: -log(p_chosen) sobre vocabulario completo del transformador (target_token_logprobs). NLL renorm4: softmax restringido a los 4 deck tokens del sujeto (cross-link via deck_label_map del manifest base). Política mass_below_4 = (a) conservadora: excluir trial si los 4 deck tokens no están en top-K saved. Bootstrap por conglomerados de experimentos (10⁴ remuestreos, semilla 20260506) sobre la d de Cohen pareada.",
        "params": {"seed": SEED, "n_boot": N_BOOT},
        "policy_mass_below_4": "Opción (a) conservadora: trials excluidos si los 4 deck tokens del sujeto no están todos presentes en top-K_saved. Centaur K=5 cobertura 100,00 %; Llama base K=10 cobertura 99,09 %.",
        "partitions_defined": {k: {"n": len(v)} for k, v in parts.items()},
        "coverage": {
            "centaur": {
                "n_subjects_processed": len(centaur),
                "n_trials_kept_total": sum(s["n_kept"] for s in centaur.values()),
                "n_trials_excluded_total": sum(s["n_excluded"] for s in centaur.values()),
                "mass_on_4_mean_across_subjects": (
                    sum(s["mass_on_4_mean"] for s in centaur.values()) / len(centaur)
                    if centaur
                    else None
                ),
            },
            "llama_base": {
                "n_subjects_processed": len(llama_base),
                "n_trials_kept_total": sum(s["n_kept"] for s in llama_base.values()),
                "n_trials_excluded_total": sum(s["n_excluded"] for s in llama_base.values()),
                "mass_on_4_mean_across_subjects": (
                    sum(s["mass_on_4_mean"] for s in llama_base.values()) / len(llama_base)
                    if llama_base
                    else None
                ),
            },
        },
        "contrasts": {},
    }

    for part_name, sids in parts.items():
        result["contrasts"][part_name] = {}
        # Dos modelos focales del contraste vs cognitivo: Centaur + Llama base.
        # Sólo H1 vs cognitivos del [revisión interna] [revisión interna].a/b/c reporta dual-metric (alcance Opción 2).
        for llm_label, llm_map in [("centaur", centaur), ("llama_base", llama_base)]:
            for cog_name, cog_map in cognitive.items():
                # NLL renorm4
                r_renorm4 = stats_for_contrast(
                    llm_map, cog_map, sids, experiments_by_pid, "nll_renorm4"
                )
                # NLL raw (Binz convencional, replica el canonical)
                r_raw = stats_for_contrast(
                    llm_map, cog_map, sids, experiments_by_pid, "nll_raw"
                )
                contrast_key = f"{llm_label}_vs_{cog_name}"
                result["contrasts"][part_name][contrast_key] = {
                    "renorm4": r_renorm4,
                    "raw": r_raw,
                }
                d_renorm4 = r_renorm4.get("cohens_d")
                d_raw = r_raw.get("cohens_d")
                d_renorm4_str = f"{d_renorm4:.4f}" if d_renorm4 is not None and isinstance(d_renorm4, float) else "n/a"
                d_raw_str = f"{d_raw:.4f}" if d_raw is not None and isinstance(d_raw, float) else "n/a"
                print(
                    f"  {part_name} / {contrast_key}: "
                    f"renorm4 d={d_renorm4_str} | raw d={d_raw_str} (n={r_renorm4.get('n', 0)})",
                    file=sys.stderr,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n# Written: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
