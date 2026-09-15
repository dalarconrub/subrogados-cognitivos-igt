"""Anexo A, apartado A.5 · Análisis estadísticos: procedimiento paso a paso.

Para cada contraste de Centaur con otro modelo, retira un estudio de origen completo, recalcula la d de
Cohen con las sesiones de los trece restantes y repite con cada uno de los catorce; registra la mediana y el rango
de esas d, cuántas conservan el signo y qué estudio desplaza más la estimación. Las 511 sesiones de A se reparten
entre los nueve estudios de origen de la compilación de Steingroever et al. (2015) de los que proceden.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("<REPO>")

# Reusar funciones canonical de v1 (sin importar v2 que ejecuta main() sin guard)
sys.path.insert(0, str(ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout"))
from compute_nll_absolute_clean_1087 import (  # noqa: E402
    load_manifest as load_clean_manifest,
    build_partitions,
    load_cache_nll,
    load_llama_base_nll,
    load_cognitive_nll,
    CACHES,
)

# Override paths del módulo (los del módulo apuntan a paths inexistentes)
COGNITIVE_DIR_REAL = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout/results/h1_pooled_ml_subject_kfold_extended"

# Cache randominit canonical post-patch top-K (Colab Pro+ 2026-06-07)
RANDOMINIT_CACHE = ROOT / (
    "tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_h2_topk_recompute/"
    "randominit/nll_randominit__marcelbinz__Llama-3_1-RandomInit-70B_h2_recompute.json"
)

# Caches inline (sin importar v2 buggy)
LORA_IGT_470 = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/nll_lora_igt.json'
LORA_NOIGT_470 = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/nll_lora_noigt.json'
LORA_IGT_617 = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/nll_lora_igt_clean1087_gap617.json'
LORA_NOIGT_617 = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/nll_lora_noigt_clean1087_gap617.json'

NLL_LORA_NOIGT_FULLSCALE = ROOT / (
    "tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/"
    "nll_lora_noigt_fullscale_clean1087.json"
)

TRAIN_COHORT_MAPPING = ROOT / "tesis/research_records/analyses/A_TRAIN_511_subjects_to_cohort_historico_mapping.json"


def load_fused(p470, p617):
    out = {}
    for p in [p470, p617]:
        with open(p) as f:
            for r in json.load(f):
                out[r['prompt_id']] = r['nll']
    return out


def cohen_d_paired(deltas):
    if len(deltas) < 2:
        return float("nan")
    sd = statistics.stdev(deltas)
    if sd == 0:
        return float("nan")
    return statistics.mean(deltas) / sd


def remap_experiments(manifest, train_mapping):
    """Construir experiments_by_sid con A_TRAIN desagregado a cohortes históricas.

    Para los 511 sujetos de A_TRAIN: usar la cohorte histórica del mapping
    (formato `igt_steingroever2015_<cohorte>`).
    Para los 66 B (cohortes residuales Steingroever) + 510 C (externos): usar el
    `experiment` original del manifest.

    Esto unifica Wood_A (136 de A_TRAIN) con Wood_B (17 de HOLDOUT proxy) bajo
    `igt_steingroever2015_Wood` (153 sujetos totales), consolidando 23 → 20
    experimentos efectivos sobre Combinado_1087.
    """
    by_sid = {}
    n_train_remapped = 0
    n_other = 0
    for r in manifest:
        sid = r["subject_id"]
        if sid in train_mapping:
            cohort = train_mapping[sid]["cohort"]
            by_sid[sid] = f"igt_steingroever2015_{cohort}"
            n_train_remapped += 1
        else:
            by_sid[sid] = r["experiment"]
            n_other += 1
    return by_sid, n_train_remapped, n_other


def stats_for_contrast_loeo(a_nll, b_nll, subject_ids, experiments_by_sid):
    deltas_by_exp = {}
    for sid in subject_ids:
        if sid in a_nll and sid in b_nll and sid in experiments_by_sid:
            d = a_nll[sid] - b_nll[sid]
            e = experiments_by_sid[sid]
            deltas_by_exp.setdefault(e, []).append(d)

    all_deltas = [d for ds in deltas_by_exp.values() for d in ds]
    n_total = len(all_deltas)
    if n_total < 2:
        return None
    d_primary = cohen_d_paired(all_deltas)
    sign_primary = 1 if d_primary > 0 else (-1 if d_primary < 0 else 0)

    experiments_sorted = sorted(deltas_by_exp.keys())
    loeo_estimates = []
    for e_excl in experiments_sorted:
        complement = [d for e, ds in deltas_by_exp.items() if e != e_excl for d in ds]
        n_compl = len(complement)
        d_loeo = cohen_d_paired(complement) if n_compl >= 2 else float("nan")
        loeo_estimates.append({
            "experiment_excluded": e_excl,
            "n_subjects_excluded": len(deltas_by_exp[e_excl]),
            "n_complement": n_compl,
            "d_loeo": d_loeo,
        })

    d_values = [r["d_loeo"] for r in loeo_estimates if isinstance(r["d_loeo"], float) and math.isfinite(r["d_loeo"])]
    d_median = statistics.median(d_values) if d_values else float("nan")
    d_min = min(d_values) if d_values else float("nan")
    d_max = max(d_values) if d_values else float("nan")
    n_iter_signed = sum(1 for d_e in d_values if (d_e > 0 and sign_primary > 0) or (d_e < 0 and sign_primary < 0))

    influences = [
        (r["experiment_excluded"], abs(r["d_loeo"] - d_primary))
        for r in loeo_estimates if isinstance(r["d_loeo"], float) and math.isfinite(r["d_loeo"])
    ]
    influences.sort(key=lambda x: -x[1])
    most_influential = influences[0] if influences else (None, None)

    return {
        "n_primary": n_total,
        "n_experiments_in_intersection": len(experiments_sorted),
        "d_primary": d_primary,
        "d_median_loeo": d_median,
        "d_min_loeo": d_min,
        "d_max_loeo": d_max,
        "n_iter_total": len(d_values),
        "n_iter_signed_preserved": n_iter_signed,
        "fraction_signed_preserved": n_iter_signed / len(d_values) if d_values else float("nan"),
        "most_influential_experiment": most_influential[0],
        "most_influential_deviation": most_influential[1],
        "loeo_estimates": loeo_estimates,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data_runtime/analysis/dm_paper1_igt1598/h1_h2_h3_loeo_centaur_disaggregated.json",
    )
    args = ap.parse_args()

    print("# 1. Loading manifest + partitions + train remapeo...", file=sys.stderr)
    manifest = load_clean_manifest()
    manifest_no_holdout = [
        r for r in manifest
        if not r["experiment"].startswith("igt_steingroever2015_psych101_holdout")
    ]
    parts = build_partitions(manifest)
    parts["COMBINED_1087"] = (
        parts["A_TRAIN_511"]
        + parts["B_HOLDOUT_66_proxy_hash_gemelos"]
        + parts["C_EXTERNAL_510"]
    )
    print(f"   COMBINED_1087: n={len(parts['COMBINED_1087'])}", file=sys.stderr)

    train_map = json.loads(TRAIN_COHORT_MAPPING.read_text())["mapping"]
    experiments_by_sid, n_train_remapped, n_other = remap_experiments(manifest, train_map)
    print(f"   Remapeados A_TRAIN: {n_train_remapped} sujetos; otros: {n_other}", file=sys.stderr)

    sids_combined = set(parts["COMBINED_1087"])
    exps_eff = Counter(experiments_by_sid[s] for s in sids_combined if s in experiments_by_sid)
    print(f"\n# 2. Experimentos efectivos sobre Combinado_1087 (desagregado):", file=sys.stderr)
    for e, n in sorted(exps_eff.items(), key=lambda x: -x[1]):
        print(f"   {e:<55} {n:>5}", file=sys.stderr)
    print(f"   TOTAL: {len(exps_eff)}", file=sys.stderr)

    print("\n# 3. Cargando caches NLL de los 10 modelos del contraste Centaur...", file=sys.stderr)
    models = {}
    models["centaur"] = load_cache_nll(CACHES["centaur"])
    models["llama_base"] = load_llama_base_nll(manifest_no_holdout)
    for mname, fname in [("vse", "vse.jsonl"), ("orl", "orl.jsonl"), ("pvldelta", "pvldelta.jsonl")]:
        models[mname] = load_cognitive_nll(COGNITIVE_DIR_REAL / fname)
    models["lora_noise"] = load_cache_nll(CACHES["lora_noise"])
    models["lora_irrelevant"] = load_cache_nll(CACHES["lora_irrelevant"])
    models["randominit"] = load_cache_nll(RANDOMINIT_CACHE)
    models["lora_igt"] = load_fused(LORA_IGT_470, LORA_IGT_617)
    models["lora_noigt"] = load_fused(LORA_NOIGT_470, LORA_NOIGT_617)
    models["lora_noigt_fullscale"] = load_cache_nll(NLL_LORA_NOIGT_FULLSCALE)
    for k, v in models.items():
        print(f"   {k}: {len(v)} sujetos", file=sys.stderr)

    contrasts_10_centaur = [
        ("centaur_vs_llama_base", "centaur", "llama_base", "Externos"),
        ("centaur_vs_vse", "centaur", "vse", "Externos"),
        ("centaur_vs_orl", "centaur", "orl", "Externos"),
        ("centaur_vs_pvldelta", "centaur", "pvldelta", "Externos"),
        ("centaur_vs_lora_noise", "centaur", "lora_noise", "Arquitectura-contenido"),
        ("centaur_vs_lora_irrelevant", "centaur", "lora_irrelevant", "Arquitectura-contenido"),
        ("centaur_vs_randominit", "centaur", "randominit", "Arquitectura-contenido"),
        ("centaur_vs_lora_igt", "centaur", "lora_igt", "Intra-contenido"),
        ("centaur_vs_lora_noigt", "centaur", "lora_noigt", "Intra-contenido"),
        ("centaur_vs_lora_noigt_fullscale", "centaur", "lora_noigt_fullscale", "Intra-contenido"),
    ]

    print(f"\n# 4. Computando LOEO sobre {len(contrasts_10_centaur)} contrastes Centaur con {len(exps_eff)} experimentos desagregados...", file=sys.stderr)
    results = []
    for cname, a, b, bloque in contrasts_10_centaur:
        stats = stats_for_contrast_loeo(models[a], models[b], parts["COMBINED_1087"], experiments_by_sid)
        if stats is None:
            print(f"   {cname}: SIN datos", file=sys.stderr)
            continue
        stats["contrast"] = cname
        stats["bloque"] = bloque
        stats["a"] = a
        stats["b"] = b
        results.append(stats)
        print(
            f"   {cname:<40} "
            f"d={stats['d_primary']:+.3f} "
            f"d_min={stats['d_min_loeo']:+.3f} "
            f"d_max={stats['d_max_loeo']:+.3f} "
            f"signo={stats['n_iter_signed_preserved']}/{stats['n_iter_total']} "
            f"most_influential={stats['most_influential_experiment']} "
            f"(|Δd|={stats['most_influential_deviation']:.3f})",
            file=sys.stderr,
        )

    output = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "phase": "LOEO desagregado 10 contrastes Centaur vs resto (director-decision 2026-06-07; A_TRAIN desagregado a cohortes históricas Steingroever 2015; consolidación 23 → 20 experimentos efectivos)",
        "method": "Hash SHA-256 sobre (gain, |loss|) × N-1 trials para remapear 511 A_TRAIN → cohortes históricas Steingroever 2015. Las 9 cohortes que aparecen en A y B se unifican bajo un solo experimento. LOEO determinístico sobre Combinado_1087 con la nueva granularidad.",
        "n_experiments_effective_disaggregated": len(exps_eff),
        "experiments_distribution": dict(exps_eff),
        "contrasts": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n# 5. JSON output: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
