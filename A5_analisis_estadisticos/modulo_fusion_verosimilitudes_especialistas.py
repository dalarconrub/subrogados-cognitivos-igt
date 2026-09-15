"""Anexo A, apartado A.5 · Análisis estadísticos: procedimiento paso a paso.

Une, para LoRA-IGT-1época y LoRA-noIGT-1época, la verosimilitud por sesión calculada sobre las cohortes
independientes con la calculada después sobre las sesiones restantes, y añade el desglose de la verosimilitud
absoluta de cada modelo por partición.
"""

import json, hashlib, statistics, math
from pathlib import Path

ROOT = Path('<REPO>')

# Reusar v1
import sys
sys.path.insert(0, str(ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout'))
from compute_nll_absolute_clean_1087 import (
    load_manifest, build_partitions, load_cache_nll, load_llama_base_nll,
    load_cognitive_nll, stats_block, CACHES, COGNITIVE_DIR
)

LORA_IGT_470 = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/nll_lora_igt.json'
LORA_NOIGT_470 = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/nll_lora_noigt.json'
LORA_IGT_617 = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/nll_lora_igt_clean1087_gap617.json'
LORA_NOIGT_617 = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base/nll_lora_noigt_clean1087_gap617.json'


def load_fused(p470, p617):
    """Fusiona caches 470 + 617 = 1087 sujetos."""
    out = {}
    for p in [p470, p617]:
        with open(p) as f:
            for r in json.load(f):
                out[r['prompt_id']] = r['nll']
    return out


def main():
    manifest = load_manifest()
    manifest_no_holdout = [r for r in manifest if not r['experiment'].startswith('igt_steingroever2015_psych101_holdout')]
    parts = build_partitions(manifest)
    parts['COMBINED_1087'] = parts['A_TRAIN_511'] + parts['B_HOLDOUT_66_proxy_hash_gemelos'] + parts['C_EXTERNAL_510']

    print('Partitions:', {k: len(v) for k, v in parts.items()})

    # Carga 10 modelos (8 antiguos + 2 nuevos LoRA-IGT/noIGT)
    models = {}
    for name in ['centaur', 'lora_noise', 'lora_irrelevant', 'randominit_tier_c25']:
        models[name] = load_cache_nll(CACHES[name])
    models['llama_base'] = load_llama_base_nll(manifest_no_holdout)
    for mname, fname in [('vse', 'vse.jsonl'), ('orl', 'orl.jsonl'), ('pvldelta', 'pvldelta.jsonl')]:
        models[mname] = load_cognitive_nll(COGNITIVE_DIR / fname)
    models['lora_igt'] = load_fused(LORA_IGT_470, LORA_IGT_617)
    models['lora_noigt'] = load_fused(LORA_NOIGT_470, LORA_NOIGT_617)

    print(f"\nCache coverages:")
    for mname, mmap in models.items():
        print(f"  {mname}: {len(mmap)} sujetos")

    # Resultado
    result = {
        'generator': 'tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout/compute_nll_absolute_clean_1087_v2.py',
        'version': '2.0',
        'phase': '[revisión interna] resolutivo — extensión NLL absolutos por modelo y partición (v2: añadidos LoRA-IGT y LoRA-noIGT)',
        'purpose': 'Director-driven 2026-05-30 PM opción c matriz completa: extiende la [revisión interna] del [revisión interna] [revisión interna] a 10 modelos × 4 particiones (40 celdas) tras la cobertura simétrica conseguida por el [revisión interna] extended (extension corriendo en Lightning A100 80GB, [revisión interna] gap × 2 adapters).',
        'method': v_method_str(),
        'lora_igt_noigt_interpretation_caveats': {
            'A_TRAIN_511': 'LEAKAGE TRIVIAL: LoRA-IGT y LoRA-noIGT entrenaron sobre psych101_train; la NLL aquí es cota inferior de memorización del especialista, no medida de generalización.',
            'B_HOLDOUT_66_proxy_hash_gemelos': 'LEAKAGE PARCIAL: los 66 gemelos hash son los mismos sujetos físicos del psych101_holdout 10%-test exacto Binz bajo prompt SHA-seeded distinto. LoRA-IGT/noIGT NO entrenaron sobre el prompt Steingroever externo, pero sí sobre el psych101_train original que contiene a los mismos sujetos físicos.',
            'C_EXTERNAL_510': 'CERO LEAKAGE: 470 cohortes externas Track B (Ahn HC/Amp/Her + Kildahl + Sullivan-Toole + Chávez) + 40 Maia (excluida del cache OOD por filtro EXTERNAL_PREFIXES, no por razón anti-leakage real).',
        },
        'partitions_defined': {k: {'n': len(v)} for k, v in parts.items()},
        'absolute_nll_per_partition_x_model': {},
    }

    for part_name, sids in parts.items():
        result['absolute_nll_per_partition_x_model'][part_name] = {}
        for mname, mmap in models.items():
            result['absolute_nll_per_partition_x_model'][part_name][mname] = stats_block(mmap, sids)

    # Print tabla
    print('\n# TABLA I.9 EXTENDIDA (mean ± SD; n_cobertura)\n')
    parts_order = ['A_TRAIN_511', 'B_HOLDOUT_66_proxy_hash_gemelos', 'C_EXTERNAL_510', 'COMBINED_1087']
    models_order = ['centaur', 'llama_base', 'lora_noise', 'lora_irrelevant', 'lora_igt', 'lora_noigt', 'randominit_tier_c25', 'vse', 'orl', 'pvldelta']
    print('| Modelo | ' + ' | '.join(parts_order) + ' |')
    print('|---' + '|---' * len(parts_order) + '|')
    for m in models_order:
        row = [m]
        for p in parts_order:
            s = result['absolute_nll_per_partition_x_model'][p][m]
            if s['n'] == 0:
                row.append('—')
            else:
                row.append(f"{s['mean']:.4f} ± {s['sd']:.4f} (n={s['n']})")
        print('| ' + ' | '.join(row) + ' |')

    OUT = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout/results/aggregate_stats_clean_1087_nll_absolute_breakdown_v2.json'
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {OUT}")

    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"SHA-256: {sha}")


def v_method_str():
    return ('NLL per-sujeto-mean-across-tokens en métrica homogénea para los 10 modelos. '
            'Centaur+LoRA-noise+LoRA-irrelevant+RandomInit del cache canonical stage_7_5_2; '
            'Llama base recomputado per-sujeto desde per_trial_logprobs del run real_centaur_psych101 sobre [revisión interna] fused; '
            'VSE/ORL/PVL-Δ del cache extended pooled-ML K=5 subject-kfold; '
            'LoRA-IGT y LoRA-noIGT fusionados desde cache canonical scale-matched preprint Track B (n=470 OOD) + cache extension [revisión interna] ext (n=617 gap: 511 A_TRAIN leakage trivial + 66 B_HOLDOUT proxy hash-gemelos + 40 C_Maia gap del filtro EXTERNAL_PREFIXES) = [revisión interna] sujetos union disjunta verificada (overlap=0). '
            'Bug loader Steingroever 2015 preservado opción α statu quo en los 106 sujetos B+C_Maia para comparabilidad bit-identical con el resto del cache.')

main()
