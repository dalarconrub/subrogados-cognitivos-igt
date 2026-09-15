"""Anexo A, apartado A.5 · Análisis estadísticos: procedimiento paso a paso.

Define las funciones comunes a todos los análisis: lectura del manifiesto por ensayo, construcción de las
particiones A (511 sesiones incluidas en el ajuste de Centaur), B (66 retenidas, identificadas por su huella de
ganancias y pérdidas), C (cohortes independientes) y Combinado, y carga de la verosimilitud negativa media por
sesión de cada modelo de lenguaje y de cada modelo cognitivo.
"""

import json
import hashlib
import statistics
import math
from pathlib import Path

ROOT = Path('<REPO>')

MANIFEST = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout/manifests/subjects_always4_n1598_plus_binz66.jsonl'

CACHES = {
    'centaur': ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_2/canonical_base/nll_centaur.json',
    'lora_noise': ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_2/canonical_base/nll_lora_noise.json',
    'lora_irrelevant': ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_2/canonical_base/nll_lora_irrelevant.json',
    'randominit_tier_c25': ROOT / 'data_runtime/paper_01_igt_outputs/stage_7_5_2/randominit_sensitivity/nll_randominit__marcelbinz__Llama-3_1-RandomInit-70B.json',
}
LLAMA_BASE_RESPONSES = ROOT / 'data_runtime/use_cases/real_centaur_psych101/runs/track-d-paper1-llama-3.1-70b-base-igt-fused-1598-pertrial/centaur_responses.json'
COGNITIVE_DIR = ROOT / 'data_runtime/analysis/dm_paper1_igt1598/h1_pooled_ml_subject_kfold_fixed'


def load_manifest():
    out = []
    with open(MANIFEST) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def fingerprint(rec, trim_last=False):
    """Hash de la secuencia (gain, loss) per trial.
    Bug loader Steingroever 2015 omite el último trial → para matchear
    holdout (length=100/95/150) con externos (length=99/94/149) se debe
    aplicar trim_last=True al holdout.
    """
    trials = rec['trials'][:-1] if trim_last else rec['trials']
    seq = [(t['gain'], t['loss']) for t in trials]
    return hashlib.sha256(json.dumps(seq, separators=(',', ':')).encode()).hexdigest()


def build_partitions(manifest):
    A_exps = {
        'igt_steingroever2015_psych101_train_exp1',
        'igt_steingroever2015_psych101_train_exp2',
        'igt_steingroever2015_psych101_train_exp3',
    }
    HOLDOUT_exps = {
        'igt_steingroever2015_psych101_holdout_exp1',
        'igt_steingroever2015_psych101_holdout_exp2',
        'igt_steingroever2015_psych101_holdout_exp3',
    }
    STEINGROEVER_EXT_exps = {
        'igt_steingroever2015_Horstmann', 'igt_steingroever2015_Wood',
        'igt_steingroever2015_SteingroverInPrep', 'igt_steingroever2015_Worthy',
        'igt_steingroever2015_Premkumar', 'igt_steingroever2015_Kjome',
        'igt_steingroever2015_Fridberg', 'igt_steingroever2015_Steingroever2011',
        'igt_steingroever2015_Wetzels',
    }
    C_exps = {
        'igt_kildahl2020', 'igt_sullivantoole2022',
        'igt_sullivantoole2022_retest_vv1', 'igt_sullivantoole2022_retest_vv2',
        'igt_sullivantoole2022_retest_vv3', 'igt_sullivantoole2022_retest_vv4',
        'igt_chavez2026_eeg59', 'igt_ahn2014_hc', 'igt_ahn2014_amphetamine',
        'igt_ahn2014_heroin', 'igt_steingroever2015_Maia',
    }
    A = [r for r in manifest if r['experiment'] in A_exps]
    HOLDOUT_exact = [r for r in manifest if r['experiment'] in HOLDOUT_exps]
    STEINGROEVER_ext = [r for r in manifest if r['experiment'] in STEINGROEVER_EXT_exps]
    C = [r for r in manifest if r['experiment'] in C_exps]

    holdout_fps = {fingerprint(r, trim_last=True) for r in HOLDOUT_exact}
    for r in STEINGROEVER_ext:
        r['_fp'] = fingerprint(r, trim_last=False)
    B = [r for r in STEINGROEVER_ext if r['_fp'] in holdout_fps]
    assert len(B) == 66, f'B proxy hash-gemelos = {len(B)}, esperado 66'

    return {
        'A_TRAIN_511': [r['subject_id'] for r in A],
        'B_HOLDOUT_66_proxy_hash_gemelos': [r['subject_id'] for r in B],
        'C_EXTERNAL_510': [r['subject_id'] for r in C],
    }


def load_cache_nll(path):
    """Returns dict subject_id (=prompt_id) -> nll_per_sujeto_mean."""
    with open(path) as f:
        records = json.load(f)
    return {r['prompt_id']: r['nll'] for r in records}


def load_llama_base_nll(manifest_no_holdout):
    """Reconstruir NLL per-sujeto-mean desde per_trial_logprobs del cache Llama base.

    Los records del cache NO tienen prompt_id/subject_id; el mapping es por orden:
    response[i] ↔ manifest_no_holdout[i] (verificado contra cache canonical Centaur
    nll_centaur.json que sí tiene prompt_id en el mismo orden con 0 mismatches/[revisión interna]).
    """
    with open(LLAMA_BASE_RESPONSES) as f:
        records = json.load(f)
    assert len(records) == len(manifest_no_holdout), (
        f'mismatch responses {len(records)} vs manifest_no_holdout {len(manifest_no_holdout)}'
    )
    out = {}
    for i, r in enumerate(records):
        ptl = r.get('per_trial_logprobs') or []
        if not ptl:
            continue
        per_trial_nlls = []
        for t in ptl:
            tlp = t.get('target_token_logprobs') or []
            if tlp:
                per_trial_nlls.append(-sum(tlp))
        if not per_trial_nlls:
            continue
        sid = manifest_no_holdout[i]['subject_id']
        out[sid] = sum(per_trial_nlls) / len(per_trial_nlls)
    return out


def load_cognitive_nll(model_jsonl):
    """JSONL records with eval_mean_nll per sujeto under pooled-ML K=5."""
    out = {}
    with open(model_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('low_train_warning'):
                continue
            sid = r['subject_id']
            nll = r.get('eval_mean_nll')
            if nll is not None and math.isfinite(nll):
                out[sid] = nll
    return out


def stats_block(nll_map, subject_ids):
    """Subset + mean ± SD."""
    vals = [nll_map[s] for s in subject_ids if s in nll_map]
    n = len(vals)
    if n == 0:
        return {'n': 0, 'mean': None, 'sd': None, 'n_missing': len(subject_ids)}
    if n == 1:
        return {'n': 1, 'mean': vals[0], 'sd': 0.0, 'n_missing': len(subject_ids) - 1}
    m = statistics.mean(vals)
    sd = statistics.stdev(vals)
    return {'n': n, 'mean': m, 'sd': sd, 'n_missing': len(subject_ids) - n}


def main():
    manifest = load_manifest()
    manifest_no_holdout = [r for r in manifest if not r['experiment'].startswith('igt_steingroever2015_psych101_holdout')]
    parts = build_partitions(manifest)
    parts['COMBINED_1087'] = parts['A_TRAIN_511'] + parts['B_HOLDOUT_66_proxy_hash_gemelos'] + parts['C_EXTERNAL_510']

    print(f"# Partition sizes:")
    for k, v in parts.items():
        print(f"  {k}: {len(v)}")
    print()

    models = {}
    for name in ['centaur', 'lora_noise', 'lora_irrelevant', 'randominit_tier_c25']:
        models[name] = load_cache_nll(CACHES[name])
    models['llama_base'] = load_llama_base_nll(manifest_no_holdout)
    for mname, fname in [('vse', 'vse.jsonl'), ('orl', 'orl.jsonl'), ('pvldelta', 'pvldelta.jsonl')]:
        models[mname] = load_cognitive_nll(COGNITIVE_DIR / fname)

    print(f"# Cache coverages:")
    for mname, mmap in models.items():
        print(f"  {mname}: {len(mmap)} sujetos")
    print()

    result = {
        'generator': 'tools/compute_nll_absolute_clean_1087.py (ad-hoc post-R3 closeout — director request 2026-05-30 PM)',
        'phase': '[revisión interna] resolutivo — extensión NLL absolutos por modelo y partición',
        'purpose': 'Director solicitó exponer NLL absolutos (no solo Δ NLL) de cada modelo sobre el re-encuadre limpio clean_1087. Tabla 8 modelos × 4 particiones (32 celdas). LoRA-IGT y LoRA-noIGT excluidos por diseño anti-leakage [revisión interna] del preprint Paper 1 (decisión director 2026-05-30: documentar asimetría en lugar de re-inferencia GPU).',
        'method': 'NLL per-sujeto-mean-across-tokens en métrica homogénea para los 8 modelos. Centaur+LoRA-noise+LoRA-irrelevant+RandomInit del cache canonical stage_7_5_2; Llama base recomputado per-sujeto desde per_trial_logprobs del run real_centaur_psych101 sobre [revisión interna] fused; VSE/ORL/PVL-Δ del cache extended pooled-ML K=5 subject-kfold con low_train_warning excluidos. Particiones: A=TRAIN_511 (psych101_train exp1+2+3); B=HOLDOUT_66 proxy via hash-gemelos SHA-256 trial-a-trial sobre 577 sub-cohortes Steingroever externas; C=EXTERNAL_510 (11 experiments externos genuinos: Maia + Kildahl + SST + Chávez + Ahn HC/Amp/Her). HOLDOUT_66 exacto Binz NO disponible en métrica canonical (cache n=1598, no incluye los 66 exactos); proxy hash-gemelos es la solución metodológica documentada en cap6 [revisión interna] D3 [revisión interna] + [revisión interna] [revisión interna] [revisión interna].5.',
        'partitions_defined': {
            'A_TRAIN_511': {'n': len(parts['A_TRAIN_511']), 'rationale': 'psych101_train_exp1+exp2+exp3 (sujetos vistos por Centaur durante fine-tuning)'},
            'B_HOLDOUT_66_proxy_hash_gemelos': {'n': len(parts['B_HOLDOUT_66_proxy_hash_gemelos']), 'rationale': 'sub-cohortes Steingroever externas que hash-gemelean trial-a-trial con los 66 sujetos del psych101_holdout exp1+exp2+exp3 — los mismos sujetos clínicos bajo prompt SHA-seeded distinto, no vistos por Centaur en su prompt-form canonical'},
            'C_EXTERNAL_510': {'n': len(parts['C_EXTERNAL_510']), 'rationale': 'cohortes y sub-cohortes IGT clínicas independientes (Maia, Kildahl, Sullivan-Toole, Chávez, Ahn) — sujetos genuinamente no entrenados ni hash-gemelos del training'},
            'COMBINED_1087': {'n': len(parts['COMBINED_1087']), 'rationale': 'unión disjunta A+B+C — corpus IGT clínico limpio sin duplicación interna'},
        },
        'models_included': {
            'centaur': 'Llama-3.1-70B + Psych-101 LoRA fine-tuned (marcelbinz/Llama-3.1-Centaur-70B-adapter)',
            'llama_base': 'Llama-3.1-70B base sin adapter — control de baseline',
            'lora_noise': 'LoRA-ruido entrenado sobre ruido aleatorio (control H2 #1)',
            'lora_irrelevant': 'LoRA-irrelevante entrenado sobre dataset no-IGT no-relevante (control H2 #2)',
            'randominit_tier_c25': '[revisión interna] RandomInit — Llama-3.1-RandomInit-70B (weights aleatorios sin entrenamiento) — lower-bound sanity check',
            'vse': 'Value-plus-Sequential Exploration cognitive model (pooled-ML K=5 subject-kfold)',
            'orl': 'Outcome-Representation Learning cognitive model (pooled-ML K=5 subject-kfold)',
            'pvldelta': 'Prospect Valence Learning - Decay cognitive model (pooled-ML K=5 subject-kfold)',
        },
        'models_excluded': {
            'lora_igt': 'LoRA-IGT: entrenado sobre IGT psych101_train; por diseño anti-leakage [revisión interna] SOLO evaluado sobre 470 sujetos OOD; cobertura asimétrica en A y B → excluido de esta tabla',
            'lora_noigt': 'LoRA-noIGT: entrenado sobre no-IGT psych101_train; misma asimetría anti-leakage [revisión interna] → excluido de esta tabla',
        },
        'caches_consumed': {k: str(v.relative_to(ROOT)) for k, v in CACHES.items()},
        'llama_base_responses_cache': str(LLAMA_BASE_RESPONSES.relative_to(ROOT)),
        'cognitive_dir': str(COGNITIVE_DIR.relative_to(ROOT)),
        'manifest_extendido': str(MANIFEST.relative_to(ROOT)),
        'absolute_nll_per_partition_x_model': {},
    }

    for part_name, sids in parts.items():
        result['absolute_nll_per_partition_x_model'][part_name] = {}
        for mname, mmap in models.items():
            result['absolute_nll_per_partition_x_model'][part_name][mname] = stats_block(mmap, sids)

    # Print tabla
    print('# TABLA NLL ABSOLUTOS (mean ± SD; n_cobertura)\n')
    parts_order = ['A_TRAIN_511', 'B_HOLDOUT_66_proxy_hash_gemelos', 'C_EXTERNAL_510', 'COMBINED_1087']
    models_order = ['centaur', 'llama_base', 'lora_noise', 'lora_irrelevant', 'randominit_tier_c25', 'vse', 'orl', 'pvldelta']
    header = ['Modelo'] + parts_order
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    for m in models_order:
        row = [m]
        for p in parts_order:
            s = result['absolute_nll_per_partition_x_model'][p][m]
            if s['n'] == 0:
                row.append('—')
            else:
                row.append(f"{s['mean']:.4f} ± {s['sd']:.4f} (n={s['n']})")
        print('| ' + ' | '.join(row) + ' |')

    OUT = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout/results/aggregate_stats_clean_1087_nll_absolute_breakdown.json'
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n# Wrote: {OUT}")

    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"SHA-256: {sha}")


if __name__ == '__main__':
    main()
