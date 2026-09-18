#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: módulos comunes y scripts.

Compara, sesión a sesión, la verosimilitud negativa de Centaur recalculada desde las probabilidades por
ensayo guardadas con la de la caché agregada de la inferencia, y mide cuánto desplaza esa diferencia la d del
contraste con VSE.
"""

import importlib.util, json, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import dedup_common
ROOT = HERE.parents[3]
LOEO_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
RENORM_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_renorm4_dual_metric_compute.py"
def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
m = load("loeo_canon_mod", LOEO_MOD); base = sys.modules["compute_nll_absolute_clean_1087"]; dedup_common.patch_build_partitions(base, enabled=True); m.build_partitions = base.build_partitions
manifest = m.load_clean_manifest(); parts = m.build_partitions(manifest)
sids = parts["A_TRAIN_511"] + parts["B_HOLDOUT_66_proxy_hash_gemelos"] + parts["C_EXTERNAL_510"]
prim = m.load_cache_nll(m.CACHES["centaur"]); vse = m.load_cognitive_nll(m.COGNITIVE_DIR_REAL / "vse.jsonl")
r = load("renorm_canon_mod", RENORM_MOD); r.build_partitions = base.build_partitions
cen = r.compute_renorm4_per_subject(r.CENTAUR_TOPK, r.load_prompt_ids(r.CENTAUR_AGG), r.load_deck_maps(), "Centaur")
common = [s for s in sids if s in prim and s in cen]
diffs = [cen[s]["nll_raw"] - prim[s] for s in common]
def d(a, b):
    dl = [a[s] - b[s] for s in common if s in b]; return statistics.mean(dl) / statistics.stdev(dl), statistics.mean(dl), statistics.stdev(dl)
out = {"n_sesiones": len(common), "n_ensayos_renorm": sum(cen[s]["n_trials"] for s in common), "n_ensayos_excluidos_centaur": sum(cen[s]["n_excluded"] for s in common),
       "nll_raw_menos_primaria": {"media": statistics.mean(diffs), "mediana": statistics.median(diffs), "dt": statistics.stdev(diffs), "min": min(diffs), "max": max(diffs), "sesiones_con_dif_mayor_1e-4": sum(abs(x) > 1e-4 for x in diffs)},
       "d_centaur_vs_vse": {"primaria": d(prim, vse), "nll_raw": d({s: cen[s]["nll_raw"] for s in common}, vse)},
       "conclusion": "Mismas sesiones y ensayos, sin exclusiones para Centaur; la caché por ensayo (K logprobs en bfloat16) y la caché agregada del pase primario difieren en milésimas de nat por sesión, del orden de la resolución de bfloat16 en torno a 1 nat, lo que desplaza d en ≈ 0,03 porque la DT de Δ es ≈ 0,16."}
Path(__file__).with_name("diagnostico_nota56_1041.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out, indent=1, ensure_ascii=False)[:600])
