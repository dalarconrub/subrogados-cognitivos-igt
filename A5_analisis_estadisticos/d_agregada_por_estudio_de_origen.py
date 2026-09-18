#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: módulos comunes y scripts.

Calcula la media de Δ (Centaur menos base) dentro de cada uno de los catorce estudios de origen y, sobre
esas catorce medias, su media dividida por su desviación típica, con un intervalo por bootstrap de estudios. Es la
cifra que se compara con la del estudio que introdujo Centaur.
"""

import importlib.util, json, statistics, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dedup_common
ROOT = Path("<REPO>")
LOEO_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
canon_mode = "--canon" in sys.argv; SEED, N_BOOT = 20260506, 10_000
spec = importlib.util.spec_from_file_location("loeo_canon_mod", LOEO_MOD); m = importlib.util.module_from_spec(spec); sys.modules["loeo_canon_mod"] = m; spec.loader.exec_module(m)
base = sys.modules["compute_nll_absolute_clean_1087"]; dedup_common.patch_build_partitions(base, enabled=not canon_mode); m.build_partitions = base.build_partitions
manifest = m.load_clean_manifest(); mnh = [r for r in manifest if not r["experiment"].startswith("igt_steingroever2015_psych101_holdout")]
parts = m.build_partitions(manifest); comb = parts["A_TRAIN_511"] + parts["B_HOLDOUT_66_proxy_hash_gemelos"] + parts["C_EXTERNAL_510"]
train_mapping = json.loads(m.TRAIN_COHORT_MAPPING.read_text())["mapping"]; by_sid_raw, _, _ = m.remap_experiments(manifest, train_mapping)
def study(exp):
    if exp.startswith("igt_ahn2014"): return "Ahn 2014"
    if exp.startswith("igt_sullivantoole2022"): return "Sullivan-Toole 2022"
    return exp.replace("igt_", "")
cen = m.load_cache_nll(m.CACHES["centaur"]); base_nll = m.load_llama_base_nll(mnh)
by_study = {}
for s in comb: by_study.setdefault(study(by_sid_raw[s]), []).append(cen[s] - base_nll[s])
assert len(by_study) == 14, len(by_study)
deltas14 = [statistics.mean(v) for v in by_study.values()]
v = np.asarray(deltas14); sd = np.std(v, ddof=1); d = float(np.mean(v) / sd)
rng = np.random.default_rng(SEED); boots = []
for _ in range(N_BOOT):
    b = v[rng.integers(0, len(v), len(v))]; s = np.std(b, ddof=1)
    if s > 0: boots.append(float(np.mean(b) / s))
lo, hi = np.percentile(boots, [2.5, 97.5])
out = {"dedup": not canon_mode, "n_sesiones": len(comb), "n_estudios": 14, "delta_media_agregada": float(np.mean(v)), "sd_inter_estudio": float(sd),
       "d_agregada": d, "ci95": [float(lo), float(hi)], "estudios_favorecen": f"{int(np.sum(v < 0))}/14",
       "por_estudio": {k: {"n": len(vv), "delta_mean": statistics.mean(vv)} for k, vv in by_study.items()}}
Path(__file__).with_name("d_agregada_canon_1087_replica.json" if canon_mode else "d_agregada_1041.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"n={out['n_sesiones']} Δ agregada={out['delta_media_agregada']:.4f} sd={sd:.4f} d={d:.3f} [{lo:.2f}; {hi:.2f}] {out['estudios_favorecen']}")
if canon_mode:
    ref = json.loads((ROOT / "tesis/research_records/scripts/2026-06-12_d_agregado_20_cohortes_origen/d_agregado_por_particion_y_niveles.json").read_text())
    print("canon 14 estudios en JSON:", json.dumps(ref.get("total_por_convencion", ref).get("14 estudios", ref.get("total_por_convencion", "?")))[:300] if isinstance(ref, dict) else ref)
