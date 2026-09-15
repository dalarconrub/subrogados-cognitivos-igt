#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: procedimiento paso a paso.

Para cada contraste y partición, cambia al azar el signo de las diferencias pareadas diez mil veces con
la semilla del estudio, calcula la media en cada permutación y sitúa la media observada en esa distribución; el
valor p unilateral es la proporción de permutaciones al menos tan extremas como lo observado en la dirección
declarada. Calcula además, como descripción complementaria, el intervalo de la d por bootstrap de sesiones.
"""

import importlib.util, json, statistics, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dedup_common
ROOT = Path("<REPO>")
LOEO_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
CB = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base"
SEED, NPERM = 20260506, 10_000
spec = importlib.util.spec_from_file_location("loeo_canon_mod", LOEO_MOD); m = importlib.util.module_from_spec(spec); sys.modules["loeo_canon_mod"] = m; spec.loader.exec_module(m)
base = sys.modules["compute_nll_absolute_clean_1087"]; dedup_common.patch_build_partitions(base); m.build_partitions = base.build_partitions
manifest = m.load_clean_manifest(); mnh = [r for r in manifest if not r["experiment"].startswith("igt_steingroever2015_psych101_holdout")]
parts = m.build_partitions(manifest); parts["COMBINED"] = parts["A_TRAIN_511"] + parts["B_HOLDOUT_66_proxy_hash_gemelos"] + parts["C_EXTERNAL_510"]
models = {"centaur": m.load_cache_nll(m.CACHES["centaur"]), "llama_base": m.load_llama_base_nll(mnh)}
for mn, fn in [("vse","vse.jsonl"),("orl","orl.jsonl"),("pvldelta","pvldelta.jsonl")]: models[mn] = m.load_cognitive_nll(m.COGNITIVE_DIR_REAL / fn)
models["lora_noise"] = m.load_cache_nll(m.CACHES["lora_noise"]); models["lora_irrelevant"] = m.load_cache_nll(m.CACHES["lora_irrelevant"]); models["randominit"] = m.load_cache_nll(m.RANDOMINIT_CACHE)
models["lora_igt"] = m.load_fused(m.LORA_IGT_470, m.LORA_IGT_617); models["lora_noigt"] = m.load_fused(m.LORA_NOIGT_470, m.LORA_NOIGT_617)
models["lora_noigt_fullscale"] = m.load_cache_nll(m.NLL_LORA_NOIGT_FULLSCALE); models["lora_igt_fullscale"] = m.load_cache_nll(CB / "nll_lora_igt_fullscale_clean1087.json"); models["lora_noigt_sm40"] = m.load_cache_nll(CB / "nll_lora_noigt_sm40_clean1087.json")
S = json.loads((Path(__file__).parent / "stats_1041.json").read_text())
out = {"seed": SEED, "n_perm": NPERM, "method": "permutación del signo unilateral: proporción de permutaciones con media ≤ media observada (o ≥ si la observada es positiva); mínimo informable 1/(n_perm+1)", "contrasts": {}}
for name, c in S["contrasts"].items():
    a, b = c["a"], c["b"]; out["contrasts"][name] = {}
    for p, sids in parts.items():
        d = np.array([models[a][s] - models[b][s] for s in sids if s in models[a] and s in models[b]]); obs = d.mean()
        rng = np.random.default_rng(SEED); signs = rng.choice([-1.0, 1.0], size=(NPERM, len(d))); perm = (signs * d).mean(axis=1)
        p_perm = ((perm <= obs).sum() if obs < 0 else (perm >= obs).sum()) / NPERM
        rng2 = np.random.default_rng(SEED); idx = rng2.integers(0, len(d), size=(NPERM, len(d))); bd = d[idx]; dboot = bd.mean(axis=1) / bd.std(axis=1, ddof=1)
        out["contrasts"][name][p] = {"n": int(len(d)), "p_permutacion": float(p_perm), "p_reportable": "< ,001" if p_perm < 0.001 else f"{p_perm:.4f}", "ci_d_bootstrap_sesion": [float(np.percentile(dboot, 2.5)), float(np.percentile(dboot, 97.5))]}
Path(__file__).with_name("permutacion_signo_1041.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
peor = max((v["p_permutacion"], k, p) for k, c in out["contrasts"].items() for p, v in c.items())
print("23 contrastes × 4 particiones; p máximo:", peor); print("ejemplo primario:", out["contrasts"]["centaur_vs_llama_base"]["COMBINED"])
