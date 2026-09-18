#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: módulos comunes y scripts.

Calcula, sobre las 1.041 sesiones, la masa media de probabilidad que Centaur y la base asignan a los
cuatro tokens de mazo de cada sesión en cada ensayo.
"""

import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dedup_common
ROOT = Path("<REPO>")
RENORM_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_renorm4_dual_metric_compute.py"
LOEO_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
OUT = Path(__file__).with_name("masa_1041.json")
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
loeo = load_mod("loeo_mod_for_mapping", LOEO_MOD)
base = sys.modules["compute_nll_absolute_clean_1087"]; dedup_common.patch_build_partitions(base, enabled=True)
m = load_mod("renorm_canon_mod", RENORM_MOD); m.build_partitions = base.build_partitions
manifest = m.load_clean_manifest(); parts = m.build_partitions(manifest)
sids_1041 = set(parts["A_TRAIN_511"] + parts["B_HOLDOUT_66_proxy_hash_gemelos"] + parts["C_EXTERNAL_510"]); assert len(sids_1041) == 1041, len(sids_1041)
deck_maps = m.load_deck_maps(); pids = m.load_prompt_ids(m.CENTAUR_AGG)
out = {"generator": "04b_masa_1041.py", "n_sessions": 1041, "source": "mismas cachés por ensayo que renorm4_1041.json (D5); media entre sesiones de la masa media por ensayo sobre los cuatro tokens-mazo"}
for label, cache in [("centaur", m.CENTAUR_TOPK), ("llama_base", m.LLAMA_TOPK)]:
    per = m.compute_renorm4_per_subject(cache, pids, deck_maps, label)
    sel = {k: v for k, v in per.items() if k in sids_1041}
    missing = sids_1041 - set(per)
    out[label] = {"n_subjects_in_cache": len(per), "n_sessions_1041_found": len(sel), "n_missing": len(missing),
                  "mass_on_4_mean_across_sessions_1041": sum(s["mass_on_4_mean"] for s in sel.values()) / len(sel),
                  "mass_on_4_mean_across_sessions_1598": sum(s["mass_on_4_mean"] for s in per.values()) / len(per),
                  "n_trials_excluded_1041": sum(s["n_excluded"] for s in sel.values()), "n_trials_kept_1041": sum(s["n_kept"] for s in sel.values())}
    print(label, out[label])
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"); print("→", OUT)
