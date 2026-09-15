#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: procedimiento paso a paso.

Para el par de una época y el par de cuarenta pasos, calcula la diferencia por sesión entre el adaptador
entrenado con la tarea y el entrenado con el complemento, con su media, su desviación típica, la fracción de
sesiones con Δ negativa y su d.
"""

from __future__ import annotations
import importlib.util, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dedup_common
ROOT = Path("<REPO>")
LOEO_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
CB = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base"
spec = importlib.util.spec_from_file_location("loeo_canon_mod", LOEO_MOD); m = importlib.util.module_from_spec(spec); sys.modules["loeo_canon_mod"] = m; spec.loader.exec_module(m)
base = sys.modules["compute_nll_absolute_clean_1087"]; dedup_common.patch_build_partitions(base, enabled=True); m.build_partitions = base.build_partitions
manifest = m.load_clean_manifest(); parts = m.build_partitions(manifest)
sids = parts["A_TRAIN_511"] + parts["B_HOLDOUT_66_proxy_hash_gemelos"] + parts["C_EXTERNAL_510"]; assert len(set(sids)) == 1041
models = {"lora_igt": m.load_fused(m.LORA_IGT_470, m.LORA_IGT_617), "lora_noigt": m.load_fused(m.LORA_NOIGT_470, m.LORA_NOIGT_617),
          "lora_igt_40": m.load_cache_nll(CB / "nll_lora_igt_fullscale_clean1087.json"), "lora_noigt_40": m.load_cache_nll(CB / "nll_lora_noigt_sm40_clean1087.json")}
out = {"generator": "09_intrapar_1041.py", "n_sessions": 1041, "delta_def": "NLL(LoRA-IGT) - NLL(LoRA-noIGT) por sesión; negativo = ventaja del subcorpus IGT"}
for label, a, b in [("paridad_exposicion_1epoca", "lora_igt", "lora_noigt"), ("paridad_pasos_40", "lora_igt_40", "lora_noigt_40")]:
    d = [models[a][s] - models[b][s] for s in sids if s in models[a] and s in models[b]]
    mn, sd = statistics.mean(d), statistics.stdev(d)
    out[label] = {"n": len(d), "delta_mean": mn, "delta_sd": sd, "d_paired": mn / sd, "frac_delta_neg": sum(1 for x in d if x < 0) / len(d), "n_delta_neg": sum(1 for x in d if x < 0)}
    print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[label].items()})
Path(__file__).with_name("intrapar_1041.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"); print("→ intrapar_1041.json")
