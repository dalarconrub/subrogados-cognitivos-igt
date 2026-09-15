#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: procedimiento paso a paso.

Escribe la media (y su desviación típica) de la verosimilitud negativa de cada modelo de lenguaje y
cognitivo en cada partición; es el insumo con el que se ordenan los modelos frente a la base.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dedup_common
ROOT = Path("<REPO>")
V2 = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout/compute_nll_absolute_clean_1087_v2.py"
canon_mode = "--canon" in sys.argv
OUT = Path(__file__).with_name("nll_breakdown_v2_canon_1087_replica.json" if canon_mode else "nll_breakdown_v2_1041.json")
sys.path.insert(0, str(V2.parent))
import compute_nll_absolute_clean_1087 as base
dedup_common.patch_build_partitions(base, enabled=not canon_mode)
base.CACHES['randominit_tier_c25'] = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_h2_topk_recompute/randominit/nll_randominit__marcelbinz__Llama-3_1-RandomInit-70B_h2_recompute.json'  # cache canon post-patch top-K (mismo que el módulo LOEO)
base.COGNITIVE_DIR = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout/results/h1_pooled_ml_subject_kfold_extended'  # mismo fix que el módulo LOEO (kfold_fixed no existe)
src = V2.read_text(encoding="utf-8")
old = "OUT = ROOT / 'tesis/data_analyses/llm_evaluation/paper_01_igt/anexo_binz_holdout/results/aggregate_stats_clean_1087_nll_absolute_breakdown_v2.json'"
assert old in src, "ruta OUT del v2 no encontrada"
src = src.replace(old, f"OUT = Path({str(OUT)!r})")
g = {"__name__": "__main__", "__file__": str(V2)}
exec(compile(src, str(V2), "exec"), g)
if "build_partitions" in g and g["build_partitions"] is not base.build_partitions:
    raise SystemExit("el v2 importó build_partitions sin parche")
