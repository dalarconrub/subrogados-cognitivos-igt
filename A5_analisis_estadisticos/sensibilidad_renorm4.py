#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: procedimiento paso a paso.

Ejecuta el módulo de métrica dual sobre las 1.041 sesiones y escribe, para los contrastes de Centaur y de
la base con los tres modelos cognitivos, la d con las dos convenciones y cuánto cambia entre ellas.
"""

import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dedup_common
ROOT = Path("<REPO>")
RENORM_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_renorm4_dual_metric_compute.py"
LOEO_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
canon_mode = "--canon" in sys.argv
OUT = Path(__file__).with_name("renorm4_canon_1087_replica.json" if canon_mode else "renorm4_1041.json")
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
loeo = load_mod("loeo_mod_for_mapping", LOEO_MOD)
base = sys.modules["compute_nll_absolute_clean_1087"]; dedup_common.patch_build_partitions(base, enabled=not canon_mode)
loeo.build_partitions = base.build_partitions
m = load_mod("renorm_canon_mod", RENORM_MOD); m.build_partitions = base.build_partitions
manifest = loeo.load_clean_manifest(); train_mapping = json.loads(loeo.TRAIN_COHORT_MAPPING.read_text())["mapping"]
by_sid_raw, _, _ = loeo.remap_experiments(manifest, train_mapping)
def study(exp):
    if exp.startswith("igt_ahn2014"): return "estudio_ahn2014"
    if exp.startswith("igt_sullivantoole2022"): return "estudio_sullivantoole2022"
    return exp
STUDY_BY_SID = {sid: study(e) for sid, e in by_sid_raw.items()}
_orig_stats = m.stats_for_contrast
def stats_14(llm_map, cog_map, sids, experiments_by_pid, metric): return _orig_stats(llm_map, cog_map, sids, STUDY_BY_SID, metric)
m.stats_for_contrast = stats_14; m.COGNITIVE_DIR = loeo.COGNITIVE_DIR_REAL
sys.argv = ["04_renorm4_1041.py", "--out", str(OUT)]
raise SystemExit(m.main())
