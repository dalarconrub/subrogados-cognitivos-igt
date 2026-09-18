#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: módulos comunes y scripts.

Ejecuta el módulo de exclusión sucesiva sobre las 1.041 sesiones y escribe, por contraste, la d
completa, la mediana y el rango de las catorce d con un estudio retirado, cuántas conservan el signo y el estudio
de mayor influencia.
"""

import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dedup_common
ROOT = Path("<REPO>")
CANON = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
canon_mode = "--canon" in sys.argv
OUT = Path(__file__).with_name("loeo_canon_1087_replica.json" if canon_mode else "loeo_1041.json")
spec = importlib.util.spec_from_file_location("loeo_canon", CANON); m = importlib.util.module_from_spec(spec); sys.modules["loeo_canon"] = m; spec.loader.exec_module(m)
base = sys.modules["compute_nll_absolute_clean_1087"]; dedup_common.patch_build_partitions(base, enabled=not canon_mode); m.build_partitions = base.build_partitions
_orig_remap = m.remap_experiments
def remap_14(manifest, train_mapping):
    by_sid, n_train, n_other = _orig_remap(manifest, train_mapping)
    def study(exp):
        if exp.startswith("igt_ahn2014"): return "estudio_ahn2014"
        if exp.startswith("igt_sullivantoole2022"): return "estudio_sullivantoole2022"
        return exp
    return {sid: study(e) for sid, e in by_sid.items()}, n_train, n_other
m.remap_experiments = remap_14
sys.argv = ["02_loeo_1041.py", "--out", str(OUT)]; m.main()
res = json.loads(OUT.read_text())
for c in res["contrasts"]:
    print(f"{c['contrast']:36} n={c['n_primary']} d={c['d_primary']:.3f} med={c['d_median_loeo']:.3f} [{c['d_min_loeo']:.3f}; {c['d_max_loeo']:.3f}] {c['n_iter_signed_preserved']}/{c['n_iter_total']} infl={c['most_influential_experiment']} ({c['most_influential_deviation']:.3f})")
if canon_mode:
    ref = {c["contrast"]: c for c in json.loads((ROOT / "tesis/research_records/scripts/2026-06-12_d_agregado_20_cohortes_origen/loeo_14_estudios.json").read_text())["contrasts"]}
    bad = [c["contrast"] for c in res["contrasts"] if abs(c["d_primary"] - ref[c["contrast"]]["d_primary"]) > 1e-12 or abs(c["d_min_loeo"] - ref[c["contrast"]]["d_min_loeo"]) > 1e-12]
    print("PUERTA LOEO:", "OK bit-idéntico" if not bad else f"DIFIERE {bad}")
