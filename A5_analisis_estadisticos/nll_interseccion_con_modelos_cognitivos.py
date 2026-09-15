#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: procedimiento paso a paso.

Para los contrastes de Centaur con VSE, ORL y PVL-Δ, calcula la verosimilitud negativa media de ambos modelos
sobre exactamente las mismas sesiones, para que las medias de las Tablas 5.2, 5.3 y 5.4 se refieran al mismo conjunto.
"""

import importlib.util, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dedup_common
ROOT = Path("<REPO>")
LOEO_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
spec = importlib.util.spec_from_file_location("loeo_canon_mod", LOEO_MOD); m = importlib.util.module_from_spec(spec); sys.modules["loeo_canon_mod"] = m; spec.loader.exec_module(m)
base = sys.modules["compute_nll_absolute_clean_1087"]; orig = base.build_partitions
manifest = m.load_clean_manifest(); mnh = [r for r in manifest if not r["experiment"].startswith("igt_steingroever2015_psych101_holdout")]
cen = m.load_cache_nll(m.CACHES["centaur"]); cog = {mn: m.load_cognitive_nll(m.COGNITIVE_DIR_REAL / fn) for mn, fn in [("vse","vse.jsonl"),("orl","orl.jsonl"),("pvldelta","pvldelta.jsonl")]}
out = {}
for tag, enabled in (("1087", False), ("1041", True)):
    base.build_partitions = orig; dedup_common.patch_build_partitions(base, enabled=enabled)
    parts = base.build_partitions(manifest); parts["COMBINED"] = parts["A_TRAIN_511"] + parts["B_HOLDOUT_66_proxy_hash_gemelos"] + parts["C_EXTERNAL_510"]
    out[tag] = {}
    for key, comp in [("centaur_vs_vse","vse"),("centaur_vs_orl","orl"),("centaur_vs_pvldelta","pvldelta")]:
        out[tag][key] = {}
        for p, sids in parts.items():
            inter = [s for s in sids if s in cen and s in cog[comp]]
            out[tag][key][p] = {"n": len(inter), "centaur": statistics.mean(cen[s] for s in inter), comp: statistics.mean(cog[comp][s] for s in inter)}
    print(tag, {k: {p: (v["n"], round(v["centaur"], 3)) for p, v in d.items()} for k, d in out[tag].items()})
Path(__file__).with_name("nll_interseccion.json").write_text(json.dumps(out, indent=1))
print("✓ nll_interseccion.json")
