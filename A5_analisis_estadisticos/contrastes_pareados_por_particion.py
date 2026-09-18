#!/usr/bin/env python3
"""Anexo A, apartado A.5 · Análisis estadísticos: módulos comunes y scripts.

Para cada par de modelos y cada partición (A, B, C y Combinado) calcula la diferencia Δ de verosimilitud
negativa por sesión, su media y desviación típica, la t pareada unilateral con sus grados de libertad, la d de
Cohen con signo (media de Δ dividida por su desviación típica), la fracción de sesiones con Δ negativa y el
intervalo de confianza del 95 % de la d por bootstrap de los catorce estudios de origen (diez mil réplicas,
semilla 20260506).
"""

from __future__ import annotations
import argparse, importlib.util, json, math, random, statistics, sys
from pathlib import Path
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dedup_common

ROOT = Path("<REPO>")
LOEO_MOD = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/stage_6_2_igt_cognitive_baselines/h1_h2_h3_loeo_centaur_disaggregated.py"
CANON_JSON = ROOT / "tesis/research_records/scripts/2026-06-12_d_agregado_20_cohortes_origen/ic95_bootstrap_14_estudios.json"
CB = ROOT / "tesis/data_analyses/llm_evaluation/paper_01_igt/results/stage_7_5_3/canonical_base"
SEED, N_BOOT = 20260506, 10_000

ap = argparse.ArgumentParser(); ap.add_argument("--canon", action="store_true"); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("loeo_canon_mod", LOEO_MOD); m = importlib.util.module_from_spec(spec)
sys.modules["loeo_canon_mod"] = m; spec.loader.exec_module(m)
base = sys.modules["compute_nll_absolute_clean_1087"]
dedup_common.patch_build_partitions(base, enabled=not a.canon)
m.build_partitions = base.build_partitions

def cohen_d_paired(deltas): sd = statistics.stdev(deltas); return statistics.mean(deltas) / sd if sd else float("nan")
def t_paired(deltas):
    n = len(deltas); mn, sd = statistics.mean(deltas), statistics.stdev(deltas); return mn / (sd / math.sqrt(n)), n - 1
def cluster_bootstrap_ci_d(deltas_by_experiment, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed); experiments = list(deltas_by_experiment.keys()); n_exp = len(experiments); reps = []
    for _ in range(n_boot):
        sampled = [experiments[rng.randint(0, n_exp - 1)] for _ in range(n_exp)]
        flat = []
        for e in sampled: flat.extend(deltas_by_experiment[e])
        if len(flat) < 2: continue
        mn, sd = statistics.mean(flat), statistics.stdev(flat)
        if sd > 0: reps.append(mn / sd)
    reps.sort(); return reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps))]
def stats_for_contrast(a_nll, b_nll, sids, exp_by_sid):
    deltas, by_exp = [], {}
    for sid in sids:
        if sid not in a_nll or sid not in b_nll: continue
        d = a_nll[sid] - b_nll[sid]; deltas.append(d); by_exp.setdefault(exp_by_sid.get(sid, "UNKNOWN"), []).append(d)
    lo, hi = cluster_bootstrap_ci_d(by_exp); t, df = t_paired(deltas)
    return {"n": len(deltas), "n_experiments": len(by_exp), "delta_mean": statistics.mean(deltas), "delta_sd": statistics.stdev(deltas),
            "cohens_d": cohen_d_paired(deltas), "t_stat": t, "df": df, "p_one_sided": float(sps.t.sf(abs(t), df)),
            "ci_d_95": [lo, hi], "n_delta_neg": sum(1 for d in deltas if d < 0), "frac_delta_neg": sum(1 for d in deltas if d < 0) / len(deltas)}
def nll_block(nll, sids):
    v = [nll[s] for s in sids if s in nll]; return {"n": len(v), "mean": statistics.mean(v), "sd": statistics.stdev(v)}

manifest = m.load_clean_manifest()
manifest_no_holdout = [r for r in manifest if not r["experiment"].startswith("igt_steingroever2015_psych101_holdout")]
parts = m.build_partitions(manifest)
parts["COMBINED"] = parts["A_TRAIN_511"] + parts["B_HOLDOUT_66_proxy_hash_gemelos"] + parts["C_EXTERNAL_510"]
print({k: len(v) for k, v in parts.items()}, file=sys.stderr)
train_mapping = json.loads(m.TRAIN_COHORT_MAPPING.read_text())["mapping"]
by_sid_raw, _, _ = m.remap_experiments(manifest, train_mapping)
def study(exp):
    if exp.startswith("igt_ahn2014"): return "estudio_ahn2014"
    if exp.startswith("igt_sullivantoole2022"): return "estudio_sullivantoole2022"
    return exp
exp_by_sid = {sid: study(e) for sid, e in by_sid_raw.items()}
models = {"centaur": m.load_cache_nll(m.CACHES["centaur"]), "llama_base": m.load_llama_base_nll(manifest_no_holdout)}
for mn, fn in [("vse","vse.jsonl"),("orl","orl.jsonl"),("pvldelta","pvldelta.jsonl")]: models[mn] = m.load_cognitive_nll(m.COGNITIVE_DIR_REAL / fn)
models["lora_noise"] = m.load_cache_nll(m.CACHES["lora_noise"]); models["lora_irrelevant"] = m.load_cache_nll(m.CACHES["lora_irrelevant"])
models["randominit"] = m.load_cache_nll(m.RANDOMINIT_CACHE)
models["lora_igt"] = m.load_fused(m.LORA_IGT_470, m.LORA_IGT_617); models["lora_noigt"] = m.load_fused(m.LORA_NOIGT_470, m.LORA_NOIGT_617)
models["lora_noigt_fullscale"] = m.load_cache_nll(m.NLL_LORA_NOIGT_FULLSCALE)
models["lora_igt_fullscale"] = m.load_cache_nll(CB / "nll_lora_igt_fullscale_clean1087.json"); models["lora_noigt_sm40"] = m.load_cache_nll(CB / "nll_lora_noigt_sm40_clean1087.json")
CONTRASTS = [("centaur_vs_llama_base","centaur","llama_base"),("centaur_vs_vse","centaur","vse"),("centaur_vs_orl","centaur","orl"),("centaur_vs_pvldelta","centaur","pvldelta"),
 ("vse_vs_llama_base","vse","llama_base"),("orl_vs_llama_base","orl","llama_base"),("pvldelta_vs_llama_base","pvldelta","llama_base"),
 ("centaur_vs_lora_noise","centaur","lora_noise"),("centaur_vs_lora_irrelevant","centaur","lora_irrelevant"),("centaur_vs_randominit","centaur","randominit"),
 ("lora_noise_vs_llama_base","lora_noise","llama_base"),("lora_irrelevant_vs_llama_base","lora_irrelevant","llama_base"),("randominit_vs_llama_base","randominit","llama_base"),
 ("centaur_vs_lora_igt","centaur","lora_igt"),("centaur_vs_lora_noigt","centaur","lora_noigt"),("centaur_vs_lora_noigt_fullscale","centaur","lora_noigt_fullscale"),
 ("centaur_vs_lora_igt_fullscale","centaur","lora_igt_fullscale"),("centaur_vs_lora_noigt_sm40","centaur","lora_noigt_sm40"),
 ("lora_igt_vs_llama_base","lora_igt","llama_base"),("lora_noigt_vs_llama_base","lora_noigt","llama_base"),("lora_noigt_fullscale_vs_llama_base","lora_noigt_fullscale","llama_base"),
 ("lora_igt_fullscale_vs_llama_base","lora_igt_fullscale","llama_base"),("lora_noigt_sm40_vs_llama_base","lora_noigt_sm40","llama_base")]
PARTS = ["A_TRAIN_511", "B_HOLDOUT_66_proxy_hash_gemelos", "C_EXTERNAL_510", "COMBINED"]
out = {"generator": "tesis/research_records/scripts/2026-09-03_dedup_1041_sullivantoole/01_stats_1041.py", "dedup": not a.canon,
       "partition_sizes": {p: len(parts[p]) for p in PARTS}, "seed": SEED, "n_boot": N_BOOT, "cluster": "14 estudios de origen", "contrasts": {}, "nll_absolute": {}}
for p in PARTS:
    out["nll_absolute"][p] = {mn: nll_block(models[mn], parts[p]) for mn in models}
for name, x, y in CONTRASTS:
    out["contrasts"][name] = {"a": x, "b": y, "partitions": {p: stats_for_contrast(models[x], models[y], parts[p], exp_by_sid) for p in PARTS}}
    print(name, {p: (v["n"], round(v["delta_mean"], 4), round(v["cohens_d"], 3)) for p, v in out["contrasts"][name]["partitions"].items()}, file=sys.stderr)
if a.canon:
    canon = json.loads(CANON_JSON.read_text())["contrasts"]; alias = {"centaur_vs_randominit": "centaur_vs_randominit_tier_c25", "randominit_vs_llama_base": "randominit_tier_c25_vs_llama_base"}
    partmap = {"COMBINED": "COMBINED_1087"}; ok = bad = 0
    for name, c in out["contrasts"].items():
        cn = canon.get(alias.get(name, name))
        if cn is None: print("  sin canon:", name, file=sys.stderr); continue
        for p, v in c["partitions"].items():
            cv = cn["partitions"][partmap.get(p, p)]
            same = v["n"] == cv["n"] and abs(v["delta_mean"] - cv["delta_mean"]) < 1e-12 and abs(v["cohens_d"] - cv["cohens_d"]) < 1e-12 and all(abs(x - y) < 1e-12 for x, y in zip(v["ci_d_95"], cv["ci_d_95"]))
            ok += same; bad += (not same)
            if not same: print("  DIFIERE:", name, p, v["n"], cv["n"], v["cohens_d"], cv["cohens_d"], v["ci_d_95"], cv["ci_d_95"], file=sys.stderr)
    print(f"PUERTA DE VALIDACIÓN contra el canon: {ok} celdas idénticas, {bad} distintas")
    Path(__file__).with_name("stats_canon_1087_replica.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
else:
    Path(__file__).with_name("stats_1041.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print("✓ stats_1041.json")
