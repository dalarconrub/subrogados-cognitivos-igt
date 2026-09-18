#!/usr/bin/env bash
# Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.
#
# Encadena, en una sola ejecución con GPU, la construcción del subcorpus IGT y del complemento emparejado,
# el registro de la ejecución (versiones y sumas de verificación de la configuración y los corpus), el entrenamiento de
# los dos especialistas de una época con la configuración común y su evaluación ensayo a ensayo.

set -euo pipefail   # [revisión interna]: -e para que un fallo NO siga a eval ni emita falso "LORA RUN DONE"
trap 'echo "[run] ABORT (rc=$?) $(date -u +%H:%M:%SZ)"' ERR

B=tesis/data_analyses/llm_evaluation/paper_01_igt
S=$B/stage_7_5_3_lora_igt_specialist
R=$B/results/stage_7_5_3/canonical_base
CFG=$B/stage_7_5_1_lora_noise/lora_config.yaml
RANDOMINIT=data_runtime/paper_01_igt_outputs/stage_7_5_2/randominit_sensitivity/nll_randominit__marcelbinz__Llama-3_1-RandomInit-70B.json
CN=$B/results/stage_7_5_2/canonical_base/nll_centaur.json
NN=$B/results/stage_7_5_2/canonical_base/nll_lora_noise.json
IN=$B/results/stage_7_5_2/canonical_base/nll_lora_irrelevant.json
mkdir -p "$R"

# P2.1: override opcional de max_seq_length (40GB/OOM). Default = config (32768).
SEQ_ARG=()
if [ -n "${MAX_SEQ_LENGTH:-}" ]; then SEQ_ARG=(--max-seq-length "$MAX_SEQ_LENGTH"); fi

# auth HF: acepta env HF_TOKEN O token cacheado (`huggingface-cli login`) -> el usuario
# no necesita pasar el secreto por el agente; basta con que la cuenta autentique.
python -c "from huggingface_hub import whoami; print('[run] HF user:', whoami()['name'])" 2>/dev/null \
  || { echo "ABORT: sin auth HF valida. En el Studio: 'huggingface-cli login' (o export HF_TOKEN). Acceso a Llama-3.1-70B + marcelbinz/Psych-101."; exit 1; }
echo "[run] START $(date -u +%H:%M:%SZ)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

echo "== [1/6] corpus IGT (TRAIN_psych101_original) =="
python "$S/generate_igt_corpus.py" --manifest "$B/manifests/igt_paper1_eval_data.json" --output "$R/corpus_igt.jsonl"

echo "== [2/6] corpus noIGT (Psych-101 sin IGT, scale-matched 710k) + audit =="
python "$S/generate_noigt_corpus.py" --from-hf marcelbinz/Psych-101 \
  --output "$R/corpus_noigt.jsonl" --audit-output "$R/corpus_noigt_audit.json"

echo "== [3/6] run_manifest (provenance, P2.2) =="
set +e   # la generación del manifest NO debe abortar el run
python - "$R" "$CFG" "$RANDOMINIT" "$CN" "$NN" "$IN" "${MAX_SEQ_LENGTH:-config_default}" \
  > "$R/run_manifest.json" <<'PY'
import sys, json, hashlib, subprocess, os
R, CFG, RANDOMINIT, CN, NN, IN, MAXSEQ = sys.argv[1:8]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest() if os.path.exists(p) else None
def run(c):
    try: return subprocess.check_output(c, shell=True, text=True).strip()
    except Exception: return None
vers = {}
for mod in ("torch","transformers","peft","bitsandbytes","datasets"):
    try: vers[mod] = __import__(mod).__version__
    except Exception: vers[mod] = None
m = {"git_head": run("git rev-parse HEAD"), "python": sys.version.split()[0], "which_python": run("which python"),
     "gpu": run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
     "versions": vers, "max_seq_length": MAXSEQ,
     "lora_config_sha256": sha(CFG),
     "corpus_igt_sha256": sha(f"{R}/corpus_igt.jsonl"), "corpus_noigt_sha256": sha(f"{R}/corpus_noigt.jsonl"),
     "cache_sha256": {"centaur": sha(CN), "noise": sha(NN), "irrelevant": sha(IN)},
     "randominit_present": os.path.exists(RANDOMINIT), "randominit_sha256": sha(RANDOMINIT)}
print(json.dumps(m, indent=2))
PY
[ -s "$R/run_manifest.json" ] && echo "  run_manifest -> $R/run_manifest.json" || echo "[run] WARN: run_manifest vacío/falló (no bloqueante)"
set -e

echo "== [4/6] entrenar LoRA-IGT + LoRA-noIGT (mismo recipe) =="
python "$S/train_lora_igt.py"   --corpus "$R/corpus_igt.jsonl"   --output_adapter "$R/lora_igt_adapter/"   --config "$CFG" "${SEQ_ARG[@]}"
python "$S/train_lora_noigt.py" --corpus "$R/corpus_noigt.jsonl" --output_adapter "$R/lora_noigt_adapter/" --config "$CFG" "${SEQ_ARG[@]}"

echo "== [5/6] eval disociación (NLL per-session + per-trial) leakage-free external-OOD =="
RI_ARG=()
if [ -f "$RANDOMINIT" ]; then RI_ARG=(--randominit-nll "$RANDOMINIT"); else echo "[run] nota: RandomInit floor no presente (opcional)"; fi
python "$S/eval_dissociation.py" --igt-data "$B/manifests/igt_paper1_eval_data.json" \
  --igt-adapter "$R/lora_igt_adapter/" --noigt-adapter "$R/lora_noigt_adapter/" \
  --centaur-nll "$CN" --noise-nll "$NN" --irrelevant-nll "$IN" \
  "${RI_ARG[@]}" --output "$R/"

echo "== [6/6] verdict de disociación =="
python "$S/interpret_dissociation_verdict.py" --results "$R/eval_dissociation.json"

echo ""
echo "[run] LORA RUN DONE $(date -u +%H:%M:%SZ)"
echo "Artefactos en $R/: nll_lora_{igt,noigt}.json, per_trial_lora_{igt,noigt}.json,"
echo "  corpus_{igt,noigt}.jsonl (+sha), corpus_noigt_audit.json, run_manifest.json,"
echo "  eval_dissociation.json, dissociation_verdict.md"
