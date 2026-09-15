#!/usr/bin/env bash
# Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.
#
# Variante del lanzador anterior que retoma el entrenamiento donde se quedó (salta el entrenamiento si el
# adaptador ya existe, reanuda desde el último punto de control si quedó a medias) y evalúa al terminar.

source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate lora-gpu
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /teamspace/studios/this_studio/ai-system-lab

B=tesis/data_analyses/llm_evaluation/paper_01_igt
S=$B/stage_7_5_3_lora_igt_specialist
R=$B/results/stage_7_5_3/canonical_base
CFG=$B/stage_7_5_1_lora_noise/lora_config_resumable.yaml
CORPUS=$R/corpus_noigt_chunked.jsonl
RANDOMINIT=data_runtime/paper_01_igt_outputs/stage_7_5_2/randominit_sensitivity/nll_randominit__marcelbinz__Llama-3_1-RandomInit-70B.json
CN=$B/results/stage_7_5_2/canonical_base/nll_centaur.json
NN=$B/results/stage_7_5_2/canonical_base/nll_lora_noise.json
IN=$B/results/stage_7_5_2/canonical_base/nll_lora_irrelevant.json

echo "[noigt-pipe] START $(date -u +%FT%TZ)  corpus=$(basename $CORPUS)  cfg=$(basename $CFG)"
echo "[noigt-pipe] GPU=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"

# ---- 1. TRAIN noIGT (resumible) ----
if [ -f "$R/lora_noigt_adapter/adapter_model.safetensors" ]; then
  echo "[noigt-pipe] adapter noIGT ya existe -> SALTO train"
else
  CKPT_DIR="$R/lora_noigt_adapter/_training_logs"
  LATEST=""
  [ -d "$CKPT_DIR" ] && LATEST=$(ls -1d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
  RESUME=()
  if [ -n "$LATEST" ]; then echo "[noigt-pipe] RESUME desde $LATEST"; RESUME=(--resume_from_checkpoint "$LATEST")
  else echo "[noigt-pipe] train desde cero"; fi
  python "$S/train_lora_noigt.py" --corpus "$CORPUS" \
    --output_adapter "$R/lora_noigt_adapter/" --config "$CFG" "${RESUME[@]}"
  rc=$?
  if [ $rc -ne 0 ]; then echo "[noigt-pipe] ABORT train rc=$rc (re-lanzar launcher para resumir)"; exit $rc; fi
  echo "[noigt-pipe] train noIGT OK"
fi

# ---- 2. EVAL dissociation ----
if [ -f "$R/eval_dissociation.json" ]; then
  echo "[noigt-pipe] eval_dissociation.json ya existe -> SALTO eval"
else
  RI=()
  [ -f "$RANDOMINIT" ] && RI=(--randominit-nll "$RANDOMINIT") || echo "[noigt-pipe] nota: RandomInit floor no presente (opcional)"
  # --igt-nll reusa la cache IGT si existe (salta su carga GPU -> evita 2 modelos en VRAM
  # a la vez, que OOMea; si no existe, eval la computa fresh). Robustez OOM 2026-05-23.
  IGT_CACHE=()
  [ -f "$R/nll_lora_igt.json" ] && IGT_CACHE=(--igt-nll "$R/nll_lora_igt.json")
  python "$S/eval_dissociation.py" --igt-data "$B/manifests/igt_paper1_eval_data.json" \
    --igt-adapter "$R/lora_igt_adapter/" --noigt-adapter "$R/lora_noigt_adapter/" \
    --centaur-nll "$CN" --noise-nll "$NN" --irrelevant-nll "$IN" \
    "${IGT_CACHE[@]}" "${RI[@]}" --output "$R/"
  rc=$?
  if [ $rc -ne 0 ]; then echo "[noigt-pipe] ABORT eval rc=$rc"; exit $rc; fi
  echo "[noigt-pipe] eval_dissociation OK"
fi

# ---- 3. VERDICT ----
python "$S/interpret_dissociation_verdict.py" --results "$R/eval_dissociation.json"
echo ""
echo "[noigt-pipe] NOIGT PIPELINE DONE $(date -u +%FT%TZ)"
echo "Artefactos en $R/: lora_noigt_adapter/, nll_lora_{igt,noigt}.json, per_trial_lora_{igt,noigt}.json, eval_dissociation.json, dissociation_verdict.md"
