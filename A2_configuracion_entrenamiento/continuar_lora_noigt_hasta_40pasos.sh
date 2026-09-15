#!/usr/bin/env bash
# Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.
#
# Reanuda el entrenamiento de LoRA-noIGT-1época desde su punto de control del paso 7 y lo lleva hasta 40
# pasos con la configuración correspondiente, en un adaptador aparte.

REPO="${REPO:-/teamspace/studios/this_studio/ai-system-lab}"
ENVNAME="${ENVNAME:-lora-gpu}"
source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVNAME"
set -uo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$REPO/tesis/data_analyses/llm_evaluation/paper_01_igt/stage_7_5_3_lora_igt_specialist"

BASE=../results/stage_7_5_3/canonical_base
CORPUS="$BASE/corpus_noigt.jsonl"
ADAPTER="$BASE/lora_noigt_sm40_adapter"
CONFIG=../stage_7_5_1_lora_noise/lora_config_noigt_sm40.yaml
MAXSEQ="${MAXSEQ:-8192}"

echo "[noigt-sm40] START $(date -u +%FT%TZ)  maxseq=$MAXSEQ"
[ -f "$CORPUS" ] || { echo "[noigt-sm40] FALTA $CORPUS"; exit 2; }

# Seed: copiar checkpoint-7 del LoRA-noIGT original al dir sm40 (no contamina original)
ORIG_CKPT="$BASE/lora_noigt_adapter/_training_logs/checkpoint-7"
SM40_CKPT_DIR="$ADAPTER/_training_logs"
if [ ! -d "$SM40_CKPT_DIR/checkpoint-7" ] && [ -d "$ORIG_CKPT" ]; then
  echo "[noigt-sm40] sembrado: $ORIG_CKPT -> $SM40_CKPT_DIR/checkpoint-7"
  mkdir -p "$SM40_CKPT_DIR"
  cp -r "$ORIG_CKPT" "$SM40_CKPT_DIR/checkpoint-7"
fi

LAST=$(ls -d "$SM40_CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
RESUME_ARG=""
if [ -n "$LAST" ]; then echo "[noigt-sm40] resume $LAST"; RESUME_ARG="--resume_from_checkpoint $LAST"; fi

python train_lora_noigt.py --corpus "$CORPUS" --output_adapter "$ADAPTER" --config "$CONFIG" --max-seq-length "$MAXSEQ" $RESUME_ARG
rc=$?
echo "[noigt-sm40] train rc=$rc $(date -u +%FT%TZ)"
if [ $rc -ne 0 ]; then echo "[noigt-sm40] NOIGT_SM40_ABORT rc=$rc"; exit $rc; fi
echo "[noigt-sm40] NOIGT_SM40_DONE adapter -> $ADAPTER"
