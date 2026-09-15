#!/usr/bin/env bash
# Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.
#
# Reanuda el entrenamiento de LoRA-IGT-1época desde su punto de control del paso 15 y lo lleva hasta 40
# pasos con la configuración correspondiente, guardando el resultado como un adaptador distinto para no alterar el original.

REPO="${REPO:-/teamspace/studios/this_studio/ai-system-lab}"
ENVNAME="${ENVNAME:-lora-gpu}"
source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVNAME"
set -uo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$REPO/tesis/data_analyses/llm_evaluation/paper_01_igt/stage_7_5_3_lora_igt_specialist"

BASE=../results/stage_7_5_3/canonical_base
CORPUS="$BASE/corpus_igt.jsonl"
ADAPTER="$BASE/lora_igt_fullscale_adapter"
CONFIG=../stage_7_5_1_lora_noise/lora_config_igt_fullscale.yaml
MAXSEQ="${MAXSEQ:-8192}"

echo "[igt-fs] START $(date -u +%FT%TZ)  repo=$REPO env=$ENVNAME maxseq=$MAXSEQ"
[ -f "$CORPUS" ] || { echo "[igt-fs] FALTA $CORPUS"; exit 2; }

# Seed: copiar checkpoint-15 del LoRA-IGT original al directorio fullscale para reanudar (no contamina el original)
ORIG_CKPT="$BASE/lora_igt_adapter/_training_logs/checkpoint-15"
FS_CKPT_DIR="$ADAPTER/_training_logs"
if [ ! -d "$FS_CKPT_DIR/checkpoint-15" ] && [ -d "$ORIG_CKPT" ]; then
  echo "[igt-fs] sembrado: copiando $ORIG_CKPT -> $FS_CKPT_DIR/checkpoint-15"
  mkdir -p "$FS_CKPT_DIR"
  cp -r "$ORIG_CKPT" "$FS_CKPT_DIR/checkpoint-15"
fi

# Auto-resume: último checkpoint-N
LAST=$(ls -d "$FS_CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
RESUME_ARG=""
if [ -n "$LAST" ]; then
  echo "[igt-fs] reanudando desde $LAST"
  RESUME_ARG="--resume_from_checkpoint $LAST"
else
  echo "[igt-fs] sin checkpoint previo -> entreno desde cero"
fi

# Entrenar (mismo recipe que LoRA-IGT original; única variable vs original = max_steps 15→40)
python train_lora_igt.py \
  --corpus "$CORPUS" \
  --output_adapter "$ADAPTER" \
  --config "$CONFIG" \
  --max-seq-length "$MAXSEQ" \
  $RESUME_ARG
rc=$?
echo "[igt-fs] train rc=$rc $(date -u +%FT%TZ)"
if [ $rc -ne 0 ]; then echo "[igt-fs] IGT_FS_ABORT rc=$rc"; exit $rc; fi
echo "[igt-fs] IGT_FS_DONE adapter -> $ADAPTER"
