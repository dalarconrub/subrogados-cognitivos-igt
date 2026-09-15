#!/usr/bin/env bash
# Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.
#
# Encadena sin intervención manual las cuatro fases del par a paridad de pasos: espera a que termine
# el entrenamiento de LoRA-IGT-40pasos, lo evalúa sobre el corpus, entrena LoRA-noIGT-40pasos y lo evalúa. Deja
# una marca en disco al terminar cada fase, de modo que si se interrumpe puede reanudarse sin repetir lo hecho.

REPO=/teamspace/studios/this_studio/ai-system-lab
LOGS=$HOME/_studio_logs
SENTINEL_DIR=$LOGS/chain_b_sentinels
mkdir -p "$SENTINEL_DIR"

cd "$REPO"
source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate lora-gpu
export PYTORCH_ALLOC_CONF=expandable_segments:True

CHAIN_LOG="$LOGS/chain_b_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[chain-b] START $(date -u +%FT%TZ)"

# === Stage 1: ESPERAR IGT_FS_DONE ===
S1_SENTINEL="$SENTINEL_DIR/stage1_igt_fs_train_done.flag"
if [ ! -f "$S1_SENTINEL" ]; then
  echo "[chain-b] Stage 1: esperando IGT_FS_DONE en logs..."
  while true; do
    if grep -q "IGT_FS_DONE" $LOGS/igt_fullscale_train_*.log 2>/dev/null; then
      echo "[chain-b] Stage 1 DONE: IGT_FS_DONE detectado"
      touch "$S1_SENTINEL"
      break
    fi
    if grep -qE "IGT_FS_ABORT|CUDA out of memory" $LOGS/igt_fullscale_train_*.log 2>/dev/null; then
      echo "[chain-b] Stage 1 ABORT"; exit 11
    fi
    sleep 60
  done
else
  echo "[chain-b] Stage 1: ya OK (sentinel existe), skip"
fi

# === Stage 2: EVAL LoRA-IGT-fullscale sobre clean_1087 ===
S2_SENTINEL="$SENTINEL_DIR/stage2_igt_fs_eval_done.flag"
if [ ! -f "$S2_SENTINEL" ]; then
  echo "[chain-b] Stage 2: lanzando eval LoRA-IGT-fullscale sobre clean_1087..."
  cd $REPO/tesis/data_analyses/llm_evaluation/paper_01_igt/stage_7_5_3_lora_igt_specialist
  python eval_lora_igt_fullscale_clean1087.py \
    --igt-data ../manifests/igt_paper1_eval_data.json \
    --whitelist-clean1087 subjects_clean1087_full.json \
    --igt-fullscale-adapter ../results/stage_7_5_3/canonical_base/lora_igt_fullscale_adapter/ \
    --base-model-id meta-llama/Llama-3.1-70B \
    --top-k 10 \
    --output ../results/stage_7_5_3/canonical_base/
  rc=$?
  if [ $rc -ne 0 ]; then echo "[chain-b] Stage 2 ABORT rc=$rc"; exit 12; fi
  touch "$S2_SENTINEL"
  echo "[chain-b] Stage 2 DONE: IGT_FS_EVAL_DONE $(date -u +%FT%TZ)"
else
  echo "[chain-b] Stage 2: ya OK (sentinel existe), skip"
fi

# === Stage 3: TRAIN LoRA-noIGT-SM40 ===
S3_SENTINEL="$SENTINEL_DIR/stage3_noigt_sm40_train_done.flag"
if [ ! -f "$S3_SENTINEL" ]; then
  echo "[chain-b] Stage 3: lanzando train LoRA-noIGT-SM40..."
  cd $REPO
  S3_LOG=$LOGS/noigt_sm40_train_$(date -u +%Y%m%dT%H%M%SZ).log
  bash tesis/data_analyses/llm_evaluation/paper_01_igt/stage_7_5_3_lora_igt_specialist/run_noigt_sm40_train.sh > $S3_LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[chain-b] Stage 3 ABORT rc=$rc see $S3_LOG"; exit 13; fi
  if ! grep -q "NOIGT_SM40_DONE" $S3_LOG; then echo "[chain-b] Stage 3 ABORT: no NOIGT_SM40_DONE en $S3_LOG"; exit 14; fi
  touch "$S3_SENTINEL"
  echo "[chain-b] Stage 3 DONE: NOIGT_SM40_DONE $(date -u +%FT%TZ)"
else
  echo "[chain-b] Stage 3: ya OK (sentinel existe), skip"
fi

# === Stage 4: EVAL LoRA-noIGT-SM40 sobre clean_1087 ===
S4_SENTINEL="$SENTINEL_DIR/stage4_noigt_sm40_eval_done.flag"
if [ ! -f "$S4_SENTINEL" ]; then
  echo "[chain-b] Stage 4: lanzando eval LoRA-noIGT-SM40 sobre clean_1087..."
  cd $REPO/tesis/data_analyses/llm_evaluation/paper_01_igt/stage_7_5_3_lora_igt_specialist
  python eval_lora_noigt_sm40_clean1087.py \
    --igt-data ../manifests/igt_paper1_eval_data.json \
    --whitelist-clean1087 subjects_clean1087_full.json \
    --noigt-sm40-adapter ../results/stage_7_5_3/canonical_base/lora_noigt_sm40_adapter/ \
    --base-model-id meta-llama/Llama-3.1-70B \
    --top-k 10 \
    --output ../results/stage_7_5_3/canonical_base/
  rc=$?
  if [ $rc -ne 0 ]; then echo "[chain-b] Stage 4 ABORT rc=$rc"; exit 15; fi
  touch "$S4_SENTINEL"
  echo "[chain-b] Stage 4 DONE: NOIGT_SM40_EVAL_DONE $(date -u +%FT%TZ)"
else
  echo "[chain-b] Stage 4: ya OK (sentinel existe), skip"
fi

echo "[chain-b] CHAIN_B_COMPLETE $(date -u +%FT%TZ)"
touch "$SENTINEL_DIR/CHAIN_B_COMPLETE.flag"
