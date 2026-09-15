#!/usr/bin/env bash
# Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.
#
# Lanza de principio a fin la preparación del complemento íntegro (construcción y troceado a 8.192
# tokens) y su entrenamiento reanudable; es el lanzador del script de entrenamiento del complemento íntegro.

source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate lora-gpu
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /teamspace/studios/this_studio/ai-system-lab
B=tesis/data_analyses/llm_evaluation/paper_01_igt
S=$B/stage_7_5_3_lora_igt_specialist
R=$B/results/stage_7_5_3/canonical_base
CFG=$B/stage_7_5_1_lora_noise/lora_config_resumable.yaml
RAW=$R/corpus_noigt_fullscale.jsonl
CHUNKED=$R/corpus_noigt_fullscale_chunked.jsonl
ADAPTER=$R/lora_noigt_fullscale_adapter

echo "[fs] START $(date -u +%FT%TZ)  GPU=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"

# 1. corpus full-scale (todo Psych-101 menos IGT)
if [ ! -f "$RAW" ]; then
  echo "[fs] gen corpus full-scale (--tokens 0)..."; t0=$(date +%s)
  python "$S/generate_noigt_corpus.py" --from-hf marcelbinz/Psych-101 --tokens 0 \
    --output "$RAW" --audit-output "$R/corpus_noigt_fullscale_audit.json"
  echo "[fs] gen $(($(date +%s)-t0))s  records=$(wc -l < "$RAW")"
else echo "[fs] corpus raw ya existe ($(wc -l < "$RAW") records)"; fi

# 2. re-chunk marker-aware @8192 (mismo metodo que el canonico)
if [ ! -f "$CHUNKED" ]; then
  echo "[fs] chunk marker-aware @8192 (tokenizando full corpus, puede tardar)..."; t0=$(date +%s)
  python "$S/chunk_noigt_corpus.py" --in "$RAW" --out "$CHUNKED" \
    --audit-out "$R/corpus_noigt_fullscale_chunked_audit.json" --chunk-max-tokens 8192
  echo "[fs] chunk $(($(date +%s)-t0))s  records=$(wc -l < "$CHUNKED")"
else echo "[fs] corpus chunked ya existe ($(wc -l < "$CHUNKED") records)"; fi

# 3. train resumible (chunked, sin truncar)
CKPT=""; [ -d "$ADAPTER/_training_logs" ] && CKPT=$(ls -1d "$ADAPTER/_training_logs"/checkpoint-* 2>/dev/null | sort -V | tail -1)
RESUME=(); if [ -n "$CKPT" ]; then echo "[fs] RESUME desde $CKPT"; RESUME=(--resume_from_checkpoint "$CKPT"); fi
echo "[fs] train (chunked @8192, save_steps=1)..."
python "$S/train_lora_noigt.py" --corpus "$CHUNKED" --output_adapter "$ADAPTER/" \
  --config "$CFG" "${RESUME[@]}"
echo "[fs] TRAIN STOPPED/DONE $(date -u +%FT%TZ)"
