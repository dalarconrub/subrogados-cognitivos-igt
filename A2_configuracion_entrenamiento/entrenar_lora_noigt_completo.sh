#!/usr/bin/env bash
# Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.
#
# Entrena el adaptador sobre el complemento íntegro de Psych-101 sin la Iowa Gambling Task (unos 250
# millones de tokens, troceados a 8.192), reanudando desde el último punto de control si se interrumpe, y se detiene
# en el paso 40 para igualar los pasos del par a paridad de pasos.

REPO="${REPO:-/teamspace/studios/this_studio/ai-system-lab}"
ENVNAME="${ENVNAME:-lora-gpu}"
source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate "$ENVNAME"
set -uo pipefail   # DESPUES de conda activate (activate.d puede tener vars sin definir, p.ej. cmdstan)

# Evitar fragmentación CUDA en resumes from checkpoint (lección OOM 2026-05-30 12:31 UTC):
# tras resume del adapter en step 20, ~9 GB quedaron "reservados pero no usables" → OOM en el primer
# backward step. expandable_segments:True reduce fragmentación. Aplicado al launcher para que
# cualquier re-launch futuro (manual o por chain watcher) la use automáticamente.
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$REPO/tesis/data_analyses/llm_evaluation/paper_01_igt/stage_7_5_3_lora_igt_specialist"

BASE=../results/stage_7_5_3/canonical_base
CHUNKED="$BASE/corpus_noigt_fullscale_chunked.jsonl"
TRAINABLE="$BASE/corpus_noigt_fullscale_chunked_trainable.jsonl"
ADAPTER="$BASE/lora_noigt_fullscale_adapter"
CONFIG=../stage_7_5_1_lora_noise/lora_config_fullscale.yaml   # = canónica salvo save_steps=20 (resumible multi-día)
MAXSEQ="${MAXSEQ:-8192}"

echo "[noigt-fs] START $(date -u +%FT%TZ)  repo=$REPO env=$ENVNAME maxseq=$MAXSEQ"
[ -f "$CHUNKED" ] || { echo "[noigt-fs] FALTA $CHUNKED (desplegar; sha esperado b8a44f59)"; exit 2; }

# 1) Filtrar chunks SIN answer-marker (n_answer_spans==0): secuencia 100% enmascarada -> NaN en
#    masked loss. Idempotente (salta si ya existe). 7 esperados en el full-scale.
if [ ! -f "$TRAINABLE" ]; then
  echo "[noigt-fs] filtrando chunks markerless -> $TRAINABLE"
  python - "$CHUNKED" "$TRAINABLE" <<'PY'
import json, sys
inp, out = sys.argv[1], sys.argv[2]
kept = dropped = 0
with open(inp, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("n_answer_spans", 0) > 0:
            g.write(line + "\n"); kept += 1
        else:
            dropped += 1
print(f"[noigt-fs] trainable: kept={kept} dropped_markerless={dropped}")
PY
else
  echo "[noigt-fs] $TRAINABLE ya existe (skip filtro)"
fi

# 2) Auto-resume: último checkpoint-N en _training_logs/
CKPTDIR="$ADAPTER/_training_logs"
LAST=$(ls -d "$CKPTDIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
RESUME_ARG=""
if [ -n "$LAST" ]; then
  echo "[noigt-fs] reanudando desde $LAST"
  RESUME_ARG="--resume_from_checkpoint $LAST"
else
  echo "[noigt-fs] sin checkpoint previo -> entreno desde cero"
fi

# 3) Entrenar (mismo recipe que los controles; única variable = corpus = Psych-101 SIN IGT, full-scale)
python train_lora_noigt.py \
  --corpus "$TRAINABLE" \
  --output_adapter "$ADAPTER" \
  --config "$CONFIG" \
  --max-seq-length "$MAXSEQ" \
  $RESUME_ARG
rc=$?
echo "[noigt-fs] train rc=$rc $(date -u +%FT%TZ)"
# Sentinels para watcher: NOIGT_FS_DONE | NOIGT_FS_ABORT | Traceback | CUDA out of memory
if [ $rc -ne 0 ]; then echo "[noigt-fs] NOIGT_FS_ABORT rc=$rc (reanudable: re-lanzar el mismo comando)"; exit $rc; fi
echo "[noigt-fs] NOIGT_FS_DONE adapter -> $ADAPTER"
echo "[noigt-fs] Next: eval_dissociation full-scale (Centaur vs noIGT-fullscale vs IGT) + bajar a local (researcher)"
