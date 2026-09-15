#!/usr/bin/env bash
# Anexo A, apartado A.1 · Modelos cognitivos clásicos: parametrización y ajuste.
#
# Recoge la forma exacta del comando con el que se ajustaron VSE, ORL y PVL-Δ, uno tras otro, en una
# máquina sin GPU; cada modelo reparte sus pliegues entre varios procesos y puede reanudarse sin repetir los
# pliegues ya escritos.

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /home/zeus/miniconda3/etc/profile.d/conda.sh
conda activate igt-dm
set -uo pipefail
cd /teamspace/studios/this_studio/ai-system-lab
M=data_runtime/analysis/dm_paper1_igt1598/subjects_always4_n1598.jsonl
OUT=data_runtime/analysis/dm_paper1_igt1598/h1_pooled_ml_subject_kfold
JOBS="${JOBS:-32}"; NSTARTS="${NSTARTS:-32}"; MAXITER="${MAXITER:-1000}"
mkdir -p "$OUT"
echo "[h1pool] START $(date -u +%FT%TZ)  nproc=$(nproc)  jobs=$JOBS n_starts=$NSTARTS maxiter=$MAXITER"
for m in vse orl pvldelta; do
  echo "[h1pool] === $m START $(date -u +%FT%TZ) ==="
  python ops/dm_baselines.py "$m" --manifest "$M" --out "$OUT/$m.jsonl" \
    --pooled-kfold-subjects --kfolds 5 --seed 20260506 \
    --jobs "$JOBS" --n-starts "$NSTARTS" --maxiter "$MAXITER"
  echo "[h1pool] === $m DONE rc=$? $(date -u +%FT%TZ) ==="
done
echo "[h1pool] ALL DONE $(date -u +%FT%TZ)"
