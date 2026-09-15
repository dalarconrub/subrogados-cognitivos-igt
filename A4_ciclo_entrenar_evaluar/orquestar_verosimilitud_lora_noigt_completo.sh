#!/usr/bin/env bash
# Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.
#
# Comprueba la GPU y lanza la evaluación de LoRA-noIGT-completo sobre el corpus al terminar su
# entrenamiento.

set -euo pipefail
trap 'echo "[eval] ABORT (rc=$?) $(date -u +%H:%M:%SZ)"' ERR

B=tesis/data_analyses/llm_evaluation/paper_01_igt
S=$B/stage_7_5_3_lora_igt_specialist
R=$B/results/stage_7_5_3/canonical_base
MANIFEST_SOURCE=$B/manifests/igt_paper1_eval_data.json
WHITELIST_1087=$S/subjects_clean1087_full.json
FULLSCALE_ADAPTER=$R/lora_noigt_fullscale_adapter
IGT_ADAPTER=$R/lora_igt_adapter

# Caches reusados del eval_dissociation original (Track B canonical Stage 7.5.3)
CENTAUR_NLL=$B/results/stage_7_5_2/canonical_base/nll_centaur.json
NOISE_NLL=$B/results/stage_7_5_2/canonical_base/nll_lora_noise.json
IRRELEVANT_NLL=$B/results/stage_7_5_2/canonical_base/nll_lora_irrelevant.json
RANDOMINIT=data_runtime/paper_01_igt_outputs/stage_7_5_2/randominit_sensitivity/nll_randominit__marcelbinz__Llama-3_1-RandomInit-70B.json

echo "[eval] START $(date -u +%H:%M:%SZ)"

# 0. env
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "lora-gpu" ]; then
    if [ -f /home/zeus/miniconda3/etc/profile.d/conda.sh ]; then
        source /home/zeus/miniconda3/etc/profile.d/conda.sh
        conda activate lora-gpu
    fi
fi

# 1. Pre-flight
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(); print(f'[preflight] torch={torch.__version__}')"
python -c "from huggingface_hub import whoami; print('[preflight] HF user:', whoami()['name'])" \
    || { echo "ABORT: HF auth invalida"; exit 1; }
[ -f "$FULLSCALE_ADAPTER/adapter_model.safetensors" ] \
    || { echo "ABORT: LoRA-noIGT-fullscale adapter no encontrado en $FULLSCALE_ADAPTER (¿terminó el train?)"; exit 1; }
[ -f "$WHITELIST_1087" ] || { echo "ABORT: whitelist clean1087_full no encontrada"; exit 1; }
N_WL=$(python -c "import json; print(len(json.load(open('$WHITELIST_1087'))['subjects_flat_1087']))")
[ "$N_WL" = "1087" ] || { echo "ABORT: whitelist tiene $N_WL, esperado 1087"; exit 1; }
FS_SZ=$(du -sh "$FULLSCALE_ADAPTER" | awk '{print $1}')
echo "[preflight] LoRA-noIGT-fullscale adapter: $FS_SZ; whitelist clean1087: 1087 sujetos"

# 2. Eval B único — LoRA-noIGT-fullscale sobre clean_1087 ([revisión interna] sujetos)
echo "[eval] Eval B: LoRA-noIGT-fullscale sobre clean_1087 (1087) $(date -u +%H:%M:%SZ)"
python "$S/eval_lora_noigt_fullscale_clean1087.py" \
    --igt-data "$MANIFEST_SOURCE" \
    --whitelist-clean1087 "$WHITELIST_1087" \
    --noigt-fullscale-adapter "$FULLSCALE_ADAPTER" \
    --base-model-id meta-llama/Llama-3.1-70B \
    --top-k 10 \
    --output "$R"

# 4. Sentinel
echo "[eval] NOIGT_FULLSCALE_EVAL_DONE $(date -u +%H:%M:%SZ)"
python - "$R" <<'PY'
import sys, json, hashlib, os, datetime
R = sys.argv[1]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest() if os.path.exists(p) else None
manifest = {
    "phase": "[revisión interna] extended FULLSCALE eval doble ([revisión interna] 2026-05-30 PM, D5=b)",
    "run_at_utc": datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    "outputs": {
        "nll_lora_noigt_fullscale_clean1087.json": sha(f"{R}/nll_lora_noigt_fullscale_clean1087.json"),
    },
    "method": "Eval B: eval_lora_noigt_fullscale_clean1087.py sobre 1087 sujetos del re-encuadre limpio (input [revisión interna] [revisión interna] [revisión interna] extendida). Verdict de disociacion full-scale (470 OOD) se calcula POST-PROC local desde el output filtrado a los 470 prompt_ids del cache canonical Track B y comparado contra Centaur + LoRA-IGT cacheados.",
}
out = f"{R}/noigt_fullscale_eval_run_manifest.json"
with open(out, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"[manifest] wrote {out}")
PY
