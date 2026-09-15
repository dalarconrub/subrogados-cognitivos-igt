#!/usr/bin/env bash
# Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.
#
# Comprueba la GPU y el acceso a los modelos y lanza, uno tras otro, la evaluación de LoRA-IGT-1época y de
# LoRA-noIGT-1época sobre las sesiones restantes, dejando una marca al terminar.

set -euo pipefail
trap 'echo "[run] ABORT (rc=$?) $(date -u +%H:%M:%SZ)"' ERR

B=tesis/data_analyses/llm_evaluation/paper_01_igt
S=$B/stage_7_5_3_lora_igt_specialist
R=$B/results/stage_7_5_3/canonical_base
MANIFEST_SOURCE=$B/manifests/igt_paper1_eval_data.json
WHITELIST=$S/subjects_clean1087_gap617.json
IGT_ADAPTER=$R/lora_igt_adapter
NOIGT_ADAPTER=$R/lora_noigt_adapter

echo "[run] START $(date -u +%H:%M:%SZ)"

# 0. Activar env conda lora-gpu si no está activo (SSH no-interactivo NO lo auto-activa)
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "lora-gpu" ]; then
    if [ -f /home/zeus/miniconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /home/zeus/miniconda3/etc/profile.d/conda.sh
        conda activate lora-gpu
        echo "[run] conda env activated: $(python -c 'import sys; print(sys.executable)')"
    else
        echo "[run] WARN: conda.sh no encontrado; asumiendo env Python correcto."
    fi
fi

# 1. Pre-flight GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(f'[preflight] torch={torch.__version__} cuda={torch.version.cuda}')"

# 2. Pre-flight HF auth (P2.3 del RUNBOOK_A100.md)
python -c "from huggingface_hub import whoami; print('[preflight] HF user:', whoami()['name'])" \
    || { echo "ABORT: HF auth invalida"; exit 1; }

# 3. Pre-flight HF cache hygiene (regla feedback Lightning HF cache hygiene)
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
echo "[preflight] HF_HOME=$HF_CACHE"
if [ -d "$HF_CACHE/hub/models--meta-llama--Llama-3.1-70B" ]; then
    BASE_SZ=$(du -sh "$HF_CACHE/hub/models--meta-llama--Llama-3.1-70B" 2>/dev/null | awk '{print $1}')
    echo "[preflight] Llama-3.1-70B cache: $BASE_SZ (esperado ~130GB completo)"
else
    echo "[preflight] WARN: Llama-3.1-70B no en cache; primera carga descargará ~130GB"
fi

# 4. Pre-flight adapters locales
[ -f "$IGT_ADAPTER/adapter_model.safetensors" ] || { echo "ABORT: LoRA-IGT adapter no encontrado en $IGT_ADAPTER"; exit 1; }
[ -f "$NOIGT_ADAPTER/adapter_model.safetensors" ] || { echo "ABORT: LoRA-noIGT adapter no encontrado en $NOIGT_ADAPTER"; exit 1; }
IGT_SZ=$(du -sh "$IGT_ADAPTER" | awk '{print $1}')
NOIGT_SZ=$(du -sh "$NOIGT_ADAPTER" | awk '{print $1}')
echo "[preflight] LoRA-IGT adapter: $IGT_SZ; LoRA-noIGT adapter: $NOIGT_SZ"

# 5. Pre-flight whitelist + manifest source
[ -f "$WHITELIST" ] || { echo "ABORT: whitelist no encontrada en $WHITELIST"; exit 1; }
[ -f "$MANIFEST_SOURCE" ] || { echo "ABORT: manifest fuente no encontrado en $MANIFEST_SOURCE"; exit 1; }
N_WL=$(python -c "import json; print(len(json.load(open('$WHITELIST'))['subjects_flat_617']))")
[ "$N_WL" = "617" ] || { echo "ABORT: whitelist tiene $N_WL, esperado 617"; exit 1; }
echo "[preflight] whitelist OK: [revisión interna]"

# 6. Inferencia LoRA-IGT (resumable)
echo "[run] Inferencia LoRA-IGT clean_1087 gap (617) $(date -u +%H:%M:%SZ)"
python "$S/eval_lora_clean1087_extension.py" \
    --igt-data "$MANIFEST_SOURCE" \
    --whitelist "$WHITELIST" \
    --igt-adapter "$IGT_ADAPTER" \
    --noigt-adapter "$NOIGT_ADAPTER" \
    --base-model-id meta-llama/Llama-3.1-70B \
    --top-k 10 \
    --only igt \
    --output "$R"

# 7. Inferencia LoRA-noIGT (resumable)
echo "[run] Inferencia LoRA-noIGT clean_1087 gap (617) $(date -u +%H:%M:%SZ)"
python "$S/eval_lora_clean1087_extension.py" \
    --igt-data "$MANIFEST_SOURCE" \
    --whitelist "$WHITELIST" \
    --igt-adapter "$IGT_ADAPTER" \
    --noigt-adapter "$NOIGT_ADAPTER" \
    --base-model-id meta-llama/Llama-3.1-70B \
    --top-k 10 \
    --only noigt \
    --output "$R"

# 8. Sentinel + manifest del run
echo "[run] CLEAN1087 EXT DONE $(date -u +%H:%M:%SZ)"
python - "$R" <<'PY'
import sys, json, hashlib, os, datetime
R = sys.argv[1]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest() if os.path.exists(p) else None
manifest = {
    "phase": "[revisión interna] extended ([revisión interna] 2026-05-30 PM, opcion alpha statu quo)",
    "run_at_utc": datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    "outputs": {
        "nll_lora_igt_clean1087_gap617.json": sha(f"{R}/nll_lora_igt_clean1087_gap617.json"),
        "nll_lora_noigt_clean1087_gap617.json": sha(f"{R}/nll_lora_noigt_clean1087_gap617.json"),
    },
    "method": "Variante de eval_lora_igt.py con filtro WHITELIST 617 prompt_ids en lugar de EXTERNAL_PREFIXES. Métrica per-sujeto-mean-across-tokens identica al cache canonical. Bug loader Steingroever 2015 preservado (opcion alpha) para comparabilidad bit-identical con Centaur/Llama base/LoRA-noise/LoRA-irrelev/RandomInit/cognitivos.",
}
out = f"{R}/clean1087_extension_run_manifest.json"
with open(out, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"[manifest] wrote {out}")
PY
