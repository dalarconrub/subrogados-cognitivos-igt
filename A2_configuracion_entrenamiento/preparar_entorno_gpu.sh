#!/usr/bin/env bash
# Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.
#
# Prepara el entorno de Python de la máquina con GPU en la que se entrenaron y evaluaron los adaptadores,
# instalando las versiones fijadas y comprobando que la GPU está disponible.

set -euo pipefail   # [revisión interna]: -e para que un fallo NO emita un falso "SETUP GPU OK"
trap 'echo "[setup] ABORT (rc=$?) $(date -u +%H:%M:%SZ)"' ERR

echo "== [1/4] GPU presente? =="
if ! command -v nvidia-smi >/dev/null; then
  echo "ABORT: no hay nvidia-smi -> este Studio no tiene GPU. Provisiona A100 80GB."; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== [2/4] torch + CUDA en el python activo? =="
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "OK torch+CUDA del Studio: $(python -c 'import torch;print(torch.__version__)')"
else
  echo "torch+CUDA no funcional -> instalando torch 2.10.0 (cu128, como el run canónico)"
  pip install -q "torch==2.10.0" || pip install -q torch
fi

echo "== [3/4] stack LoRA pineado (transformers <4.50 por el fix core_model_loading) =="
pip install -q --upgrade \
  "transformers>=4.43,<4.50" \
  "peft==0.19.1" \
  "bitsandbytes==0.49.2" \
  "accelerate>=0.27" \
  "datasets>=2.18" \
  scipy numpy pyyaml packaging huggingface_hub

echo "== [4/4] sanity: imports + CUDA + HF token =="
python - <<'PY'
import torch, transformers, peft, bitsandbytes, datasets, scipy, numpy
from packaging.version import parse as vparse   # comparación SEMÁNTICA, no de strings
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO-GPU")
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| bitsandbytes", bitsandbytes.__version__, "| datasets", datasets.__version__)
assert torch.cuda.is_available(), "CUDA no disponible"
# OJO: NO usar comparación de strings ("4.49.0" < "4.50" da False por '9'>'5').
assert vparse("4.43") <= vparse(transformers.__version__) < vparse("4.50"), \
    f"transformers {transformers.__version__} fuera de [4.43, 4.50): >=4.50 rompe el load del 70B (core_model_loading)"
print("SANITY OK")
PY
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN: set" || echo "WARN: exporta HF_TOKEN antes de correr (acceso a Llama-3.1-70B + Psych-101)"
echo ""
echo "SETUP GPU OK. Lanzar el run:"
echo "  bash <stage>/lightning_a100/run_lora_igt_noigt.sh"
