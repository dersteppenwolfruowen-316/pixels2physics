#!/usr/bin/env bash
# One-time build of the three Python environments (run in CPU mode: free). The container disk is wiped on every
# pod restart and /workspace is an object-store FUSE mount (slow for many small files), so environments are built
# on local disk under /opt/mpm and saved to the volume as one tar each; boot.sh unpacks them on every start.
#   /opt/mpm/env/train   ms-swift 4.6.0 env, same pins as full2fps (SFT, LoRA merge)
#   /opt/mpm/env/vllm    vLLM env for fast generation (separate so it cannot move the training pins)
#   /opt/mpm/env/sim     Taichi env for simulation / scoring
#   /opt/mpm/python      uv-managed interpreters (ffmpeg comes from apt in boot.sh: torchcodec needs its libraries)
# Also downloads Qwen3-VL-4B-Instruct (same snapshot as before) to /workspace/hf. Idempotent.
set -euo pipefail
export UV_PYTHON_INSTALL_DIR=/opt/mpm/python UV_CACHE_DIR=/opt/mpm/.uv-cache UV_LINK_MODE=copy HF_HOME=/workspace/hf
V=/workspace/envs; L=/workspace/mpm/logs; mkdir -p $V $L /opt/mpm/env
log(){ echo "### $(date -u +%T) $*" | tee -a $L/setup.log; }
command -v uv >/dev/null || pip install -q uv
save(){ tar -C /opt/mpm -cf $V/$1.tar.tmp $2 && mv $V/$1.tar.tmp $V/$1.tar && log "saved $1.tar $(du -sh $V/$1.tar | cut -f1)"; }

if [[ ! -f $V/base.tar ]]; then
  log base; uv python install 3.12 3.10
  save base python
fi

if [[ ! -f $V/sim.tar ]]; then
  log sim-env; uv venv -q --python 3.10 /opt/mpm/env/sim
  uv pip install -q --python /opt/mpm/env/sim/bin/python taichi numpy imageio
  uv pip freeze --python /opt/mpm/env/sim/bin/python > $L/sim-env.freeze.txt; save sim env/sim
fi

if [[ ! -f $V/train.tar ]]; then
  log train-env; uv venv -q --python 3.12 /opt/mpm/env/train
  uv pip install -q --python /opt/mpm/env/train/bin/python torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python /opt/mpm/env/train/bin/python 'ms-swift==4.6.0' 'transformers==4.57.6' 'torchcodec==0.7.0' \
    'qwen-vl-utils==0.0.14' 'peft==0.20.0' 'accelerate==1.15.0' 'datasets==4.8.4' 'trl==0.29.1' 'huggingface-hub==0.36.2' \
    'tokenizers==0.22.2' 'tensorboard==2.21.0' 'numpy==2.5.2' 'safetensors==0.8.0' 'av==19.0.2'
  uv pip freeze --python /opt/mpm/env/train/bin/python > $L/train-env.freeze.txt; save train env/train
fi

if [[ ! -f $V/vllm.tar ]]; then
  # ms-swift 4.6.0 + vLLM 0.11.0 (tested pair in ms-swift's docs, torch 2.8 line, has Qwen3-VL). Generation goes
  # through swift's VllmEngine so videos are sampled by the same qwen_vl_utils path as training; also the GRPO env.
  # numpy is left to the resolver (vLLM's numba needs an older numpy than the training env's pin).
  log vllm-env; uv venv -q --python 3.12 /opt/mpm/env/vllm
  uv pip install -q --python /opt/mpm/env/vllm/bin/python 'vllm==0.11.0' 'torch==2.8.0' 'ms-swift==4.6.0' \
    'transformers==4.57.6' 'torchcodec==0.7.0' 'qwen-vl-utils==0.0.14' 'peft==0.20.0' 'accelerate==1.15.0' \
    'datasets==4.8.4' 'trl==0.29.1' 'huggingface-hub==0.36.2' 'tokenizers==0.22.2' 'safetensors==0.8.0' 'av==19.0.2'
  uv pip freeze --python /opt/mpm/env/vllm/bin/python > $L/vllm-env.freeze.txt; save vllm env/vllm
fi
rm -rf /opt/mpm/.uv-cache

if [[ ! -f /workspace/mpm/model_path.txt ]]; then
  log model; /opt/mpm/env/train/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17')
open('/workspace/mpm/model_path.txt', 'w').write(p + '\n'); print(p)
PY
fi
log SETUP_DONE
