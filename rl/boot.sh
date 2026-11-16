#!/usr/bin/env bash
# Run on every pod start (the container disk is wiped on restart): unpack environments and the dataset to local
# disk. Videos are read many times per run, so they live on local disk, not on the FUSE volume.
#   source rl/boot.sh        (W = work dir on the volume, default /workspace/mpm; R = this folder)
set -uo pipefail
mkdir -p /opt/mpm
for t in base sim train vllm; do
  [[ -f /opt/mpm/.$t ]] && continue
  [[ -f /workspace/envs/$t.tar ]] || { echo "missing /workspace/envs/$t.tar"; continue; }
  tar -C /opt/mpm -xf /workspace/envs/$t.tar && touch /opt/mpm/.$t && echo "unpacked $t"
done
if [[ ! -f /data/SFT-2000/.ok ]]; then
  command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip >/dev/null)
  mkdir -p /data && unzip -q -o /workspace/SFT-2000.zip -x '__MACOSX/*' -d /data/ \
    && [[ $(find /data/SFT-2000/raw -name video.mp4 | wc -l) == 2000 ]] && touch /data/SFT-2000/.ok && echo "dataset ok"
fi
command -v ffmpeg >/dev/null || apt-get install -y -qq ffmpeg >/dev/null 2>&1 || true
export PATH=/opt/mpm/bin:$PATH HF_HOME=${HF_HOME:-/workspace/hf} W=${W:-/workspace/mpm} R=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# vLLM 0.11.0 bug: Qwen3-VL get_mm_mapping uses "model.visual." but vLLM names the modules "visual.*", so LoRA
# wrappers land on the vision tower and assert at profiling. Our adapter only touches the language model.
Q=/opt/mpm/env/vllm/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_vl.py
[[ -f $Q ]] && sed -i 's/connector="model.visual.merger",/connector="visual.merger",/; s/tower_model="model.visual.",/tower_model="visual.",/' $Q
# ms-swift copies args.json into each checkpoint with shutil.copy, whose chmod fails on the geesefs volume
S=/opt/mpm/env/train/lib/python3.12/site-packages/swift/trainers/mixin.py
[[ -f $S ]] && sed -i "s/shutil.copy(args_path, os.path.join(output_dir, 'args.json'))/shutil.copyfile(args_path, os.path.join(output_dir, 'args.json'))/" $S
export TRAIN=/opt/mpm/env/train/bin SIM=/opt/mpm/env/sim/bin/python VLLM=/opt/mpm/env/vllm/bin
df -h / /workspace | tail -2
