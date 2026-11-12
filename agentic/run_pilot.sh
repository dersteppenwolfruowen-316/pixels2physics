#!/usr/bin/env bash
# Agentic pilot on the GPU VM: staged refinement of the c4_scaffold model's own programs, then the equal-budget
# resampling control. On exit (also on failure) runs/agentic is uploaded and the VM stops.
#
#   bash agentic/run_pilot.sh            # 30 scenes (PILOT=30), name "pilot"
#
# Needs on the VM: the c4_scaffold checkpoint and runs/eval/c4_scaffold__bo8/pred.jsonl (both from the training
# run), taichi + pyflakes in the venv (installed below), ffmpeg. Scoring runs afterwards on the Mac:
#   .venv-eval/bin/python scripts/evaluate.py score --pred runs/agentic/pilot/pred.jsonl --config c4_no_state --simulate
set -uo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-pilot}
PILOT=${PILOT:-30}
BUCKET=${BUCKET:-gs://p2p-data-rl948}
SIM_WORKERS=${SIM_WORKERS:-10}

finish() {
  echo "uploading runs/agentic to $BUCKET/runs/agentic"
  gcloud storage rsync -r runs/agentic "$BUCKET/runs/agentic" || echo "upload failed: results are still on this disk"
  [ "${AUTO_STOP:-1}" = 1 ] || return
  curl -s -m 2 -H "Metadata-Flavor: Google" http://metadata.google.internal >/dev/null && { echo "shutting down"; sudo shutdown -h now; }
}
trap finish EXIT

python -c "import taichi, pyflakes, PIL" 2>/dev/null || pip install -q "taichi==1.7.4" pyflakes pillow
[ -f data/sft_v1/swift/c4_scaffold/test.jsonl ] || python scripts/build_sft_data.py --out data/sft_v1 --configs c4_scaffold --no-controls
CKPT=$(python - <<'EOF'
import glob, json, os
states = glob.glob("runs/c4_scaffold__weighted/*/checkpoint-*/trainer_state.json")
print(json.load(open(max(states, key=os.path.getmtime)))["best_model_checkpoint"] if states else "")
EOF
)
[ -n "$CKPT" ] || { echo "no c4_scaffold checkpoint found"; exit 1; }
echo "checkpoint: $CKPT"

mkdir -p runs/agentic
python agentic/loop.py run --name "$NAME" --pilot "$PILOT" --adapter "$CKPT" --sim-workers "$SIM_WORKERS" 2>&1 \
  | grep -v --line-buffered '^\[Taichi\]' | tee -a "runs/agentic/$NAME.log"
python agentic/loop.py resample --name "${NAME}_resample" --like "$NAME" --adapter "$CKPT" --sim-workers "$SIM_WORKERS" 2>&1 \
  | grep -v --line-buffered '^\[Taichi\]' | tee -a "runs/agentic/${NAME}_resample.log"
echo "== agentic pilot done ($(date +%H:%M))"
