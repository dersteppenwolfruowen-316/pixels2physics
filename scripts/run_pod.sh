#!/usr/bin/env bash
# Full experiment pipeline on a RunPod pod or Google Cloud VM: train -> generate -> score -> compare, then stop the machine.
#
#   bash scripts/run_pod.sh c2_no_position c1_full_aux
#
# The machine is stopped when the script exits (also on failure), so GPU billing ends.
# Set AUTO_STOP=0 to keep it running. Extra train args: TRAIN_ARGS="--epochs 2".
# Restartable: after a Spot preemption, start the VM and rerun the same command. Finished
# trainings, predictions and scores are skipped; an unfinished training resumes from its last checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=data/sft_v1
CONFIGS=("$@")
[ ${#CONFIGS[@]} -gt 0 ] || { echo "usage: bash scripts/run_pod.sh <config> [<config> ...]"; exit 1; }
mkdir -p runs/eval

stop_pod() {  # RunPod pod or Google Cloud VM: either way GPU billing stops
  [ "${AUTO_STOP:-1}" = 1 ] || return
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID"
  elif curl -s -m 2 -H "Metadata-Flavor: Google" http://metadata.google.internal >/dev/null; then
    echo "shutting down Google Cloud VM"
    sudo shutdown -h now
  fi
}
trap stop_pod EXIT

best_ckpt() {  # best checkpoint by eval loss, from the newest trainer_state.json of a run
  python - "$1" <<'EOF'
import glob, json, os, sys
states = glob.glob(f"runs/{sys.argv[1]}__weighted/*/checkpoint-*/trainer_state.json")
print(json.load(open(max(states, key=os.path.getmtime)))["best_model_checkpoint"] if states else "")
EOF
}

for config in "${CONFIGS[@]}"; do
  done_marker="runs/${config}__weighted/.done"
  [ -f "$done_marker" ] && { echo "training $config already finished"; continue; }
  python scripts/train_sft.py --config "$config" --data "$DATA" --resume ${TRAIN_ARGS:-} -- --eval_steps 50 --save_steps 50 \
    2>&1 | tee -a "runs/train_$config.log"
  [ "${PIPESTATUS[0]}" = 0 ] && touch "$done_marker"
done

generate() {  # generate <out_dir> <args...>; skipped when predictions exist
  [ -f "$1/pred.jsonl" ] && return
  local out=$1; shift
  python scripts/evaluate.py generate --out "$out/pred.jsonl" "$@"
}

first=${CONFIGS[0]}
python scripts/evaluate.py oracle --dataset "$DATA/swift/$first/test.jsonl" --out runs/eval/oracle/pred.jsonl
python scripts/evaluate.py baseline --data "$DATA" --out runs/eval/mode/pred.jsonl
generate "runs/eval/base_$first" --dataset "$DATA/swift/$first/test.jsonl"
for config in "${CONFIGS[@]}"; do
  ckpt=$(best_ckpt "$config")
  [ -n "$ckpt" ] || { echo "no checkpoint for $config, skipping"; continue; }
  generate "runs/eval/$config" --dataset "$DATA/swift/$config/test.jsonl" --adapter "$ckpt"
done

for pred in runs/eval/*/pred.jsonl; do
  [ -f "$(dirname "$pred")/summary.json" ] && continue
  python scripts/evaluate.py score --data "$DATA" --pred "$pred" --simulate --sim-workers "${SIM_WORKERS:-6}" \
    --wandb-project pixels2physics
done
python scripts/evaluate.py compare runs/eval/*/summary.json | tee runs/eval/table.md
