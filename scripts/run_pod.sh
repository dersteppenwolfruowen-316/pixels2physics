#!/usr/bin/env bash
# GPU part of the experiment on a Google Cloud VM or RunPod pod: data -> train -> generate -> upload -> stop.
# Simulation and scoring need no GPU; they run afterwards on a CPU machine: bash scripts/score_all.sh
#
#   bash scripts/run_pod.sh c4_no_state
#
# The 1-epoch scaffold run (one epoch, cosine to zero, all four checkpoints kept, greedy + best-of-8, no controls):
#   STOP_AT= EVAL_EVERY=25 TRAIN_ARGS="--epochs 1" CONTROLS= GEN_CONFIGS=c4_scaffold BON_CONFIGS=c4_scaffold BON_N=8 \
#     LAST_CONFIGS= bash scripts/run_pod.sh c4_scaffold
#
# Arguments: configs to train (C4 by default follows the C1/C2 3-epoch schedule and stops after step 150).
# Generation, most important first, for configs that have a trained checkpoint:
#   1. greedy test predictions                 every config in GEN_CONFIGS
#   2. controls (blank / shuffled frames)      the configs trained here; physics block only
#   3. best-of-4 samples                       BON_CONFIGS
#   4. last checkpoint instead of best         LAST_CONFIGS (does longer training write programs that run?)
# Finished steps are skipped on a rerun. On exit (also on failure) runs/ is uploaded to $BUCKET and the
# machine is stopped, so GPU billing ends. AUTO_STOP=0 keeps it running.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=data/sft_v1
TRAIN=("$@")
GEN_CONFIGS=(${GEN_CONFIGS:-c1_full_aux c2_no_position c4_no_state})
BON_CONFIGS=(${BON_CONFIGS:-c4_no_state c1_full_aux})
LAST_CONFIGS=(${LAST_CONFIGS-c1_full_aux})  # empty: none
STOP_AT=${STOP_AT-150}  # empty: train the full --epochs schedule
EVAL_EVERY=${EVAL_EVERY:-50}
CONTROLS=(${CONTROLS-blank shuffled})  # empty: no control sets
BON_N=${BON_N:-4}
BUCKET=${BUCKET:-gs://p2p-data-rl948}
mkdir -p runs/eval

finish() {
  echo "uploading runs/ to $BUCKET/runs"
  gcloud storage rsync -r -x '.*-merged/.*' runs "$BUCKET/runs" || echo "upload failed: results are still on this disk"
  [ "${AUTO_STOP:-1}" = 1 ] || return
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  elif curl -s -m 2 -H "Metadata-Flavor: Google" http://metadata.google.internal >/dev/null; then
    echo "shutting down Google Cloud VM"
    sudo shutdown -h now
  fi
}
trap finish EXIT

best_ckpt() {  # best checkpoint by eval loss, from the newest trainer_state.json of a run
  python - "$1" <<'EOF'
import glob, json, os, sys
states = glob.glob(f"runs/{sys.argv[1]}__weighted/*/checkpoint-*/trainer_state.json")
print(json.load(open(max(states, key=os.path.getmtime)))["best_model_checkpoint"] if states else "")
EOF
}

last_ckpt() {  # highest-step checkpoint of a run
  python - "$1" <<'EOF'
import glob, re, sys
ckpts = [p for p in glob.glob(f"runs/{sys.argv[1]}__weighted/*/checkpoint-*") if re.search(r"checkpoint-\d+$", p)]
print(max(ckpts, key=lambda p: int(p.rsplit("-", 1)[1])) if ckpts else "")
EOF
}

generate() {  # generate <out_dir> <args...>; skipped when predictions exist
  [ -f "$1/pred.jsonl" ] && return
  local out=$1; shift
  echo "== generate $out ($(date +%H:%M))"
  python scripts/evaluate.py generate --out "$out/pred.jsonl" "$@"
}

for config in "${TRAIN[@]}"; do
  if [ ! -f "$DATA/swift/$config/train.jsonl" ]; then
    if [ ${#CONTROLS[@]} -gt 0 ]; then controls=(--controls "${CONTROLS[@]}"); else controls=(--no-controls); fi
    python scripts/build_sft_data.py --out "$DATA" --configs "$config" "${controls[@]}"
  fi
  done_marker="runs/${config}__weighted/.done"
  [ -f "$done_marker" ] && { echo "training $config already finished"; continue; }
  stop=(); [ -n "$STOP_AT" ] && stop=(--stop-at-step "$STOP_AT")
  python scripts/train_sft.py --config "$config" --data "$DATA" --resume "${stop[@]}" ${TRAIN_ARGS:-} \
    -- --eval_steps "$EVAL_EVERY" --save_steps "$EVAL_EVERY" --save_total_limit 4 2>&1 | tee -a "runs/train_$config.log"
  [ "${PIPESTATUS[0]}" = 0 ] && touch "$done_marker"
done

for config in "${GEN_CONFIGS[@]}"; do
  ckpt=$(best_ckpt "$config")
  [ -n "$ckpt" ] || { echo "no checkpoint for $config, skipping"; continue; }
  generate "runs/eval/$config" --dataset "$DATA/swift/$config/test.jsonl" --adapter "$ckpt"
done
for config in "${TRAIN[@]}"; do
  ckpt=$(best_ckpt "$config")
  for control in "${CONTROLS[@]}"; do
    generate "runs/eval/${config}__$control" --dataset "$DATA/swift/controls/${config}__$control/test.jsonl" --adapter "$ckpt" --physics-only
  done
done
for config in "${BON_CONFIGS[@]}"; do
  generate "runs/eval/${config}__bo$BON_N" --dataset "$DATA/swift/$config/test.jsonl" --adapter "$(best_ckpt "$config")" --num-samples "$BON_N"
done
for config in "${LAST_CONFIGS[@]}"; do
  generate "runs/eval/${config}__last" --dataset "$DATA/swift/$config/test.jsonl" --adapter "$(last_ckpt "$config")"
done
echo "== GPU work done ($(date +%H:%M))"
