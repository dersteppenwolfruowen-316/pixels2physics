#!/usr/bin/env bash
# Frame-rate experiment on one A100 80GB: data -> 30/10/2 FPS pilots -> Full2FPS -> generate -> score.
# Every stage is skipped when its output exists, so after a Spot preemption just run it again.
#
#   bash framerate/run_gpu.sh                 # everything
#   bash framerate/run_gpu.sh full gen score  # only these stages
#
# Stages:
#   data    framerate/prepare_data.py for 30, 10, 2 FPS and 2 FPS + scaffold          (CPU, about 10 min)
#   pilot   46 optimizer steps per frame rate, rank 16; greedy generation on the 16 test scenes
#   full    Full2FPS: 2 FPS + scaffold, rank 64, 2 epochs = 400 steps                  (about 1 h 50 min)
#   gen     checkpoints 100 and 400: greedy + 3 samples at T = 0.7 on the 16 test scenes (about 1 h each)
#   score   run every program in Taichi and score with scripts/evaluate.py             (CPU is enough)
# Needs: ms-swift 4.6, transformers 4.57, torchcodec 0.7, taichi 1.7.4, ffmpeg; data/SFT-2000 unzipped.
# If BUCKET is set, runs/framerate is uploaded there on exit.
set -uo pipefail
cd "$(dirname "$0")/.."

SRC=${SRC:-data/SFT-2000}
D=data/fps
R=runs/framerate
STEPS=${PILOT_STEPS:-46}
STAGES=("${@:-data pilot full gen score}")
STAGES=(${STAGES[*]})
want() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }
log() { echo "### $(date -u +%T) $*" | tee -a $R/stage.log; }
mkdir -p $R
finish() { [ -n "${BUCKET:-}" ] && gcloud storage rsync -r $R "$BUCKET/runs/framerate"; }
trap finish EXIT

if want data; then
  for f in 30 10 2; do
    [ -f $D/fps_$f/manifest.json ] || { log "data fps_$f"; python framerate/prepare_data.py --source $SRC --out $D/fps_$f --fps $f || exit 1; }
  done
  [ -f $D/fps_2s/manifest.json ] || { log "data fps_2s"; python framerate/prepare_data.py --source $SRC --out $D/fps_2s --fps 2 --scaffold || exit 1; }
fi

if want pilot; then
  for f in 30 10 2; do
    CK=$R/fps_$f/pilot/checkpoint-$STEPS
    [ -d $CK ] || { log "pilot train fps_$f"; bash framerate/train.sh pilot $D/fps_$f $f $STEPS >> $R/pilot_$f.log 2>&1 || exit 1; }
    [ -f $R/fps_$f/pilot/eval/pred.jsonl ] || { log "pilot gen fps_$f"; python framerate/generate.py --data $D/fps_$f --fps $f \
      --adapter $CK --samples 1 --out $R/fps_$f/pilot/eval/pred.jsonl >> $R/pilot_$f.log 2>&1 || exit 1; }
  done
  # the 2 FPS pilot adapter with the scaffold in the prompt only (it was never trained on scaffolds)
  [ -f $R/fps_2/pilot/eval_scaffold/pred.jsonl ] || { log "pilot gen fps_2 + scaffold"; python framerate/generate.py --data $D/fps_2s \
    --fps 2 --adapter $R/fps_2/pilot/checkpoint-$STEPS --samples 1 --out $R/fps_2/pilot/eval_scaffold/pred.jsonl >> $R/pilot_2.log 2>&1 || exit 1; }
fi

if want full; then
  [ -f $R/fps_2s/full/TRAIN_DONE ] || { log "full train"; bash framerate/train.sh full $D/fps_2s 2 >> $R/full.log 2>&1 && touch $R/fps_2s/full/TRAIN_DONE || exit 1; }
fi

if want gen; then
  for ck in 100 400; do
    [ -f $R/fps_2s/full/ck$ck/pred.jsonl ] || { log "gen ck$ck"; python framerate/generate.py --data $D/fps_2s --fps 2 \
      --adapter $R/fps_2s/full/checkpoint-$ck --samples 4 --out $R/fps_2s/full/ck$ck/pred.jsonl >> $R/gen.log 2>&1 || exit 1; }
  done
fi

if want score; then
  for f in 30 10 2; do python framerate/score.py --data $D/fps_$f --pred $R/fps_$f/pilot/eval/pred.jsonl --greedy-only; done
  python framerate/score.py --data $D/fps_2s --pred $R/fps_2/pilot/eval_scaffold/pred.jsonl --greedy-only
  for ck in 100 400; do python framerate/score.py --data $D/fps_2s --pred $R/fps_2s/full/ck$ck/pred.jsonl; done
  log "score done"
fi
