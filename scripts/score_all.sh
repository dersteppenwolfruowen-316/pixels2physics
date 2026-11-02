#!/usr/bin/env bash
# CPU part of the experiment: fetch the predictions, run every program, score both tracks, write the tables.
# Runs on any machine with ffmpeg and a Python that has taichi, numpy and pyyaml (no GPU needed):
#
#   PYTHON=.venv-eval/bin/python bash scripts/score_all.sh
#
# A run is rescored when its pred.jsonl or scripts/evaluate.py is newer than its summary.json.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
DATA=data/sft_v1
BUCKET=${BUCKET:-gs://p2p-data-rl948}
[ "${FETCH:-1}" = 1 ] && gcloud storage rsync -r -x '.*-merged/.*' "$BUCKET/runs" runs

# References, scored on the C4 task: ground truth (ceiling) and the train-set mode (blind floor).
[ -f runs/eval/oracle/pred.jsonl ] || "$PY" scripts/evaluate.py oracle --dataset "$DATA/swift/c4_no_state/test.jsonl" --out runs/eval/oracle/pred.jsonl
[ -f runs/eval/mode/pred.jsonl ] || "$PY" scripts/evaluate.py baseline --data "$DATA" --out runs/eval/mode/pred.jsonl

for pred in runs/eval/*/pred.jsonl; do
  run=$(dirname "$pred")
  if [ -f "$run/summary.json" ] && [ "$run/summary.json" -nt "$pred" ] && [ "$run/summary.json" -nt scripts/evaluate.py ]; then
    continue
  fi
  case "$(basename "$run")" in
    oracle|mode) flags=(--config c4_no_state --simulate --fill) ;;
    *__bo4) flags=(--simulate --best-of-n) ;;
    *__blank|*__shuffled) flags=() ;;  # controls: inferred parameters only
    *__last) flags=(--simulate) ;;
    *) flags=(--simulate --fill) ;;
  esac
  echo "== score $run ($(date +%H:%M))"
  "$PY" scripts/evaluate.py score --data "$DATA" --pred "$pred" ${flags[@]+"${flags[@]}"} --sim-workers "${SIM_WORKERS:-8}" > "$run/score.log" 2>&1
done
"$PY" scripts/evaluate.py compare runs/eval/*/summary.json | tee runs/eval/table.md
