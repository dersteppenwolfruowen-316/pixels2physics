#!/usr/bin/env bash
# Phase 2, after run_gpu.sh (B1 done): B1+DPO -> eval -> 3-arm report. Resumable like run_gpu.sh.
#   bash rl/run_dpo.sh [stage...]     stages: dpo genD report3
set -uo pipefail
source "$(dirname "$0")/boot.sh"
cd $W; mkdir -p runs/D results logs
SIMW_G=${SIMW_G:-8}; GPUMEM=${GPUMEM:-0.80}; MAXPAIRS=${MAXPAIRS:-0}
sim(){ $SIM -u $R/sim_score.py --device gpu --workers $SIMW_G "$@"; }
gen(){ $VLLM/python -u $R/gen_vllm.py --gpu-mem $GPUMEM "$@"; }
log(){ echo "### $(date -u +%F' '%T) $*" | tee -a $W/logs/run_dpo.log; }
done_(){ [[ -f $W/runs/.done_$1 ]]; }; mark(){ touch $W/runs/.done_$1; log "DONE $1"; }
merge(){ [[ -f $2/config.json ]] && return 0
  $TRAIN/swift export --model $(cat $W/model_path.txt) --adapters $1 --merge_lora true --output_dir $2 >> $W/logs/merge.log 2>&1; }
B1_CK=$(ls -d $W/runs/B1/sft/checkpoint-* | sort -t- -k2 -n | tail -1)
D_M=/opt/mpm/merged/D

stage_dpo(){
  done_ dpo && return
  log "B1 reference $B1_CK"
  $SIM $R/dpo_build.py --samples runs/A/train_samples.jsonl --scores runs/A/train_scores.jsonl \
      --gt runs/A/gt_train_scores.jsonl --max-pairs $MAXPAIRS --out runs/D/dpo_pairs.jsonl | tee -a logs/dpo.log
  bash $R/train_dpo.sh $W/runs/D/dpo_pairs.jsonl $B1_CK $W/runs/D/dpo >> logs/trainD.log 2>&1 || { log "train DPO FAILED"; return 1; }
  mark dpo
}

stage_genD(){
  done_ genD && return
  local CK=$(ls -d runs/D/dpo/checkpoint-* | sort -t- -k2 -n | tail -1)
  log "DPO adapter $CK"
  rm -rf /opt/mpm/merged/A_ck400 /opt/mpm/merged/B1; merge $W/$CK $D_M || return 1
  gen --split test --indices eval100.json --model $D_M --samples 4 --greedy-first --temperature 0.7 \
      --out runs/D/test_samples.jsonl >> logs/genD.log 2>&1 || { log "genD FAILED"; return 1; }
  sim --samples runs/D/test_samples.jsonl --out runs/D/test_scores.jsonl --timeout 900 >> logs/simD_test.log 2>&1
  mark genD
}

stage_report3(){
  $SIM $R/analyze.py --arm A=runs/A/test_scores.jsonl --arm B1=runs/B1/test_scores.jsonl \
      --arm B1+DPO=runs/D/test_scores.jsonl --baseline A --out results/ | tee logs/report3.log
  $SIM $R/analyze.py --arm B1=runs/B1/test_scores.jsonl --arm B1+DPO=runs/D/test_scores.jsonl \
      --baseline B1 --out results/vsB1/ | tee -a logs/report3.log
  mark report3
}

STAGES=${*:-dpo genD report3}
for s in $STAGES; do log "stage $s"; stage_$s || { log "STOP at $s"; exit 1; }; done
log ALL_DONE
