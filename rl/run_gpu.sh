#!/usr/bin/env bash
# GPU-phase driver. Every stage is resumable (per scene / per sample / from checkpoints) and marks itself done,
# so after a crash or a pod restart just run it again:   bash rl/run_gpu.sh [stage...]
# Stages (default: all in order):
#   smoke   GPU + vLLM check: A on the old 16 test scenes (1 greedy + 3 samples), prompt tokens vs the old run
#   genA    A samples: train 1,600 x 4 (T=1.0) for ReST, eval set x 4 (greedy + 3 at T=0.7)
#   restB1  filter -> SFT from base (1 epoch over GT + kept)
#   genB1   B1 samples on the eval set x 4
#   report  analysis A vs B1
# Simulations run on CPU in a background loop (simloop) that scores whatever samples exist, so the GPU never
# waits for the simulator except at the ReST filter step.
set -uo pipefail
source "$(dirname "$0")/boot.sh"
cd $W; mkdir -p runs/A runs/B1 results logs
A_CK=${A_CK:-$W/ckpt/A_ck400}   # Full2FPS checkpoint-400 (framerate/run_gpu.sh)
NCPU=$(nproc); CG=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null); [[ ${CG:-0} -gt 0 ]] && NCPU=$CG
SIMW=$(( NCPU > 4 ? (NCPU - 2) / 2 : 1 ))          # CPU sims: 2 Taichi threads each, keep 2 cores for vLLM/data
SIMDEV=${SIMDEV:-gpu}; SIMW_G=${SIMW_G:-8}           # GPU sims (programs keep ti.cuda, ~1.5 GB each)
GPUMEM=${GPUMEM:-0.80}                               # vLLM share; the rest is for concurrent Taichi processes
# vLLM decodes ~2.6x faster from merged weights than through LoRA kernels; every arm is generated the same way.
merge(){  # merge <adapter> <outdir>
  [[ -f $2/config.json ]] && return 0
  local M=$(cat $W/model_path.txt)
  $TRAIN/swift export --model $M --adapters $1 --merge_lora true --output_dir $2 >> $W/logs/merge.log 2>&1
}
sim(){ local w=$SIMW; [[ $SIMDEV == gpu ]] && w=$SIMW_G; $SIM -u $R/sim_score.py --device $SIMDEV --workers $w "$@"; }
log(){ echo "### $(date -u +%F' '%T) $*" | tee -a $W/logs/run_gpu.log; }
done_(){ [[ -f $W/runs/.done_$1 ]]; }; mark(){ touch $W/runs/.done_$1; log "DONE $1"; }
[[ -f fps_2s/data/train.jsonl ]] || { log make_data; $SIM $R/make_data.py /data/SFT-2000 $W || exit 1; }
log "cpus=$NCPU sim_workers=$SIMW $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

simloop(){  # simloop <samples> <scores> <stopfile> [timeout]: keep scoring new rows until <stopfile> exists
  local s=$1 o=$2 stop=$3 t=${4:-300}
  while true; do
    local fin=0; [[ -f $stop ]] && fin=1
    [[ -f $s ]] && sim --samples $s --out $o --timeout $t --keep-videos 0.01 >> ${o%.jsonl}.log 2>&1
    [[ $fin == 1 ]] && break; sleep 60
  done
}
gen(){ $VLLM/python -u $R/gen_vllm.py --gpu-mem $GPUMEM "$@"; }
A_M=/opt/mpm/merged/A_ck400; B1_M=/opt/mpm/merged/B1

stage_smoke(){
  done_ smoke && return
  merge $A_CK $A_M || return 1
  gen --split test --indices $R/../framerate/test_scenes.json --model $A_M --samples 4 --greedy-first --temperature 0.7 \
      --out runs/A/smoke16.jsonl >> logs/smoke.log 2>&1 || { log "smoke gen FAILED (logs/smoke.log)"; return 1; }
  $VLLM/python - <<'PY' | tee -a logs/smoke.log
import json, os
W, R = os.environ['W'], os.environ['R']
old = {json.loads(l)['scene_id']: json.loads(l)['prompt_tokens'] for l in open(f'{R}/A_old_prompt_tokens.jsonl')}
new = {}
for l in open(f'{W}/runs/A/smoke16.jsonl'):
    r = json.loads(l); new[r['scene_id']] = r['prompt_tokens']
same = sum(old[k] == new.get(k) for k in old)
print(f'prompt tokens identical to the old HF run on {same}/{len(old)} scenes', {k: (old[k], new.get(k)) for k in old if old[k] != new.get(k)})
PY
  # same 64 programs simulated on GPU and on CPU: pick the faster device; check scores agree with the old report
  $SIM -u $R/sim_score.py --samples runs/A/smoke16.jsonl --out runs/A/smoke16_gpu.jsonl --device gpu --workers $SIMW_G --timeout 900 >> logs/smoke.log 2>&1
  $SIM - <<'PY' | tee -a logs/smoke.log
import json, os, statistics as st
W = os.environ['W']
for dev in ('gpu',):
    r = [json.loads(l) for l in open(f'{W}/runs/A/smoke16_{dev}.jsonl')]
    ok = [x for x in r if x['sim_success']]
    t = sorted(x['sim_seconds'] for x in ok)
    g = [x for x in r if x['sample'] == 0]
    print(dev, f"runs {len(ok)}/{len(r)}, greedy runs {sum(x['sim_success'] for x in g):.0f}/16 (old report 4/16),",
          f"sim s median {st.median(t) if t else None} max {max(t) if t else None}")
PY
  mark smoke
}

stage_genA(){
  done_ genA && return
  rm -f runs/A/.gen_done
  ( simloop runs/A/train_samples.jsonl runs/A/train_scores.jsonl runs/A/.gen_done 300 ) &
  local SL=$!
  merge $A_CK $A_M || return 1
  gen --split test --indices eval100.json --model $A_M --samples 4 --greedy-first --temperature 0.7 \
      --out runs/A/test_samples.jsonl >> logs/genA.log 2>&1 || { log "genA test FAILED"; return 1; }
  gen --split train --model $A_M --samples 4 --temperature 1.0 --chunk 128 --out runs/A/train_samples.jsonl \
      >> logs/genA.log 2>&1 || { log "genA train FAILED"; return 1; }
  touch runs/A/.gen_done; wait $SL
  mark genA
}

stage_restB1(){
  done_ restB1 && return
  # ground-truth reward only where a sample could pass the floor (sets tau; saves ~half the GT simulations)
  $SIM - <<'PY'
import json, os
W = os.environ['W']
s = [json.loads(l) for l in open(f'{W}/runs/A/train_scores.jsonl')]
cand = {r['scene_id'] for r in s if r['sim_success'] and r['reward'] >= 0.5}
gt = [json.loads(l) for l in open(f'{W}/gt_train_samples.jsonl')]
open(f'{W}/runs/A/gt_cand_samples.jsonl', 'w').writelines(json.dumps(g) + '\n' for g in gt if g['scene_id'] in cand)
print('GT sims needed', len(cand))
PY
  sim --samples runs/A/gt_cand_samples.jsonl --out runs/A/gt_train_scores.jsonl --timeout 600 >> logs/gt.log 2>&1
  $SIM $R/rest_build.py --samples runs/A/train_samples.jsonl --scores runs/A/train_scores.jsonl \
      --gt runs/A/gt_train_scores.jsonl --out runs/B1/rest_train.jsonl | tee -a logs/restB1.log
  # A's test-set simulations run on the CPU while B1 trains on the GPU
  ( SIMW_G=4 sim --samples runs/A/test_samples.jsonl --out runs/A/test_scores.jsonl --timeout 900 >> logs/simA_test.log 2>&1 ) &
  local SL=$!
  bash $R/train_sft.sh $W/runs/B1/rest_train.jsonl $W/runs/B1/sft 1 >> logs/trainB1.log 2>&1 || { log "train B1 FAILED"; return 1; }
  wait $SL
  mark restB1
}

stage_genB1(){
  done_ genB1 && return
  local CK=$(ls -d runs/B1/sft/checkpoint-* | sort -t- -k2 -n | tail -1)
  log "B1 adapter $CK"
  rm -rf $A_M; merge $W/$CK $B1_M || return 1   # container disk holds one merged model at a time
  gen --split test --indices eval100.json --model $B1_M --samples 4 --greedy-first --temperature 0.7 \
      --out runs/B1/test_samples.jsonl >> logs/genB1.log 2>&1 || { log "genB1 FAILED"; return 1; }
  sim --samples runs/B1/test_samples.jsonl --out runs/B1/test_scores.jsonl --timeout 900 >> logs/simB1_test.log 2>&1
  mark genB1
}

stage_report(){
  sim --samples runs/A/test_samples.jsonl --out runs/A/test_scores.jsonl --timeout 900 >> logs/simA_test.log 2>&1
  $SIM $R/analyze.py --arm A=runs/A/test_scores.jsonl --arm B1=runs/B1/test_scores.jsonl --baseline A --out results/ | tee logs/report.log
  mark report
}

STAGES=${*:-smoke genA restB1 genB1 report}
for s in $STAGES; do log "stage $s"; stage_$s || { log "STOP at $s"; exit 1; }; done
log ALL_DONE
