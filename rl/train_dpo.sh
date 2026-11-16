#!/usr/bin/env bash
# DPO on top of B1: the policy continues B1's LoRA (--adapters), the frozen reference is B1 (--ref_adapters), so the
# result is again one LoRA on the base model and is merged/evaluated exactly like A and B1. Same video settings as SFT.
#   bash rl/train_dpo.sh <pairs.jsonl> <B1 checkpoint> <out dir> [lr] [beta]
set -euo pipefail
DATA=$1; REF=$2; OUT=$3; LR=${4:-1e-5}; BETA=${5:-0.1}
W=${W:-/workspace/mpm}
export FPS=2 FPS_MAX_FRAMES=768 VIDEO_MIN_TOKEN_NUM=128 VIDEO_MAX_TOKEN_NUM=128
export FORCE_QWENVL_VIDEO_READER=torchcodec TORCHCODEC_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="$(cat $W/model_path.txt)"
ARGS=(--rlhf_type dpo --model "$MODEL" --use_hf true --split_dataset_ratio 0 --dataset "$DATA"
  --adapters "$REF" --ref_adapters "$REF" --tuner_type lora --freeze_vit true --freeze_aligner true
  --beta $BETA --rpo_alpha 1.0 --loss_type sigmoid
  --torch_dtype bfloat16 --attn_impl sdpa --max_length 32768 --truncation_strategy delete --lazy_tokenize true
  --learning_rate $LR --num_train_epochs 1 --lr_scheduler_type cosine --warmup_ratio 0.05 --weight_decay 0.0
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing true
  --dataloader_num_workers 4 --logging_steps 1 --save_strategy steps --save_steps 25 --save_total_limit 2
  --eval_strategy no --report_to tensorboard --seed 42 --data_seed 42 --add_version false --output_dir "$OUT")
LAST=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[[ -n "$LAST" ]] && ARGS+=(--resume_from_checkpoint "$LAST") && echo "resume from $LAST"
mkdir -p "$OUT"
printf '%q ' swift rlhf "${ARGS[@]}" >> "$OUT/command.txt"; echo >> "$OUT/command.txt"
${TRAIN:-/opt/mpm/env/train/bin}/swift rlhf "${ARGS[@]}"
