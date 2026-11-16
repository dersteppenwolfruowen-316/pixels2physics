#!/usr/bin/env bash
# LoRA SFT for the frame-rate track (ms-swift). The model sees the whole video at $FPS frames per second, 128 visual
# tokens per frame; the vision encoder and aligner stay frozen. Resumes from the newest checkpoint automatically.
#
#   bash framerate/train.sh pilot data/fps/fps_30 30 46   # equal-step pilot: rank 16, 46 optimizer steps
#   bash framerate/train.sh full  data/fps/fps_2s  2      # Full2FPS: rank 64, 2 epochs (400 steps), eval every 100
#
# Output: runs/framerate/<data dir name>/<mode>/checkpoint-*
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:?pilot or full}
DATA=${2:?prepared data dir (framerate/prepare_data.py --out)}
FPS=${3:?frames per second}
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
OUT=runs/framerate/$(basename "$DATA")/$MODE

export FPS FPS_MAX_FRAMES=768 VIDEO_MIN_TOKEN_NUM=128 VIDEO_MAX_TOKEN_NUM=128
export FORCE_QWENVL_VIDEO_READER=torchcodec TORCHCODEC_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARGS=(--model "$MODEL" --use_hf true --split_dataset_ratio 0 --dataset "$DATA/train.jsonl"
  --tuner_type lora --target_modules all-linear --freeze_vit true --freeze_aligner true --lora_dropout 0.05
  --torch_dtype bfloat16 --attn_impl sdpa --max_length 32768 --truncation_strategy delete --loss_scale default
  --learning_rate 1e-4 --lr_scheduler_type cosine --weight_decay 0.1 --adam_beta1 0.9 --adam_beta2 0.95
  --optim adamw_torch_fused --per_device_train_batch_size 1 --per_device_eval_batch_size 1
  --gradient_accumulation_steps 8 --gradient_checkpointing true --packing false --padding_free false
  --dataloader_num_workers 4 --logging_steps 1 --report_to tensorboard --seed 42 --data_seed 42
  --add_version false --output_dir "$OUT")

case "$MODE" in
  pilot)  # time-capped comparison: same number of optimizer steps for every frame rate, no periodic eval
    STEPS=${4:?optimizer steps}
    ARGS+=(--lora_rank 16 --lora_alpha 32 --num_train_epochs 3 --max_steps "$STEPS" --warmup_ratio 0.05
      --eval_strategy no --save_strategy steps --save_steps "$STEPS") ;;
  full)
    ARGS+=(--lora_rank 64 --lora_alpha 128 --num_train_epochs 2 --warmup_ratio 0.03 --lazy_tokenize true
      --val_dataset "$DATA/val50.jsonl" --eval_strategy steps --eval_steps 100
      --save_strategy steps --save_steps 25 --save_total_limit 2) ;;
  *) echo "mode must be pilot or full" >&2; exit 2 ;;
esac

LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[ -n "$LAST" ] && ARGS+=(--resume_from_checkpoint "$LAST") && echo "resume from $LAST"
mkdir -p "$OUT"
printf '%q ' swift sft "${ARGS[@]}" > "$OUT/command.txt"
swift sft "${ARGS[@]}"
