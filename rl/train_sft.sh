#!/usr/bin/env bash
# ReST-EM M-step: LoRA SFT from the BASE model on <dataset>, same settings as framerate/train.sh full except the
# dataset and the number of epochs (1 epoch over GT + kept samples ~ the same number of sample passes as
# full2fps' 2 epochs over GT only). Resumes from the newest checkpoint.
#   bash rl/train_sft.sh $W/runs/B1/rest_train.jsonl $W/runs/B1/sft [epochs]
set -euo pipefail
DATA=$1; OUT=$2; EPOCHS=${3:-1}
W=${W:-/workspace/mpm}
export FPS=2 FPS_MAX_FRAMES=768 VIDEO_MIN_TOKEN_NUM=128 VIDEO_MAX_TOKEN_NUM=128
export FORCE_QWENVL_VIDEO_READER=torchcodec TORCHCODEC_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="$(cat $W/model_path.txt)"
ARGS=(--model "$MODEL" --use_hf true --split_dataset_ratio 0
  --dataset "$DATA" --val_dataset "$W/fps_2s/data/val50.jsonl"
  --tuner_type lora --target_modules all-linear --freeze_vit true --freeze_aligner true
  --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 --torch_dtype bfloat16 --attn_impl sdpa
  --max_length 32768 --truncation_strategy delete --loss_scale default --lazy_tokenize true
  --learning_rate 0.0001 --num_train_epochs $EPOCHS --lr_scheduler_type cosine --warmup_ratio 0.03
  --weight_decay 0.1 --adam_beta1 0.9 --adam_beta2 0.95 --optim adamw_torch_fused
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1
  --gradient_accumulation_steps 8 --gradient_checkpointing true --packing false --padding_free false
  --dataloader_num_workers 4 --logging_steps 1
  --eval_strategy steps --eval_steps 100 --save_strategy steps --save_steps 50 --save_total_limit 2
  --report_to tensorboard --seed 42 --data_seed 42 --add_version false --output_dir "$OUT")
LAST=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[[ -n "$LAST" ]] && ARGS+=(--resume_from_checkpoint "$LAST") && echo "resume from $LAST"
mkdir -p "$OUT"
printf '%q ' swift sft "${ARGS[@]}" >> "$OUT/command.txt"; echo >> "$OUT/command.txt"
${TRAIN:-/opt/mpm/env/train/bin}/swift sft "${ARGS[@]}"
