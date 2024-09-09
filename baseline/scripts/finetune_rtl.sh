#!/bin/bash
# Uncomment and set the following variables correspondingly to run this script:

# ################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
# ################## VICUNA ##################

# ################# LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
# ################# LLaMA-2 ##################

################## TINY_LLAMA ##################
# PROMPT_VERSION=tiny_llama
# MODEL_VERSION="TinyLlama-1.1B-Chat-v1.0"
################## TINY_LLAMA ##################

### NOTE: Actually, CLIP-ViT is not used, but still being remained for cosistency with LLaVA style.

export NCCL_P2P_LEVEL=NVL
export WANDB_PROJECT=""
RUN_NAME="RTL-GeLM-7B"
deepspeed --include localhost:0,1,2,3 --master_port 29512 gelm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./datasets \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --vision_tower path/to/clip-vit-large-patch14/ \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --output_dir ./finetuned/${RUN_NAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --run_name $RUN_NAME \
    --tasks "temporal_reasoning" \
    --temporal_reasoning_data "activitynet" \
    --temporal_reasoning_sample_rate 1 \
    --task_sample_rate 1 \
    --input_type feature \
    --num_frames 180 \
    --feature_dim 768 \
    --gnd_enc_layers 1 \
    --d_gnd 1024 \
    --d_proj 256
