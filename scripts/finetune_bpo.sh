MODEL_NAME=ShapeLLM_7B_gapartnet
TAG=bpo
date=$(date +%m%d%H%M)

# echo $date'_'$MODEL_NAME'_'$TAG

deepspeed --include localhost:1 llava/train/train_bpo.py \
    --mm_projector_lr 2e-6 \
    --mm_projector_type mlp2x_gelu \
    --learning_rate 2e-6 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 256 \
    --model_name_or_path qizekun/$MODEL_NAME'_'v1.0 \
    --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path ./checkpoints/recon/large.pth \
    --version v1 \
    --data_path ./playground/data/shapellm/bpo_error_injection_for_training_debug.json \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --output_dir ./checkpoints/shapellm/$date'_'$MODEL_NAME'_'$TAG  \
    --num_train_epochs 50 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True  \
    --lora_enable
    # --image_folder path-to-image-folder \
    # --pretrain_mm_mlp_adapter path-to-projector \
    # --mm_use_im_start_end False \
    # --mm_use_im_patch_token False \