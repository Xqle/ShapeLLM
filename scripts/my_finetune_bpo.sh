MODEL_NAME=ShapeLLM_7B_gapartnet

deepspeed --include localhost:0,1,2,3 llava/train/train_bpo.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path qizekun/$MODEL_NAME'_'v1.0 \
    --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path ./checkpoints/recon/large.pth \
    --data_path ./playground/data/shapellm/bpo_error_injection_for_training_debug.json \
    --output_dir ./checkpoints/shapellm/$MODEL_NAME'_'bpo

# CUDA_VISIBLE_DEVICES=1,2,3 python llava/train/train_bpo.py \
#     --model_name_or_path qizekun/$MODEL_NAME'_'v1.0 \
#     --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
#     --vision_tower_path ./checkpoints/recon/large.pth \
#     --data_path ./playground/data/shapellm/bpo_error_injection_for_training_debug.json \
#     --output_dir ./checkpoints/shapellm/$MODEL_NAME'_'bpo_v1.0 