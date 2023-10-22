#!/usr/bin/bash
PLATFORM=a100
DEVICE=cuda

# declare -a VISION_MODELS=("vit32" "resnet50")
# for MODEL in ${VISION_MODELS[@]}; do
#     python3 main_pretrained.py \
#         --model $MODEL --model_config config/models/vision.yaml\
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name pretrained-fp16 --use_fp16;
# done;   

declare -a LANG_MODELS=("gpt2")
for MODEL in ${LANG_MODELS[@]}; do
   python3 main_pretrained.py \
        --model_name $MODEL --model_config config/models/transformers.yaml\
        --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
        --results_dir experiments/pretrained --exp_name bert-seqlens --use_fp16 --use_jit;
done; 
