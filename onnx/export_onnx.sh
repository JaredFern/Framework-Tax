#!/usr/bin/bash
declare -a VISION_MODELS=(
"gpt-2"
)

for MODEL in ${VISION_MODELS[@]}; do
        python3 export_onnx_models.py --model_name $MODEL --model_dir onnx_models;
done;
