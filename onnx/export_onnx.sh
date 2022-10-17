#!/usr/bin/bash
declare -a VISION_MODELS=(
    # "vit32" "efficientnet" "efficientnet_lite" "gernet"
    "resnet18" "resnet50" "resnet101" "resnet152" 
    # "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet"
    # "shufflenet" "mobilenet_v2" "resnext50_32x4d" "wide_resnet50_2" "mnasnet"
)

for MODEL in ${VISION_MODELS[@]}; do
        python3 export_onnx_models.py --model_name $MODEL --model_dir onnx_models --use_fp16;
done;