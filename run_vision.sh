#!/usr/bin/bash
CONFIG_FILE=$1
EXP_DIR=$2
DEVICE=$3

declare -a MODELS=(
    "vit32" "efficientnet" "efficientnet_lite" "gernet"
    "resnet18" "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet"
    "shufflenet" "mobilenet_v2" "resnext50_32x4d" "wide_resnet50_2" "mnasnet"
)


for MODEL in ${MODELS[@]}; do
  python3 main_vision.py \
    --model $MODEL --model_config config/models/vision.yaml\
    --device $DEVICE --device_config $CONFIG_FILE \
    --results_dir $EXP_DIR ;
done;
