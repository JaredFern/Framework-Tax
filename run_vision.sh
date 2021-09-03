#!/usr/bin/bash
CONFIG_FILE=$1
EXP_DIR=$2
DEVICE=$3

declare -a MODELS=(
    "vit32" "efficientnet" "efficientnet_lite" "gernet"
    "resnet18" "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet"
    "shufflenet" "mobilenet_v2" "resnext50_32x4d" "wide_resnet50_2" "mnasnet"
)


for model in ${MODELS[@]}; do
  python3 main_vision.py \
    --config_file $CONFIG_FILE --results_dir $EXP_DIR \
    --device $DEVICE --model $model;
done;
