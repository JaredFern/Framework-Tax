#!/usr/bin/bash
EXP_NAME=$1
PLATFORM=$2
DEVICE=$3

declare -a LAYER_MODELS=("feedforward" "conv2d")


for MODEL in ${LAYER_MODELS[@]}; do
  python3 main_baseline.py \
    --model $MODEL --model_config config/models/templates/${MODEL}.yaml\
    --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
    --results_dir experiments/templates --exp_name $EXP_NAME ;
done;
