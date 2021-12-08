#!/usr/bin/bash
CONFIG_FILE=$1
EXP_DIR=$2
DEVICE=$3

declare -a MODELS=(
    "feedforward"
    # "rnn" "lstm"
    # "conv1d" "conv2d" "mha" "layernorm"
)


for MODEL in ${MODELS[@]}; do
  python3 main_vanilla.py \
    --model $MODEL --model_config config/models/templates/${MODEL}.yaml \
    --device $DEVICE --device_config $CONFIG_FILE \
    --results_dir $EXP_DIR ;
done;
