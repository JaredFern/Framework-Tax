#!/usr/bin/bash
CONFIG_FILE=$1
EXP_DIR=$2
DEVICE=$3

declare -a MODELS=(
  "bert" "roberta" "deberta" "distilbert" "funnel_transformer" "albert"
  "longformer" "mobile_bert" "squeeze_bert"
)


for model in ${MODELS[@]}; do
  python3 main_language.py \
    --config_file $CONFIG_FILE --results_dir $EXP_DIR \
    --device $DEVICE --model $model;
done;
