#!/usr/bin/bash
EXP_NAME=$1
PLATFORM=$2
DEVICE=$3

declare -a VISION_MODELS=(
    "vit32" "efficientnet" "efficientnet_lite" "gernet"
    "resnet18" "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet"
    "shufflenet" "mobilenet_v2" "resnext50_32x4d" "wide_resnet50_2" "mnasnet"
)


for MODEL in ${VISION_MODELS[@]}; do
  python3 main_pretrained.py \
    --model $MODEL --model_config config/models/vision.yaml\
    --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
    --results_dir experiments/pretrained --exp_name $EXP_NAME ;
done;

declare -a LANG_MODELS=(
   "bert" "distilbert" "funnel_transformer" "albert" "longformer" "mobile_bert" "squeeze_bert")

for MODEL in ${LANG_MODELS[@]}; do
  python3 main_pretrained.py \
    --model $MODEL --model_config config/models/transformers.yaml\
    --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
    --results_dir experiments/pretrained --exp_name $EXP_NAME ;
done;
