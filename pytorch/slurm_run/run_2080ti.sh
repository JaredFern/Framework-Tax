#!/usr/bin/bash
#SBATCH --job-name bm-2080Ti
#SBATCH --out slurm_out/2080Ti.out
#SBATCH --time 1-00:00:00
#SBATCH --partition CLUSTER
#SBATCH --mem 32GB
#SBATCH --gres gpu:2080Ti:1

# Load Virtual Env
source activate device_benchmarking;

PLATFORM="2080Ti"
EXP_NAME="pretrained-fp16"
DEVICE="cuda"

# declare -a VISION_MODELS=(
#     "vit32" "efficientnet" "efficientnet_lite" "gernet"
    # "resnet18" "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet"
    # "shufflenet" "mobilenet_v2" "resnext50_32x4d" "wide_resnet50_2" "mnasnet"
# )


# for MODEL in ${VISION_MODELS[@]}; do
  # python3 main_pretrained.py \
    # --model $MODEL --model_config config/models/vision.yaml --use_fp16 \
    # --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
    # --results_dir experiments/pretrained --exp_name $EXP_NAME ;
# done;

# declare -a LANG_MODELS=(
   # "bert" "distilbert" "funnel_transformer" "albert" "longformer" "mobile_bert" "squeeze_bert")

# for MODEL in ${LANG_MODELS[@]}; do
  # python3 main_pretrained.py \
    # --model $MODEL --model_config config/models/transformers.yaml --use_fp16 True \
    # --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
    # --results_dir experiments/pretrained --exp_name $EXP_NAME ;
# done;
