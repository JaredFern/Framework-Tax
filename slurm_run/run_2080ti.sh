#!/usr/bin/bash
#SBATCH --job-name bm-2080Ti
#SBATCH --out slurm_out/2080Ti.out
#SBATCH --time 1-00:00:00
#SBATCH --partition CLUSTER
#SBATCH --mem 32GB
#SBATCH --gres gpu:2080Ti:1
#SBATCH --array 0-3

# Load Virtual Env
source activate device_benchmarking;

PLATFORM="2080Ti"
DEVICE="cuda"
EXP_TAG=("fp32" "fp16" "torchscript" "trt")
EXP_FLAG=(" " "--use_fp16" "--use_jit" "--use_tensorrt")
EXP_NAME="pretrained-"${EXP_TAG[$SLURM_ARRAY_TASK_ID]}

# Pretrained Vision Models
declare -a VISION_MODELS=(
    "vit32" "efficientnet" "efficientnet_lite" "gernet"
    "resnet18" "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet"
    "shufflenet" "mobilenet_v2" "resnext50_32x4d" "wide_resnet50_2" "mnasnet"
)

for MODEL in ${VISION_MODELS[@]}; do
  python3 pytorch/main_pretrained.py \
    --model $MODEL --model_config pytorch/config/models/vision.yaml ${EXP_FLAG[$SLURM_ARRAY_TASK_ID]} \
    --platform $PLATFORM --device $DEVICE --device_config pytorch/config/devices/${DEVICE}.yaml \
    --results_dir pytorch/experiments/pretrained --exp_name $EXP_NAME ;
done;

# Pretrained Language Models
declare -a LANG_MODELS=(
   "bert" "distilbert" "funnel_transformer" "albert" "longformer" "mobile_bert" "squeeze_bert")

for MODEL in ${LANG_MODELS[@]}; do
  python3 pytorch/main_pretrained.py \
    --model $MODEL --model_config pytorch/config/models/transformers.yaml ${EXP_FLAG[$SLURM_ARRAY_TASK_ID]}\
    --platform $PLATFORM --device $DEVICE --device_config pytorch/config/devices/${DEVICE}.yaml \
    --results_dir pytorch/experiments/pretrained --exp_name $EXP_NAME ;
done;
