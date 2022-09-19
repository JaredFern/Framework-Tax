#!/usr/bin/bash
#SBATCH --job-name bm-v100
#SBATCH --out slurm_out/v100.out
#SBATCH --time 1-00:00:00
#SBATCH --partition CLUSTER
#SBATCH --mem 32GB
#SBATCH --gres gpu:v100:1
#SBATCH --array 1-4

# Load Virtual Env
source activate device_benchmarking;

PLATFORM="v100"
EXP_NAME="pretrained"
DEVICE="cuda"

EXP_TAG=("fp32" "fp16" "torchscript" "trt")
EXP_FLAG=("" "--use_fp16" "--use_jit" "--use_tensorrt")

declare -a VISION_MODELS=(
    "vit32" "efficientnet" "efficientnet_lite" "gernet"
    "resnet18" "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet"
    "shufflenet" "mobilenet_v2" "resnext50_32x4d" "wide_resnet50_2" "mnasnet"
)

for MODEL in ${VISION_MODELS[@]}; do
  python3 main_pretrained.py \
    --model $MODEL --platform $PLATFORM --device $DEVICE $EXP_FLAG[$SLURM_JOB_ID] \
    --model_config config/models/vision.yaml --device_config config/devices/${DEVICE}.yaml \
    --results_dir experiments/pretrained --exp_name $EXP_NAME-$EXP_TAG[$SLURM_JOB_ID];
done;

# Pretrained Langauge Models
declare -a LANG_MODELS=(
   "bert" "distilbert" "funnel_transformer" "albert" "longformer" "mobile_bert" "squeeze_bert")

for MODEL in ${LANG_MODELS[@]}; do
  python3 main_pretrained.py \
    --model $MODEL --platform $PLATFORM --device $DEVICE $EXP_FLAG[$SLURM_JOB_ID] \
    --model_config config/models/transformers.yaml --device_config config/devices/${DEVICE}.yaml \
    --results_dir experiments/pretrained --exp_name $EXP_NAME-$EXP_TAG[$SLURM_JOB_ID] ;
done;
