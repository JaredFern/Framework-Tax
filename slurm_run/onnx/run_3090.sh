#!/usr/bin/bash
#SBATCH --job-name bm-3090
#SBATCH --out slurm_out/3090%a.out
#SBATCH --time 1-00:00:00
#SBATCH --partition CLUSTER
#SBATCH --mem 16GB
#SBATCH --gres gpu:3090:1
#SBATCH --array 0-1

# Load Virtual Env
source activate device_benchmarking;

PLATFORM="3090"
DEVICE="cuda"

EXP_TAG=("fp32" "tf32" "fp16" "torchscript" "torchscript-fp16" "trt" "trt-fp16")
EXP_FLAG=(
  " " "--use_tf32" "--use_fp16" "--use_jit" "--use_jit --use_fp16"
  "--use_tensorrt" "--use_tensorrt --use_fp16"
)

EXP_NAME="pretrained-"${EXP_TAG[$SLURM_ARRAY_TASK_ID]}

# Pretrained Vision Models
declare -a VISION_MODELS=(
    "vit32" "efficientnet" "efficientnet_lite" "resnet18" "resnet50" "resnet101" "resnet152"
    "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet" "shufflenet" "mobilenet_v2" "wide_resnet50_2" "mnasnet"
)

for MODEL in ${VISION_MODELS[@]}; do
  python3 pytorch/main_pretrained.py \
    --model $MODEL --model_config pytorch/config/models/vision.yaml ${EXP_FLAG[$SLURM_ARRAY_TASK_ID]} \
    --platform $PLATFORM --device $DEVICE --device_config pytorch/config/devices/${DEVICE}.yaml \
    --results_dir experiments/pretrained --exp_name $EXP_NAME;
done;

# Pretrained Language Models
declare -a LANG_MODELS=(
   "bert" "distilbert" "funnel_transformer" "albert" "longformer" "mobile_bert" "squeeze_bert")

for MODEL in ${LANG_MODELS[@]}; do
  python3 pytorch/main_pretrained.py \
    --model $MODEL --model_config pytorch/config/models/transformers.yaml ${EXP_FLAG[$SLURM_ARRAY_TASK_ID]}\
    --platform $PLATFORM --device $DEVICE --device_config pytorch/config/devices/${DEVICE}.yaml \
    --results_dir experiments/pretrained --exp_name $EXP_NAME;
done;
