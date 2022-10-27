#!/usr/bin/bash
#SBATCH --job-name bm-1080Ti
#SBATCH --time 1-00:00:00
#SBATCH --partition CLUSTER
#SBATCH --mem 16GB
#SBATCH --gres gpu:1080Ti:1
#SBATCH --out slurm_out/1080Ti_%a.out
#SBATCH --array 0-1


module load cuda-11.1.1;
module load cudnn-11.1.1-v8.0.4.30;
# Load Virtual Env
source activate device_benchmarking;


PLATFORM="1080ti"
DEVICE="cuda"

EXP_TAG=("fp32" "fp16")
EXP_FLAG=(" " "--use_fp16")

EXP_NAME="pretrained-onnx-"${EXP_TAG[$SLURM_ARRAY_TASK_ID]}

# Pretrained Vision Models
declare -a VISION_MODELS=(
    "vit32" "efficientnet" "efficientnet_lite" "resnet18" "resnet50" "resnet101" "resnet152"
    "alexnet" "squeezenet" "vgg16" "densenet" "inception" "googlenet" "shufflenet" "mobilenet_v2" "wide_resnet50_2" "mnasnet"
)

for MODEL in ${VISION_MODELS[@]}; do
  python3 onnx/run_onnx.py \
    --exp_name $EXP_NAME --results_dir experiments/pretrained \
    --model_name $MODEL --model_dir onnx/onnx_models  ${EXP_FLAG[$SLURM_ARRAY_TASK_ID]} \
    --platform $PLATFORM --device $DEVICE --device_idx 0;
done;

# Pretrained Language Models
declare -a LANG_MODELS=(
   "bert" "distilbert" "funnel_transformer" "albert" "longformer" "mobile_bert" "squeeze_bert")

for MODEL in ${LANG_MODELS[@]}; do
  python3 onnx/run_onnx.py \
    --exp_name $EXP_NAME --results_dir experiments/pretrained \
    --model_name $MODEL --model_dir onnx/onnx_models  ${EXP_FLAG[$SLURM_ARRAY_TASK_ID]} \
    --platform $PLATFORM --device $DEVICE --device_idx 0;
done;
