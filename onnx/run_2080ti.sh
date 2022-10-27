#!/usr/bin/bash
#SBATCH --job-name bm-2080Ti
#SBATCH --time 1-00:00:00
#SBATCH --partition CLUSTER
#SBATCH --mem 16GB
#SBATCH --gres gpu:2080Ti:1
#SBATCH --out 2080Ti_%a.out
#SBATCH --array 0-1

module purge
module load cuda-11.1.1
module load cudnn-11.1.1-v8.0.4.3

# Load Virtual Env
source activate device_benchmarking;

PLATFORM="2080ti"
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
  python3 run_onnx.py \
      --exp_name $EXP_NAME --results_dir onnx_results \
      --model_name $MODEL --model_dir onnx_models \
      --device cuda --platform $PLATFORM --iters 100 $EXP_FLAG;
done;
