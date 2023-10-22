#!/usr/bin/bash
PLATFORM=rtx8000
DEVICE=cuda

#### PARAMETERS ####
mamba activate device_benchmarking;
declare -a VISION_MODELS=("resnet50");
declare -a LANG_MODELS=("bert")
declare -a BATCH_SIZES=(1 2 4 8 16 32 64);
declare ITERS=20;

declare PROFILE_DIR="ncu-profiles";
declare METRICS="--section LaunchStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SpeedOfLight --section WarpStateStats";

export CUDA_VISIBLE_DEVICES=3;

#### VISION Evaluations ####
for MODEL in ${VISION_MODELS[@]}; do
    for BS in $BATCH_SIZES; do
   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2022.1.0/ncu \
            --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
            -f -o ${PROFILE_DIR}/fp16_resnet_jit_${BS} \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/vision.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --batch_size $BS --iters $ITERS \
            --use_fp16 --use_jit \
            --results_dir experiments/pretrained --exp_name vision-jit-fp16;
        done;
done;

for MODEL in ${VISION_MODELS[@]}; do
    for BS in $BATCH_SIZES; do
   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2022.1.0/ncu \
            --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
            -f -o ${PROFILE_DIR}/fp16_resnet_cg_${BS} \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/vision.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --batch_size $BS --iters $ITERS \
            --use_fp16 --use_cuda_graphs \
            --results_dir experiments/pretrained --exp_name vision-cudagraphs-fp16;
        done;
done;



