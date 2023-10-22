#!/usr/bin/bash
PLATFORM=rtx8000
DEVICE=cuda

#### PARAMETERS ####
mamba activate device_benchmarking;
declare -a VISION_MODELS=("resnet50");
declare -a LANG_MODELS=("bert")
declare -a BATCH_SIZES=(64 1);
declare ITERS=1;

declare PROFILE_DIR="ncu-profiles";
declare METRICS=""; # --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SpeedOfLight --section WarpStateStats";

export CUDA_VISIBLE_DEVICES=3;

            # --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
#### VISION Evaluations ####
for MODEL in ${LANG_MODELS[@]}; do
    for BS in ${BATCH_SIZES[@]}; do
        nsys profile \
             -s cpu -t cuda,nvtx,osrt,cudnn,cublas \
             --cudabacktrace true \
            --capture-range=cudaProfilerApi --nvtx-capture "forward/" \
            --gpu-metrics-device 0 \
            --cuda-memory-usage true \
            --show-output true --stats true -f true -o nsys_test \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/transformers.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --iters $ITERS --batch_size $BS \
            --use_fp16 --use_cuda_graphs \
            --results_dir experiments/pretrained --exp_name pretrained-cudagraphs-fp16;
        done;
done;


