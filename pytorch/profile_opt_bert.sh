#!/usr/bin/bash
PLATFORM=rtx8000
DEVICE=cuda

#### PARAMETERS ####
mamba activate device_benchmarking;
declare -a VISION_MODELS=("resnet50");
declare -a LANG_MODELS=("bert")
declare -a BATCH_SIZES=(128);
declare ITERS=20;

declare PROFILE_DIR="ncu-profiles";
declare METRICS="--section LaunchStats --section ComputeWorkloadAnalysis
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section Occupancy
    --section SpeedOfLight --section WarpStateStats --section SourceCounters";
declare METRICS="";

export CUDA_VISIBLE_DEVICES=2;

#### BERT Evaluations ###
for MODEL in ${LANG_MODELS[@]}; do
    for BS in ${BATCH_SIZES[@]}; do
   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2023.1.0/ncu \
            --target-processes all --graph-profiling graph $METRICS \
            -f -o ${PROFILE_DIR}/fp16_bert_cg_${BS} \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/transformers.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --iters $ITERS --batch_size $BS \
            --use_fp16 --use_cuda_graphs \
            --results_dir experiments/pretrained --exp_name pretrained-cudagraphs-fp16;
        done;
done;

# for MODEL in ${LANG_MODELS[@]}; do
#     for BS in ${BATCH_SIZES[@]}; do
#   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2023.1.0/ncu \
#             --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
#             -f -o ${PROFILE_DIR}/fp16_bert_jit_${BS} \
#         /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
#             --model $MODEL --platform $PLATFORM --device $DEVICE \
#             --model_config config/models/transformers.yaml \
#             --device_config config/devices/${DEVICE}.yaml \
#             --iters $ITERS --batch_size $BS \
#             --use_fp16  --use_jit \
#             --results_dir experiments/pretrained --exp_name bert-jit-fp16;
#         done;
# done;

