#!/usr/bin/bash
PLATFORM=rtx8000
DEVICE=cuda

#### PARAMETERS ####
mamba activate device_benchmarking;
declare -a VISION_MODELS=("resnet50");
declare -a LANG_MODELS=("bert")
declare -a BATCH_SIZES=(1 2 4 8 32 128);
declare ITERS=100;
declare PRECISION="--use_fp16";

declare PROFILE_DIR="ncu-profiles";
declare METRICS="--section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SpeedOfLight --section WarpStateStats";

export CUDA_VISIBLE_DEVICES=0;

#### BERT Baseline Evaluations ###
for MODEL in ${LANG_MODELS[@]}; do
    for BS in ${BATCH_SIZES[@]}; do
   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2023.1.0/ncu \
            --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
            -f -o ${PROFILE_DIR}/fp16_bert_${BS} \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/transformers.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --iters $ITERS  --batch_size $BS \
            $PRECISION \
            --results_dir experiments/pretrained --exp_name bert-fp16;
        done;
done;

#### BERT CUDA Graphs Evaluations ###
for MODEL in ${LANG_MODELS[@]}; do
    for BS in $BATCH_SIZES; do
   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2023.1.0/ncu \
            --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
            -f -o ${PROFILE_DIR}/fp16_bert_cg_${BS} \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/transformers.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --iters $ITERS --batch_size $BS \
            $PRECISION --use_cuda_graphs \
            --results_dir experiments/pretrained --exp_name pretrained-cudagraphs-fp16;
        done;
done;

#### BERT TorchScript Evaluations ###
for MODEL in ${LANG_MODELS[@]}; do
    for BS in $BATCH_SIZES; do
   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2023.1.0/ncu \
            --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
            -f -o ${PROFILE_DIR}/fp16_bert_jit_${BS} \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/transformers.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --iters $ITERS --batch_size $BS \
            $PRECISION --use_jit \
            --results_dir experiments/pretrained --exp_name bert-jit-fp16;
        done;
done;


#### ResNet Baseline Evaluations ####
for MODEL in ${VISION_MODELS[@]}; do
    for BS in $BATCH_SIZES; do
        /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2023.1.0/ncu \
                --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
                -f -o ${PROFILE_DIR}/fp16_resnet_${BS} \
            /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
                --model $MODEL --platform $PLATFORM --device $DEVICE \
                --model_config config/models/vision.yaml \
                --device_config config/devices/${DEVICE}.yaml \
                --iters $ITERS --batch_size $BS \
                $PRECISION \
                --results_dir experiments/pretrained --exp_name vision-fp16;
    done;
done;

#### ResNet TorchScript Evaluations ####
for MODEL in ${VISION_MODELS[@]}; do
    for BS in $BATCH_SIZES; do
   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2023.1.0/ncu \
            --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
            -f -o ${PROFILE_DIR}/fp16_resnet_jit_${BS} \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/vision.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --batch_size $BS --iters $ITERS \
            $PRECISION --use_jit \
            --results_dir experiments/pretrained --exp_name vision-jit-fp16;
        done;
done;

#### ResNet CUDA Graphs Evaluations ####
for MODEL in ${VISION_MODELS[@]}; do
    for BS in $BATCH_SIZES; do
   /home/jaredfer/anaconda3/envs/device_benchmarking/nsight-compute/2023.1.0/ncu \
            --target-processes all --nvtx $METRICS --nvtx-include "forward/" \
            -f -o ${PROFILE_DIR}/fp16_resnet_cg_${BS} \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/vision.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --batch_size $BS --iters $ITERS \
            $PRECISION --use_cuda_graphs \
            --results_dir experiments/pretrained --exp_name vision-cudagraphs-fp16;
        done;
done;

