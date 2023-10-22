#!/usr/bin/bash
PLATFORM=rtx8000
DEVICE=cuda

#### PARAMETERS ####
mamba activate device_benchmarking;
declare -a VISION_MODELS=("resnet50");
declare -a LANG_MODELS=("bert")
declare ITERS=100;
declare PRECISION="--use_fp16";

declare PROFILE_DIR="nsys-profiles";
declare METRICS="--section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section Occupancy --section SpeedOfLight --section WarpStateStats";

export CUDA_VISIBLE_DEVICES=0;

# # #### BERT Baseline Evaluations ###
# for MODEL in ${LANG_MODELS[@]}; do
#         # nsys profile \
#         #      -s cpu -t cuda,nvtx,osrt,cudnn,cublas \
#         #      --cudabacktrace true \
#         #     --capture-range=cudaProfilerApi --nvtx-capture "forward/" \
#         #     --gpu-metrics-device 0 \
#         #     --cuda-memory-usage true \
#         #     -f  true -o ${PROFILE_DIR}/fp16_bert_pytorch \
#         /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
#             --model $MODEL --platform $PLATFORM --device $DEVICE \
#             --model_config config/models/transformers.yaml \
#             --device_config config/devices/${DEVICE}.yaml \
#             --iters $ITERS \
#             $PRECISION  -\
#             --results_dir experiments/pretrained --exp_name ${MODEL}-fp16;
#         done;

# #### BERT CUDA Graphs Evaluations ###
for MODEL in ${LANG_MODELS[@]}; do
        nsys profile \
            -s cpu -t cuda,nvtx,osrt,cudnn,cublas \
            --cudabacktrace true \
            --capture-range=cudaProfilerApi --nvtx-capture "forward/" \
            --gpu-metrics-device 0 \
            -f true -o ${PROFILE_DIR}/fp16_bert_pytorch_cudagraphs \
        /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
            --model $MODEL --platform $PLATFORM --device $DEVICE \
            --model_config config/models/transformers.yaml \
            --device_config config/devices/${DEVICE}.yaml \
            --iters $ITERS \
            $PRECISION --use_cuda_graphs \
            --results_dir experiments/pretrained --exp_name pretrained-${MODEL}-cudagraphs-fp16;
        done;

# #### BERT TorchScript Evaluations ###
# for MODEL in ${LANG_MODELS[@]}; do
#         nsys profile \
#             -s cpu -t cuda,nvtx,osrt,cudnn,cublas \
#             --cudabacktrace true \
#             --capture-range=cudaProfilerApi --nvtx-capture "forward/" \
#             --gpu-metrics-device 0 \
#             -f true -o ${PROFILE_DIR}/fp16_bert_pytorch_jit_${BS} \
#         /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
#             --model $MODEL --platform $PLATFORM --device $DEVICE \
#             --model_config config/models/transformers.yaml \
#             --device_config config/devices/${DEVICE}.yaml \
#             --iters $ITERS $PRECISION --use_jit \
#             --results_dir experiments/pretrained --exp_name ${MODEL}-jit-fp16;
# done;


# #### ResNet Baseline Evaluations ####
# for MODEL in ${VISION_MODELS[@]}; do
#         nsys profile \
#             -s cpu -t cuda,nvtx,osrt,cudnn,cublas \
#             --cudabacktrace true \
#             --capture-range=cudaProfilerApi --nvtx-capture "forward/" \
#             --gpu-metrics-device 0 \
#             -f true -o ${PROFILE_DIR}/fp16_resnet_pytorch \
#         /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
#                 --model $MODEL --platform $PLATFORM --device $DEVICE \
#                 --model_config config/models/vision.yaml \
#                 --device_config config/devices/${DEVICE}.yaml \
#                 --iters $ITERS \
#                 $PRECISION \
#                 --results_dir experiments/pretrained --exp_name vision-fp16;
# done;

# #### ResNet TorchScript Evaluations ####
# for MODEL in ${VISION_MODELS[@]}; do
#         nsys profile \
#             -s cpu -t cuda,nvtx,osrt,cudnn,cublas \
#             --cudabacktrace true \
#             --capture-range=cudaProfilerApi --nvtx-capture "forward/" \
#             --gpu-metrics-device 0 \
#             -f true -o ${PROFILE_DIR}/fp16_resnet_pytorch_jit \
#         /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
#             --model $MODEL --platform $PLATFORM --device $DEVICE \
#             --model_config config/models/vision.yaml \
#             --device_config config/devices/${DEVICE}.yaml \
#              --iters $ITERS \
#             $PRECISION --use_jit \
#             --results_dir experiments/pretrained --exp_name ${MODEL}-jit-fp16;
# done;

# #### ResNet CUDA Graphs Evaluations ####
# for MODEL in ${VISION_MODELS[@]}; do
#         nsys profile \
#             -s cpu -t cuda,nvtx,osrt,cudnn,cublas \
#             --cudabacktrace true \
#             --capture-range=cudaProfilerApi --nvtx-capture "forward/" \
#             --gpu-metrics-device 0 \
#             -f true -o ${PROFILE_DIR}/fp16_resnet_pytorch_cudagraphs \
#         /home/jaredfer/anaconda3/envs/device_benchmarking/bin/python main_pretrained.py \
#             --model $MODEL --platform $PLATFORM --device $DEVICE \
#             --model_config config/models/vision.yaml \
#             --device_config config/devices/${DEVICE}.yaml \
#              --iters $ITERS \
#             $PRECISION --use_cuda_graphs \
#             --results_dir experiments/pretrained --exp_name ${MODEL}-cudagraphs-fp16;
# done;

