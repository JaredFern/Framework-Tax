#!/usr/bin/bash
PLATFORM=rtx8000
DEVICE=cuda



# declare -a VISION_MODELS=("resnet50")
# for MODEL in ${VISION_MODELS[@]}; do
#     python3 main_pretrained.py \
#         --model $MODEL --model_config config/models/vision.yaml --use_ptcompile \
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name pretrained-fp32;
# done;

MODEL="gpt2"
python3 main_pretrained.py \
  --model_name $MODEL --model_config config/models/transformers.yaml \
  --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
  --results_dir experiments/pretrained --exp_name camera-ready --use_fp16 --use_jit

# python3 main_pretrained.py \
#   --model_name $MODEL --model_config config/models/transformers.yaml \
#   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#   --results_dir experiments/pretrained --exp_name camera-ready --use_onnxrt

# python3 main_pretrained.py \
#   --model_name $MODEL --model_config config/models/transformers.yaml \
#   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#   --results_dir experiments/pretrained --exp_name camera-ready --use_fp16 --use_jit

# python3 main_pretrained.py \
#   --model_name $MODEL --model_config config/models/transformers.yaml \
#   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#   --results_dir experiments/pretrained --exp_name camera-ready --use_fp16 --use_onnxrt

# python3 main_pretrained.py \
#   --model_name $MODEL --model_config config/models/transformers.yaml \
#   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#   --results_dir experiments/pretrained --exp_name camera-ready-seqlen

# python3 main_pretrained.py \
#   --model_name $MODEL --model_config config/models/transformers.yaml \
#   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#   --results_dir experiments/pretrained --exp_name camera-ready-seqlen --use_jit

# python3 main_pretrained.py \
#   --model_name $MODEL --model_config config/models/transformers.yaml \
#   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#   --results_dir experiments/pretrained --exp_name camera-ready-seqlen --use_onnxrt


# for MODEL in ${LANG_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/transformers.yaml \
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name camera-ready --use_fp16 --use_jit
# done;


# for MODEL in ${LANG_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/transformers.yaml \
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name camera-ready --use_fp16 --use_onnxrt
# done;

# for MODEL in ${LANG_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/transformers.yaml \
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name camera-ready
# done;

# for MODEL in ${LANG_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/transformers.yaml \
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name camera-ready --use_jit
# done;

# for MODEL in ${LANG_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/transformers.yaml \
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name camera-ready --use_onnxrt
# done;


# declare -a SPEECH_MODELS=("hubert")
# for MODEL in ${SPEECH_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/speech.yaml\
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name pretrained-fp16 --use_fp16;
# done;

# declare -a SPEECH_MODELS=("hubert")
# for MODEL in ${SPEECH_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/speech.yaml\
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name pretrained-fp32;
# done;

# declare -a SPEECH_MODELS=("hubert")
# for MODEL in ${SPEECH_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/speech.yaml\
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name pretrained-jit-fp32 --use_jit;
# done;


# declare -a SPEECH_MODELS=("hubert")
# for MODEL in ${SPEECH_MODELS[@]}; do
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/speech.yaml\
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name pretrained-jit-fp16 --use_jit --use_fp16;

# declare -a SPEECH_MODELS=("wavlm")
# MODEL="wavlm"
#       # python3 main_pretrained.py \
#       #   --model_name $MODEL --model_config config/models/speech.yaml\
#       #   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#       #   --results_dir experiments/pretrained --exp_name pretrained-fp32;

#       # python3 main_pretrained.py \
#       #   --model_name $MODEL --model_config config/models/speech.yaml\
#       #   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#       #   --results_dir experiments/pretrained --exp_name pretrained-fp16 --use_fp16;
#       # python3 main_pretrained.py \
#       #   --model_name $MODEL --model_config config/models/speech.yaml\
#       #   --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#       #   --results_dir experiments/pretrained --exp_name pretrained-fp32-jit --use_jit;
#       python3 main_pretrained.py \
#         --model_name $MODEL --model_config config/models/speech.yaml\
#         --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
#         --results_dir experiments/pretrained --exp_name pretrained-fp16-jit --use_jit --use_fp16
