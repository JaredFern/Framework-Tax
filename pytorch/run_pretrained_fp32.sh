#!/usr/bin/bash
PLATFORM=rtx8000
DEVICE=cuda


MODEL="gpt2"
python3 main_pretrained.py \
  --model_name $MODEL --model_config config/models/transformers.yaml \
  --platform $PLATFORM --device $DEVICE --device_config config/devices/${DEVICE}.yaml \
  --results_dir experiments/pretrained --exp_name camera-ready --use_jit
