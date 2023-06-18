import argparse
import datetime
import logging
import os
from functools import partial
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import yaml
from benchmark import PyTorchBenchmark
from config import NAME2MODEL_LANGUAGE, NAME2MODEL_SPEECH, NAME2MODEL_VISION
from utils import setup_logger


def _build_language_model(model_name, device, use_jit):
    model_fn, _, checkpoint = NAME2MODEL_LANGUAGE[model_name]
    model = model_fn.from_pretrained(checkpoint, torchscript=use_jit).to(device)
    return model


def _build_vision_model(model_name, device, img_size=224, use_dquant=False, pretrained=False):
    if model_name in "vit32":
        model = NAME2MODEL_VISION[model_name](pretrained=pretrained, img_size=img_size).to(device)
    elif model_name == "mobilenet_v2":
        model = NAME2MODEL_VISION[model_name](pretrained=pretrained, quantize=use_dquant).to(device)
    else:
        model = NAME2MODEL_VISION[model_name](pretrained=pretrained).to(device)
    return model


def _build_speech_model(model_name, device, use_jit):
    model_fn, _, model_name = NAME2MODEL_SPEECH[model_name]
    model = model_fn.from_pretrained(model_name, torchscript=use_jit).to(device)
    return model


def _build_language_input(model_name, batch_size, seq_len, device=None):
    _, tokenizer_fn, model_name = NAME2MODEL_LANGUAGE[model_name]
    tokenizer = tokenizer_fn.from_pretrained(model_name)
    id_constructor = partial(
        torch.randint,
        low=0,
        high=tokenizer.vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )
    return id_constructor


def _build_speech_input(model_name, batch_size, input_size, sampling_rate=16e3, device=None, dtype=torch.float32):
    # Assume 2 second utterances sampled at 16kHz
        return partial(
            torch.rand,
            device=device,
            dtype=dtype,
            size=(batch_size, int(input_size * sampling_rate))
        )

def _build_vision_input(
    batch_size, img_size, device=None, dtype=torch.float32, memory_format=torch.channels_last
):
    return partial(
        torch.empty,
        device=device,
        dtype=dtype,
        size=(batch_size, 3, img_size, img_size),
        memory_format=memory_format,
    )


def build_model(opts, batch_size, input_size, dtype, device):
    if opts.model_name in NAME2MODEL_LANGUAGE:
        input_constructor = _build_language_input(opts.model_name, batch_size, input_size, device)
        model = _build_language_model(opts.model_name, opts.device, opts.use_jit)
    elif opts.model_name in NAME2MODEL_SPEECH:
        input_constructor = _build_speech_input(opts.model_name, batch_size, input_size, device=device, dtype=dtype)
        model = _build_speech_model(opts.model_name, opts.device, opts.use_jit)
    elif opts.model_name in NAME2MODEL_VISION:
        if opts.use_channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        input_constructor = _build_vision_input(
            batch_size, input_size, device, dtype, memory_format
        )
        model = _build_vision_model(opts.model_name, device, input_size)
    else:
        raise ValueError(f"Model {opts.model_name} not supported.")
    return model, input_constructor


def main(opts):
    device = torch.device(opts.device)
    dtype = torch.float16 if opts.use_fp16 else torch.float32

    # Setup Logging
    results_dir = os.path.join(
        opts.results_dir, f"{datetime.datetime.now().strftime('%Y_%m%d')}_{opts.exp_name}"
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(results_dir, opts.platform)
    results_file = f"{results_dir}/{opts.platform}.csv"
    dataframe = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    # Load model and inputs
    logger.info(f"Loading {opts.model_name} model")

    for batch_size in opts.batch_size:
        logger.info(f"Run Parameters: {opts}")
        for input_size in opts.input_size:
            logger.info(f"Benchmarking {opts.model_name} -- Batch: {batch_size}; Input: {input_size}")
            metadata = {}
            metadata["model"] = opts.model_name
            metadata["batch_size"] = batch_size
            metadata["input_size"] = input_size

            model, input_constructor = build_model(opts, batch_size, input_size, dtype, device)
            benchmarker = PyTorchBenchmark(model, input_constructor, opts)
            data = benchmarker.aggregate_metrics()
            results = {**data, **metadata}

            # Combine the run params with the observed metrics
            dataframe = dataframe.append(results, ignore_index=True)
            dataframe.to_csv(results_file, index=False)

    # Teardown model to (hopefully) avoid OOM on next run
    del model
    if opts.use_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--device_config", type=str)
    args, remain_args = parser.parse_known_args()
    with open(args.model_config, "r") as model_config:
        parser.set_defaults(**yaml.safe_load(model_config))

    with open(args.device_config, "r") as device_config:
        parser.set_defaults(**yaml.safe_load(device_config))

    # Base Configs
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--platform", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)

    # Optimization Params
    parser.add_argument("--requires_grad", action="store_true")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--use_channels_last", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")  # Not Supported for all platforms
    parser.add_argument("--use_jit", action="store_true")
    parser.add_argument("--use_tensorrt", action="store_true")
    parser.add_argument("--use_better_tf", action="store_true")
    parser.add_argument("--use_ipex", action="store_true")
    parser.add_argument("--use_dquant", action="store_true")
    # Load from Config Files
    parser.add_argument("--results_dir", type=str, default="experiments/")
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()

    main(args)
