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
from config import NAME2MODEL_LANGUAGE, NAME2MODEL_VISION
from utils import fill_metadata, prepare_model, setup_logger


def _build_language_model(model_name, use_jit):
    model_fn, _, checkpoint = NAME2MODEL_LANGUAGE[model_name]
    model = model_fn.from_pretrained(checkpoint, torchscript=use_jit)
    return model


def _build_vision_model(model_name, img_size=224, use_dquant=False, pretrained=False):
    if model_name in "vit32":
        model = NAME2MODEL_VISION[model_name](pretrained=pretrained, img_size=img_size)
    elif model_name == "mobilenet_v2":
        model = NAME2MODEL_VISION[model_name](pretrained=pretrained, quantize=use_dquant)
    else:
        model = NAME2MODEL_VISION[model_name](pretrained=pretrained)
    return model


def _build_language_input(model_name, batch_size, seq_len, device=None):
    _, tokenizer_fn, model_name = NAME2MODEL_LANGUAGE[model_name]
    tokenizer = tokenizer_fn.from_pretrained(model_name)
    id_constructor = partial(
        torch.empty,
        low=0,
        high=tokenizer.vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )
    return id_constructor


def _build_vision_input(batch_size, img_size, device=None, memory_format=torch.channels_last):
    return partial(
        torch.empty,
        device=device,
        size=(batch_size, 3, img_size, img_size),
        memory_format=memory_format,
    )


def main(opts):
    device_name = opts["device"]
    model_name = opts["model"]
    platform_name = opts["platform"]

    # Setup Logging
    results_dir = os.path.join(
        opts["results_dir"], f"{datetime.datetime.now().strftime('%Y_%m%d')}_{opts['exp_name']}"
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(results_dir, platform_name)

    results_file = f"{results_dir}/{platform_name}.csv"
    dataframe = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    # Load model and inputs
    logger.info(f"Loading {model_name} model")
    device = torch.device(device_name)

    # TODO: Crossproduct of input sizes
    for batch_size in opts["batch_size"]:
        for input_size in opts["input_size"]:
            metadata = {
                "model": model_name,
                "batch_size": batch_size,
                "input_size": input_size,
                **fill_metadata(opts),
            }

            if model_name in NAME2MODEL_LANGUAGE:
                input_constructor = _build_language_input(
                    model_name, batch_size, input_size, device
                )
                model = _build_language_model(model_name, opts["use_jit"])
            if model_name in NAME2MODEL_VISION:
                if opts["use_channels_last"]:
                    memory_format = torch.channels_last
                else:
                    memory_format = torch.contiguous_format

                input_constructor = _build_vision_input(
                    batch_size, input_size, device, memory_format
                )
                model = _build_vision_model(model_name, input_size)

            model = prepare_model(model, input_constructor(), opts)
            benchmarker = PyTorchBenchmark(
                model,
                input_constructor,
                opts["num_threads"],
                opts["use_cuda"],
                device_idx=opts["device_idx"],
            )
            data = benchmarker.aggregate_metrics(opts["use_dquant"], opts["use_jit"], opts["iters"])
            logger.info(f"Profile to {results_dir}")
            benchmarker.get_profile(opts["iters"], results_dir)
            results = {**data, **metadata}

            # Combine the run params with the observed metrics
            dataframe = dataframe.append(results, ignore_index=True)
            dataframe.to_csv(results_file, index=False)

    # Teardown model to (hopefully) avoid OOM on next run
    del model
    if opts["use_cuda"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Base Configs
    parser.add_argument("--model", type=str)
    parser.add_argument("--platform", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--use_channels_last", type=bool, default=False)
    parser.add_argument("--use_jit", type=bool, default=False)
    parser.add_argument("--use_dquant", type=bool, default=False)
    # Input Configs
    parser.add_argument("--batch_size", type=list)
    parser.add_argument("--input_size", type=list)
    # Load from Config Files
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--device_config", type=str)
    parser.add_argument("--results_dir", type=str, default="experiments/")
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()

    # Load config
    with open(args.model_config, "r") as model_config:
        model_params = yaml.safe_load(model_config)

    with open(args.device_config, "r") as device_config:
        device_params = yaml.safe_load(device_config)

    all_params = {
        **vars(args),
        **model_params,
        **device_params,
    }
    main(all_params)
