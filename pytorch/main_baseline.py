import argparse
import datetime
import logging
import os
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers  # Requires sentencepiece
import yaml
from benchmark import PyTorchBenchmark
from model_generator import Conv1DModel  # ; Conv2DdModel,; TransformerModel,
from model_generator import (  # MultiheadAttentionModel,
    FeedForwardModel,
    LstmModel,
    RnnModel,
)
from utils import fill_metadata, prepare_model, setup_logger


# TODO: Implement self-attention, norm, conv layer modeling
def run_metrics(opts, data, model, input_constructor, results_dir):
    benchmarker = PyTorchBenchmark(
        model,
        input_constructor,
        opts["num_threads"],
        opts["use_cuda"],
        device_idx=opts["device_idx"],
    )
    results = benchmarker.aggregate_metrics(opts["use_dquant"], opts["use_jit"], opts["iters"])
    benchmarker.get_profile(opts["iters"], results_dir)
    data = {**data, **results}

    del model
    if opts["use_cuda"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return data


def run_model(opts, model_name, input_shape, dataframe, results_dir):
    for hidden_dim in opts["hidden_size"]:
        for num_layers in opts["num_layers"]:
            input_constructor = partial(torch.randn, size=input_shape, device=opts["device"])
            if model_name == "feedforward":
                model = FeedForwardModel(num_layers * [hidden_dim], opts["act_fn"])
            elif model_name == "rnn":
                model = RnnModel(
                    num_layers * [hidden_dim],
                    dropout=opts["dropout"],
                    bidirectional=opts["bidirectional"],
                    activation_function=opts["act_fn"],
                )
            elif model_name == "lstm":
                model = LstmModel(
                    num_layers * [hidden_dim],
                    dropout=opts["dropout"],
                    bidirectional=opts["bidirectional"],
                    activation_function=opts["act_fn"],
                )
            elif model_name == "conv1d":
                model = Conv1DModel(
                    num_channels=opts["num_channels"],
                    kernel_sizes=opts["kernel_sizes"],
                    strides=opts["strides"],
                    paddings=opts["paddings"],
                    groups=opts["groups"],
                )
            elif model_name == "self_attn":
                pass

            model = prepare_model(model, input_constructor(), opts)
            metadata = {
                "model": model_name,
                "device": opts["device"],
                "platform": opts["platform"],
                "input_format": opts["input_format"],
                "num_layers": num_layers,
                "input_size": input_shape,
                **fill_metadata(opts),
            }
            data = run_metrics(opts, dataframe, model, input_constructor, results_dir)
            results = {**data, **metadata}

            # Combine the run params with the observed metrics
            dataframe = dataframe.append(results, ignore_index=True)
    return dataframe


def main(opts, model_name, device_name, results_dir):
    platform_name = opts["platform"]
    results_dir = os.path.join(
        opts["results_dir"], f"{datetime.datetime.now().strftime('%Y_%m%d')}_{opts['exp_name']}"
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_file = f"{results_dir}/{platform_name}.csv"
    dataframe = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    if opts["input_format"] == "bh":
        input_sizes = product(opts["batch_size"], opts["hidden_size"])
    if opts["input_format"] == "bsh":
        input_sizes = product(opts["batch_size"], opts["seq_len"], opts["hidden_size"])
    elif opts["input_format"] == "bchw":
        input_sizes = product(
            opts["batch_size"], opts["num_channels"], opts["img_size"], opts["img_size"]
        )

    for input_size in input_sizes:
        dataframe = run_model(opts, model_name, input_size, dataframe, results_dir)

    dataframe.to_csv(results_file, index=False)


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
    parser.add_argument("--use_jit", type=bool, default=False)
    parser.add_argument("--use_dquant", type=bool, default=False)
    # Input and Operator Configs
    parser.add_argument("--input_format", type=str)
    parser.add_argument("--batch_size", type=list, default=[1])
    parser.add_argument("--hidden_size", type=int, default=[256])
    parser.add_argument("--seq_len", type=list, default=[128])
    parser.add_argument("--num_channels", type=list, default=[])
    parser.add_argument("--kernel_size", type=list, default=[])
    parser.add_argument("--stride", type=list, default=[1])
    parser.add_argument("--act_fn", type=str, default=None)
    # Load from Config Files
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--device_config", type=str)
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()
    # Load config
    with open(args.model_config, "r") as model_config:
        model_params = yaml.safe_load(model_config)

    with open(args.device_config, "r") as device_config:
        device_params = yaml.safe_load(device_config)

    all_params = {**model_params, **device_params, **vars(args)}
    main(all_params, args.model, args.device, args.results_dir)
