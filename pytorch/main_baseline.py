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
import torch.nn as nn
import transformers  # Requires sentencepiece
import yaml
from benchmark import PyTorchBenchmark
from model_generator import (  # MultiheadAttentionModel,
    Conv1DModel,
    Conv2DModel,
    FeedForwardModel,
    LayerNormModel,
    LstmModel,
    RnnModel,
)
from tqdm import tqdm
from utils import setup_logger


def run_metrics(opts, data, model, input_constructor, results_dir):
    benchmarker = PyTorchBenchmark(model, input_constructor, opts)
    results = benchmarker.aggregate_metrics(opts.iters)
    data = {**data, **results}

    del model
    if opts.use_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return data


def run_model(opts, input_shape, dataframe, results_dir):
    hidden_dim = input_shape[-1]
    for num_layers in opts.num_layers:
        metadata = {
            "model": opts.model,
            "num_layers": num_layers,
            "input_format": opts.input_format,
            "input_size": input_shape,
            "hidden_size": hidden_dim,
            "batch_size": input_shape[0],
        }
        input_constructor = partial(torch.randn, size=input_shape, device=opts.device)
        if opts.model == "feedforward":
            model = FeedForwardModel(num_layers * [hidden_dim], opts.act_fn)
        elif opts.model == "rnn":
            model = RnnModel(
                num_layers * [hidden_dim],
                dropout=opts.dropout,
                bidirectional=opts.bidirectional,
                activation_function=opts.act_fn,
            )
        elif opts.model == "lstm":
            model = LstmModel(
                num_layers * [hidden_dim],
                dropout=opts.dropout,
                bidirectional=opts.bidirectional,
                activation_function=opts.act_fn,
            )
        elif opts.model == "conv1d":
            model = Conv1DModel(
                num_channels=opts.num_channels,
                kernel_sizes=opts.kernel_sizes,
                strides=opts.strides,
                paddings=opts.paddings,
                groups=opts.groups,
            )
        elif opts.model == "conv2d":
            if opts.use_channels_last:
                memory_format = torch.channels_last
            else:
                memory_format = torch.contiguous_format

            input_constructor = partial(
                torch.empty,
                size=(input_shape[0], input_shape[1], input_shape[-1], input_shape[-1]),
                device=opts.device,
                memory_format=memory_format,
            )

            metadata = {
                "input_size": input_constructor().shape,
                "in_channel": input_shape[1],
                "out_channel": input_shape[2],
                "kernel_size": input_shape[3],
                "stride": input_shape[4],
                "img_size": input_shape[5],
            }
            model = Conv2DModel(input_shape[1], input_shape[2], input_shape[3], input_shape[4])
        elif opts.model == "self_attn":
            pass
        elif opts.model == "layer_norm":
            model = LayerNormModel(num_layers * [hidden_dim])

        data = run_metrics(opts, dataframe, model, input_constructor, results_dir)
        results = {**data, **metadata, **vars(opts)}

        # Combine the run params with the observed metrics
        dataframe = dataframe.append(results, ignore_index=True)
    return dataframe


def main(opts):
    platform_name = opts.platform
    results_dir = os.path.join(
        opts.results_dir, f"{datetime.datetime.now().strftime('%Y_%m%d')}_{opts.exp_name}"
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_file = f"{results_dir}/{platform_name}.csv"
    dataframe = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    if opts.input_format == "bh":
        input_sizes = product(opts.batch_size, opts.hidden_size)
    if opts.input_format == "bsh":
        input_sizes = product(opts.batch_size, opts.seq_len, opts.hidden_size)
    elif opts.input_format == "bchw":
        input_sizes = product(
            opts.batch_size,
            opts.in_channels,
            opts.out_channels,
            opts.kernel_size,
            opts.stride,
            opts.img_size,
        )

    for input_size in tqdm(list(input_sizes)):
        dataframe = run_model(opts, input_size, dataframe, results_dir)

    dataframe.to_csv(results_file, mode="a", index=False)


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
    parser.add_argument("--model", type=str)
    parser.add_argument("--platform", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--requires_grad", type=bool, default=False)
    # Optimization Params
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--use_channel_last", type=bool, default=False)
    parser.add_argument("--use_fp16", type=bool, default=False)
    parser.add_argument("--use_jit", type=bool, default=False)
    parser.add_argument("--use_tensorrt", type=bool, default=False)
    parser.add_argument("--use_ipex", type=bool, default=False)
    parser.add_argument("--use_dquant", type=bool, default=False)
    # Input and Operator Configs
    parser.add_argument("--input_format", type=str)
    parser.add_argument("--batch_size", type=list)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--seq_len", type=list)
    parser.add_argument("--num_channels", type=list)
    parser.add_argument("--kernel_size", type=list)
    parser.add_argument("--stride", type=list)
    parser.add_argument("--act_fn", type=str)
    # Logging Configs
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()

    main(args)
