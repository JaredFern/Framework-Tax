import argparse
import logging
import os
from functools import partial

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
def run_metrics(opts, data, model, input_constructor):
    benchmarker = PyTorchBenchmark(
        model,
        input_constructor,
        opts["num_threads"],
        opts["use_cuda"],
        device_idx=opts["device_idx"],
    )
    results = benchmarker.aggregate_metrics(opts["use_dquant"], opts["use_jit"], opts["iters"])
    data = {**data, **results}

    del model
    if opts["use_cuda"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return data


def run_model(opts, model_name, device_name, dataframe):
    for batch_size in opts["batch_size"]:
        for num_layers in opts["num_layers"]:
            for seq_len in opts["seq_lens"]:
                for hidden_dim in opts["hidden_size"]:
                    data = {
                        "model": model_name,
                        "device": device_name,
                        "batch_size": batch_size,
                        "hidden_dim": hidden_dim,
                        "seq_len": seq_len,
                        **fill_metadata(opts),
                    }

                    input_shape = (batch_size, seq_len, hidden_dim)
                    input_constructor = partial(
                        torch.randn, size=input_shape, device=opts["device"]
                    )
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
                    data = run_metrics(opts, data, model, input_constructor)

                    # Combine the run params with the observed metrics
                    dataframe = dataframe.append(data, ignore_index=True)
    return dataframe


def main(opts, model_name, device_name, results_dir):
    logger = logging.getLogger(device_name)
    logger.info(f"Loading {model_name} model.")
    results_file = f"{results_dir}/{device_name}.csv"
    dataframe = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    if model_name in ["feedforward", "rnn", "lstm"]:
        dataframe = run_model(opts, model_name, device_name, dataframe)

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
    parser.add_argument("--batch_size", type=list, default=[1])
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--seq_len", type=list, default=[128])
    parser.add_argument("--num_channels", type=list, default=[])
    parser.add_argument("--kernel_size", type=list, default=[])
    parser.add_argument("--stride", type=list, default=[1])
    parser.add_argument("--act_fn", type=str, default=None)
    # Load from Config Files
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--device_config", type=str)
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    args = parser.parse_args()
    # Load config
    with open(args.model_config, "r") as model_config:
        model_params = yaml.safe_load(model_config)

    with open(args.device_config, "r") as device_config:
        device_params = yaml.safe_load(device_config)

    all_params = {**model_params, **device_params}
    setup_logger(args.results_dir, args.device)
    main(all_params, args.model, args.device, args.results_dir)
