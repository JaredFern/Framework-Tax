import argparse
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import transformers  # Requires sentencepiece
import yaml

from metrics import Benchmark
from model_generator import (  # Conv1DModel,; Conv2DdModel,; TransformerModel,
    FeedForwardModel,
    LstmModel,
    MultiheadAttentionModel,
    RnnModel,
)
from utils import fill_metadata, prepare_model, setup_logger


# TODO: Separate and merge model and device configs
# TODO: Implement self-attention, norm, conv layer modeling
def run_metrics(opts, data, model, input_constructor):
    benchmarker = Benchmark(
        model,
        input_constructor,
        opts["num_threads"],
        opts["use_cuda"],
        device_idx=opts["device_idx"],
    )
    memory_usage = benchmarker.get_memory()
    data["avg_memory"] = np.mean(memory_usage)
    data["max_memory"] = np.max(memory_usage)
    data["latency"] = benchmarker.get_wallclock(opts["iters"])
    if not opts["use_jit"]:
        data["macs"] = benchmarker.get_flops_count()
        data["total_params"] = benchmarker.get_param_count(False)
        data["trainable_params"] = benchmarker.get_param_count(True)

    del model
    if opts["use_cuda"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return data


def run_linear_model(opts, dataframe):
    for batch_size in opts["batch_size"]:
        for hidden_dim in opts["hidden_size"]:
            data = {
                "model": "feedforward",
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                **fill_metadata(opts),
            }

            input_shape = (batch_size, hidden_dim)
            input_constructor = partial(torch.randn, size=input_shape, device=opts["device"])
            model = FeedForwardModel(2 * [hidden_dim], opts["act_fn"])
            model = prepare_model(model, input_constructor(), opts)
            data = run_metrics(opts, data, model, input_constructor)

            # Combine the run params with the observed metrics
            dataframe = dataframe.append(data, ignore_index=True)
    return dataframe


def run_rnn_model(opts, dataframe):
    for batch_size in opts["batch_size"]:
        for hidden_dim in opts["hidden_size"]:
            for seq_len in opts["seq_lens"]:
                data = {
                    "model": "rnn",
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "bidirectional": opts["bidirectional"],
                    **fill_metadata(opts),
                }

                input_shape = (batch_size, seq_len, hidden_dim)
                input_constructor = partial(torch.randn, size=input_shape, device=opts["device"])

                model = RnnModel(
                    2 * [hidden_dim],
                    dropout=opts["dropout"],
                    bidirectional=opts["bidirectional"],
                    activation_function=opts["act_fn"],
                )
                model = prepare_model(model, input_constructor(), opts)

                data = run_metrics(opts, data, model, input_constructor)
                dataframe = dataframe.append(data, ignore_index=True)
    return dataframe


def run_lstm_model(opts, device, dataframe):
    for batch_size in opts["batch_size"]:
        for hidden_dim in opts["hidden_size"]:
            for seq_len in opts["seq_lens"]:
                data = {
                    "model": "lstm",
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "seq_len": seq_len,
                    "bidirectional": opts["bidirectional"],
                    **fill_metadata(opts),
                }

                input_shape = (batch_size, seq_len, hidden_dim)
                input_constructor = partial(torch.randn, size=input_shape, device=device)
                model = LstmModel(
                    2 * [hidden_dim],
                    dropout=opts["dropout"],
                    bidirectional=opts["bidirectional"],
                    activation_function=opts["act_fn"],
                )
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

    if model_name == "feedforward":
        dataframe = run_linear_model(opts, dataframe)
    elif model_name == "rnn":
        dataframe = run_rnn_model(opts, dataframe)
    elif model_name == "lstm":
        dataframe = run_lstm_model(opts, dataframe)

    dataframe.to_csv(results_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
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
