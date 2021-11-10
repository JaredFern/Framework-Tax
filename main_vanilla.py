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
from utils import prepare_model, setup_logger


def _fill_metadata(opts, data, device):
    data = {
        "device": device,
        "accelerator": opts["use_cuda"],
        "requires_grad": opts["requires_grad"],
        "use_torchscript": opts["use_jit"],
        "dropout": opts["dropout"],
        "act_fn": opts["act_fn"],
    }
    return data


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

    # Teardown model to (hopefully) avoid OOM on next run
    del model
    if opts["use_cuda"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return data


def run_linear_model(opts, device, dataframe):
    for batch_size in opts["batch_size"]:
        for hidden_dim in opts["hidden_size"]:
            data = {
                "device": device,
                "model": "feedforward",
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
            }
            data.update(_fill_metadata(opts, data, device))

            model = FeedForwardModel(2 * [hidden_dim], opts["act_fn"])
            model = prepare_model(
                model, device, opts["requires_grad"], opts["use_dynamic_quant"]
            )
            input_constructor = partial(
                torch.randn,
                size=(batch_size, hidden_dim),
                device=device,
            )

            if opts["use_jit"]:
                eval_model = torch.jit.trace(model, input_constructor())
            else:
                eval_model = model

            data = run_metrics(opts, data, eval_model, input_constructor)

            # Combine the run params with the observed metrics
            dataframe = dataframe.append(data, ignore_index=True)
    print(dataframe)
    return dataframe


def run_rnn_model(opts, device, dataframe):
    for batch_size in opts["batch_size"]:
        for hidden_dim in opts["hidden_size"]:
            for seq_len in opts["seq_lens"]:
                data = {
                    "model": "rnn",
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "bidirectional": opts["bidirectional"],
                }
                data.update(_fill_metadata(opts, data, device))

                model = RnnModel(
                    2 * [hidden_dim],
                    dropout=opts["dropout"],
                    bidirectional=opts["bidirectional"],
                    activation_function=opts["act_fn"],
                )
                model = prepare_model(
                    model, device, opts["requires_grad"], opts["use_dynamic_quant"]
                )
                input_constructor = partial(
                    torch.randn,
                    size=(batch_size, seq_len, hidden_dim),
                    device=device,
                )

                if opts["use_jit"]:
                    eval_model = torch.jit.trace(model, input_constructor())
                else:
                    eval_model = model

                data = run_metrics(opts, data, eval_model, input_constructor)
                print(data)

                # Combine the run params with the observed metrics
                dataframe = dataframe.append(data, ignore_index=True)
    return dataframe


def run_lstm_model(opts, device, dataframe):
    for batch_size in opts["batch_size"]:
        for hidden_dim in opts["hidden_size"]:
            for seq_len in opts["seq_lens"]:
                data = {
                    "model": "rnn",
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "bidirectional": opts["bidirectional"],
                }
                data.update(_fill_metadata(opts, data, device))
                print(data)

                model = LstmModel(
                    2 * [hidden_dim],
                    dropout=opts["dropout"],
                    bidirectional=opts["bidirectional"],
                    activation_function=opts["act_fn"],
                )
                model = prepare_model(
                    model, device, opts["requires_grad"], opts["use_dynamic_quant"]
                )
                input_constructor = partial(
                    torch.randn,
                    size=(batch_size, seq_len, hidden_dim),
                    device=device,
                )

                if opts["use_jit"]:
                    eval_model = torch.jit.trace(model, input_constructor())
                else:
                    eval_model = model

                data = run_metrics(opts, data, eval_model, input_constructor)
                print(data)

                # Combine the run params with the observed metrics
                dataframe = dataframe.append(data, ignore_index=True)
    return dataframe


# def run_conv1d_model(opts, device, dataframe):
#     for batch_size in opts["batch_size"]:
#         for seq_len in opts["sequence_lengths"]:
#             for num_channels in opts["num_channels"]:
#                 data = {
#                     "model": "conv1d",
#                     "batch_size": batch_size,
#                     "num_channels": num_channels,
#                     "kernel_size": kernel_size,
#                     "sequence_length": seq_len,
#                 }
#                 data.update(_fill_metadata(opts, data, device))

#                 model = Conv1DModel(
#                     num_channels,
#                     kernel_sizes,
#                     strides,
#                     paddings,
#                     dilations,
#                     groups,
#                     None,
#                 )
#                 input_constructor = partial(
#                     torch.randint,
#                     low=0,
#                     high=1e4,
#                     size=(batch_size, hidden_dim),
#                     device=device,
#                 )

#                 if opts["use_jit"]:
#                     eval_model = torch.jit.trace(model, input_constructor())
#                 else:
#                     eval_model = model

#                 data = run_metrics(opts, data, eval_model, input_constructor)

#                 # Combine the run params with the observed metrics
#                 dataframe = dataframe.append(data, ignore_index=True)
#     return dataframe


def run_conv2d_model(opts, device, dataframe):
    pass


def run_mha_model(opts, device, dataframe):
    for batch_size in opts["batch_size"]:
        for seq_len in opts["seq_lens"]:
            for num_heads in opts["num_heads"]:
                for hidden_dim in opts["hidden_size"]:
                    data = {
                        "device": device,
                        "model": "mha",
                        "batch_size": batch_size,
                        "hidden_dim": hidden_dim,
                    }
                    data.update(_fill_metadata(opts, data, device))

                    model = MultiheadAttentionModel(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        num_heads,
                    )

                    model = prepare_model(
                        model, device, opts["requires_grad"], opts["use_dynamic_quant"]
                    )
                    input_constructor = partial(
                        torch.randn,
                        size=(batch_size, seq_len, hidden_dim),
                        device=device,
                    )

                    if opts["use_jit"]:
                        eval_model = torch.jit.trace(model, input_constructor())
                    else:
                        eval_model = model

                    data = run_metrics(opts, data, eval_model, input_constructor)

                    # Combine the run params with the observed metrics
                    dataframe = dataframe.append(data, ignore_index=True)

    return dataframe


def main(opts, device_name, model_name, results_dir):
    logger = logging.getLogger(device_name)
    logger.info(f"Loading {model_name} model.")
    results_file = f"{results_dir}/{device_name}.csv"
    dataframe = (
        pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
    )

    # Set Device and Get Appropriate Model Functions
    device = torch.device(opts["device"])

    if model_name == "feedforward":
        dataframe = run_linear_model(opts, device, dataframe)
    elif model_name == "rnn":
        dataframe = run_rnn_model(opts, device, dataframe)
    elif model_name == "lstm":
        dataframe = run_lstm_model(opts, device, dataframe)
    elif model_name == "mha":
        dataframe = run_mha_model(opts, device, dataframe)

    dataframe.to_csv(results_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device__config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    # Load config
    with open(args.model_config, "r") as model_config:
        params = yaml.safe_load(model_config)

    setup_logger(args.results_dir, args.device)
    main(params, args.device, args.model, args.results_dir)
