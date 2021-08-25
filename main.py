import argparse
import logging
import os
import pickle
import time
import timeit
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers  # Requires sentencepiece
import yaml
from tqdm import tqdm

# TODO: Fork support for torch vision models
# TODO: Fix visualizations to handle expanded metrics

NAME2MODEL = {
    "bert": [transformers.BertModel, transformers.BertTokenizer, "bert-base-uncased"],
    "roberta": [
        transformers.RobertaModel,
        transformers.RobertaTokenizer,
        "roberta-base",
    ],
    "deberta": [
        transformers.DebertaModel,
        transformers.DebertaTokenizer,
        "microsoft/deberta-base",
    ],
    "distilbert": [
        transformers.DistilBertModel,
        transformers.DistilBertTokenizer,
        "distilbert-base-uncased",
    ],
    # "funnel_transformer": [
    #     transformers.FunnelModel,
    #     transformers.FunnelTokenizer,
    #     "funnel-transformer/small",
    # ],
    "ibert": [
        transformers.IBertModel,
        transformers.RobertaTokenizer,
        "kssteven/ibert-roberta-base",
    ],
    "albert": [
        transformers.AlbertModel,
        transformers.AlbertTokenizer,
        "albert-base-v2",
    ],
    "longformer": [
        transformers.LongformerModel,
        transformers.LongformerTokenizer,
        "allenai/longformer-base-4096",
    ],
    "mobile_bert": [
        transformers.MobileBertModel,
        transformers.MobileBertTokenizer,
        "google/mobilebert-uncased",
    ],
    "reformer": [
        transformers.ReformerModel,
        transformers.ReformerTokenizer,
        "google/reformer-enwik8",
    ],
    "squeeze_bert": [
        transformers.SqueezeBertModel,
        transformers.SqueezeBertTokenizer,
        "squeezebert/squeezebert-uncased",
    ],
}


def _get_logger(results_dir, device):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    log_fname = f"{results_dir}/{device}.log"
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(log_fname)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def _get_input_ids(tokenizer, seq_len, random_ids=True, model=""):
    if model == "google/reformer-enwik8":
        input_ids = torch.randint(128, size=(seq_len,))  # Two-Byte Utf-8 Characters
    elif random_ids:
        input_ids = torch.randint(tokenizer.vocab_size, size=(seq_len,))
    else:
        tokens = seq_len * ["a"]
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
    input_ids = input_ids.unsqueeze(0)

    return input_ids


def main(opts, device, model_name, results_dir):
    init_time = time.time()

    logger = _get_logger(results_dir, device)
    model_fn, tokenizer_fn, checkpoint = NAME2MODEL[model_name]

    # Load Model & Tokenizer
    logger.info(f"Loading {model_name} model from {checkpoint}")
    results_fname = f"{results_dir}/{device}.csv"
    dataframe = (
        pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()
    )

    # Quick Fix for reformer tokenizer
    if checkpoint == "google/reformer-enwik8":
        tokenizer = tokenizer_fn.from_pretrained("google/reformer-crime-and-punishment")
    else:
        tokenizer = tokenizer_fn.from_pretrained(checkpoint)

    model = model_fn.from_pretrained(checkpoint)
    model.requires_grad_(opts["requires_grad"])
    model.eval()

    seq_lengths = [
        2 ** seqlen for seqlen in range(10)
    ]  # Eval on sequence lengths up from 1 to 512

    for seq_len in seq_lengths:
        data = {
            "device": device,
            "model": model_name,
            "accelerator": opts["use_cuda"],
            "batch_size": opts["batch_size"],
            "sequence_length": seq_len,
        }
        time_per_example = []

        input_ids = _get_input_ids(tokenizer, seq_len, opts["randomized_text"], checkpoint)
        if opts["use_cuda"]:
            input_ids = input_ids.to("cuda")
            model.to("cuda")

        time_per_example = timeit.repeat(lambda:model(input_ids),
                                         repeat=opts['iters'] + 1, number=1)
        time_per_example = time_per_example[1:]  # Skip the first run as a warmup

        # Compute statistics over individual wallclock times
        data["median"] = np.median(time_per_example)
        data["mean"] = np.mean(time_per_example)
        data["std"] = np.std(time_per_example)
        data["min"] = np.min(time_per_example)
        data["max"] = np.max(time_per_example)
        data["5_pct"] = np.percentile(time_per_example, 5)
        data["95_pct"] = np.percentile(time_per_example, 95)
        dataframe = dataframe.append(data, ignore_index=True)

        logger.info(f"Sequence Length: {seq_len}, Time per example: {data['mean']}")

    logger.info(f"Total Runtime: {time.time() - init_time}")
    dataframe.to_csv(results_fname, index=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    # Load config
    with open(args.config_file, "r") as config_file:
        params = yaml.safe_load(config_file)

    if args.model == "all":
        for model in NAME2MODEL.keys():
            main(params, args.device, model, args.results_dir)
    else:
        main(params, args.device, args.model, args.results_dir)
