import argparse
import os
import time
import timeit

import pandas as pd
import torch
import transformers  # Requires sentencepiece
import yaml

from metrics import gather_metrics
from utils import _get_input_ids, _get_logger

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


def main(opts, device, model_name, metrics, results_dir):
    logger = _get_logger(results_dir, device)
    results_fname = f"{results_dir}/{device}.csv"
    dataframe = (
        pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()
    )

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

        input_ids = _get_input_ids(
            tokenizer, seq_len, opts["randomized_text"], checkpoint
        )
        if opts["use_cuda"]:
            input_ids = input_ids.to("cuda")
            model.to("cuda")

        # Compute statistics over individual wallclock times
        results = gather_metrics(opts, model, input_ids, metrics, logger)

        # Combine the run params with the observed metrics
        data.update(results)
        dataframe = dataframe.append(data, ignore_index=True)

    dataframe.to_csv(results_fname, index=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--metrics", choices=["wallclock", "flops", "memory", "params"], nargs="+"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config_file, "r") as config_file:
        params = yaml.safe_load(config_file)

    if args.model == "all":
        for model in NAME2MODEL.keys():
            main(params, args.device, model, args.metrics, args.results_dir)
    else:
        main(params, args.device, args.model, args.metrics, args.results_dir)

