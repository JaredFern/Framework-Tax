import argparse
import logging
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import transformers  # Requires sentencepiece
import yaml
from tqdm import tqdm

# TODO: Refactor Model Config Yamls

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
    "funnel_transformer": [
        transformers.FunnelModel,
        transformers.FunnelTokenizer,
        "funnel-transformer/small",
    ],
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
        "google/reformer-enwik8",  # Wiki8 ckpt not avail.t pul
    ],
    "squeeze_bert": [
        transformers.SqueezeBertModel,
        transformers.SqueezeBertTokenizer,
        "squeezebert/squeezebert-uncased",
    ],
}


def _get_logger(model_name, device):
    log_fname = f"log/{model_name}_{device}.log"
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def _get_input_ids(tokenizer, seq_len, random_ids=True, char_level=True):
    if char_level:
        input_ids = torch.randint(128, size=(seq_len,))  # Two-Byte Utf-8 Characters
    elif random_ids:
        input_ids = torch.randint(tokenizer.vocab_size, size=(seq_len,))
    else:
        tokens = seq_len * ["a"]
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
    input_ids = input_ids.unsqueeze(0)
    return input_ids


def main(opts, model_name):
    init_time = time.time()
    logger = _get_logger(model_name, opts["device"])
    model_fn, tokenizer_fn, checkpoint = NAME2MODEL[model_name]

    # Load Model & Tokenizer
    logger.info(f"Loading {model_name} model from {checkpoint}")

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

    results = defaultdict(list)
    for seq_len in seq_lengths:
        time_per_example = []

        # Warm up with a single example
        input_ids = _get_input_ids(tokenizer, seq_len, opts["randomized_text"])
        _ = model(input_ids)

        for iter in tqdm(
            range(opts["iters"]),
            desc=f"Batch Size: {opts['batch_size']}, Sequence Length: {seq_len}",
        ):
            input_ids = _get_input_ids(tokenizer, seq_len, opts["randomized_text"])
            if opts["use_cuda"]:
                input_ids = input_ids.to("cuda")
                model.to("cuda")

            init_example_time = time.time()
            _ = model(input_ids)

            time_per_example.append(time.time() - init_example_time)

        # Compute statistics over individual wallclock times
        results["median"].append(np.median(time_per_example))
        results["mean"].append(np.mean(time_per_example))
        results["std"].append(np.std(time_per_example))
        results["min"].append(np.min(time_per_example))
        results["max"].append(np.max(time_per_example))
        results["pct_5"].append(np.percentile(time_per_example, 5))
        results["pct_95"].append(np.percentile(time_per_example, 95))
        logger.info(f"Sequence Length: {seq_len}, Time per example: {results['mean']}")

    logger.info(f"Total Runtime: {time.time() - init_time}")
    pickle.dump(results, open(f"log/{model_name}_{opts['device']}.out", "wb"))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as config_file:
        params = yaml.safe_load(config_file)

    if args.model == "all":
        for model in NAME2MODEL.keys():
            main(params, model)
    else:
        main(params, args.model)
