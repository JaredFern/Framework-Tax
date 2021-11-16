import argparse
import logging
import os
from functools import partial

import pandas as pd
import torch
import transformers  # Requires sentencepiece
import yaml

from metrics import Benchmark
from utils import fill_metadata, prepare_model, setup_logger

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
        "google/reformer-enwik8",
    ],
    "squeeze_bert": [
        transformers.SqueezeBertModel,
        transformers.SqueezeBertTokenizer,
        "squeezebert/squeezebert-uncased",
    ],
}


def main(opts, model_name, device_name, results_dir):
    logger = logging.getLogger(device_name)
    logger.info(f"Loading {model_name} model.")
    results_file = f"{results_dir}/{device_name}.csv"
    dataframe = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    # Set Device and Get Appropriate Model Functions
    device = torch.device(opts["device"])
    model_fn, tokenizer_fn, checkpoint = NAME2MODEL[model_name]

    # Load Tokenizer to get Vocab Size
    tokenizer = tokenizer_fn.from_pretrained(checkpoint)

    for batch_size in opts["batch_size"]:
        for seq_len in opts["sequence_lengths"]:
            if model_name == "funnel_transformer" and seq_len < 8:
                continue
            metadata = {
                "model": model_name,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                **fill_metadata(opts),
            }

            id_constructor = partial(
                torch.randint,
                low=0,
                high=tokenizer.vocab_size,
                size=(batch_size, seq_len),
                device=device,
            )
            # Load Model and Set CUDA Device
            model = model_fn.from_pretrained(checkpoint, torchscript=opts["use_jit"])
            model = prepare_model(model, id_constructor(), opts)

            benchmarker = Benchmark(
                model,
                id_constructor,
                opts["num_threads"],
                opts["use_cuda"],
                device_idx=opts["device_idx"],
            )
            data = benchmarker.aggregate_metrics(opts["use_dquant"], opts["use_jit"], opts["iters"])
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
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--device_config", type=str, required=True)
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
