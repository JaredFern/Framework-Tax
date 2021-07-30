import argparse
import logging
import time

import numpy as np
import torch
import transformers  # Requires sentencepiece
import yaml

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
        "google/reformer-crime-and-punishment",  # Wiki8 ckpt not avail.t pul
    ],
    "squeeze_bert": [
        transformers.SqueezeBertModel,
        transformers.SqueezeBertTokenizer,
        "squeezebert/squeezebert-uncased",
    ],
}


def main(opts, model_name):
    device = "gpu" if opts["use_cuda"] else "cpu"
    # Setup Logger
    file_handler = logging.FileHandler(filename=f"log/{model_name}_{device}.log")
    file_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    init_time = time.time()
    model_fn, tokenizer, checkpoint = NAME2MODEL[model_name]
    logger.info(f"Loading {model_name} model from {checkpoint}")
    tokenizer = tokenizer.from_pretrained(checkpoint)
    model = model_fn.from_pretrained(checkpoint)
    model.requires_grad_(opts["requires_grad"])
    model.eval()

    seq_lengths = [
        2 ** seqlen for seqlen in range(10)
    ]  # Eval on sequence lengths up from 1 to 512
    wallclock_times = []
    for seq_len in seq_lengths:
        time_per_example = []
        for _ in range(opts["iters"]):

            if opts["randomized_text"]:
                input_ids = torch.randint(tokenizer.vocab_size, (seq_len))
            else:
                tokens = seq_len * ["a"]
                input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
            input_ids = input_ids.unsqueeze(0)
            if opts["use_cuda"]:
                input_ids = input_ids.to("cuda")
                model.to("cuda")

            init_example_time = time.time()
            _ = model(input_ids)
            time_per_example.append(time.time() - init_example_time)
        average_time = np.mean(time_per_example)
        wallclock_times.append(average_time)
        logger.info(f"Sequence Length: {seq_len}, Time per example: {average_time}")
    logger.info(f"Averaged Times: {wallclock_times}")
    logger.info(f"Total Runtime: {time.time() - init_time}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str)
    opts = parser.parse_args()

    # Load config
    with open(opts.config, "r") as config_file:
        params = yaml.safe_load(config_file)

    if opts.model == "all":
        for model in NAME2MODEL.keys():
            main(params, model)
    else:
        main(params, opts.model)
