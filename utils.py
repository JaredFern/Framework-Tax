import logging
from pathlib import Path

import torch


def _get_logger(results_dir, device):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    log_fname = f"{results_dir}/{device}.log"
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(log_fname)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def _get_parameter_count(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


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
