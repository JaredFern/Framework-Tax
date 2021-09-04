import logging
from pathlib import Path

import torch


def setup_logger(results_dir, device):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(device)
    log_fname = f"{results_dir}/{device}.log"

    # Add File
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def prepare_model(model, device, requires_grad):
    model = model.to(device)
    if not requires_grad:
        model.requires_grad_(requires_grad)
        model.eval()
    return model
