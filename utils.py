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
