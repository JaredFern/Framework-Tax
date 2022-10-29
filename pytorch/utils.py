import datetime
import logging
import os


def setup_logger(results_dir, platform_name):
    logger = logging.getLogger(platform_name)
    log_fname = f"{results_dir}/{platform_name}.log"

    # Add File
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
