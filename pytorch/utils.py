import datetime
import logging
import os

import intel_extension_for_pytorch as ipex
import torch


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


def fill_metadata(opts):
    return {
        "accelerator": opts["use_cuda"],
        "requires_grad": opts["requires_grad"],
        "use_torchscript": opts["use_jit"],
        "use_dquant": opts["use_dquant"],
        "dropout": opts["dropout"],
        "act_fn": opts["act_fn"],
    }


def prepare_model(model, input_example, opts):
    model = model.to(torch.device(opts["device"]))
    input_example = input_example.to(opts["device"])

    if not opts["requires_grad"]:
        model.requires_grad_(False)
        model.eval()
    if opts["use_intel_exts"]:
        model = ipex.optimize(model)
    if opts["use_jit"]:
        model = torch.jit.trace(model, input_example)
    if opts["use_dquant"]:
        torch.backends.quantized.engine = "qnnpack"
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    return model
