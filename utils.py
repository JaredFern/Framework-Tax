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


def fill_metadata(opts):
    return {
        "device": opts["device"],
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

    if opts["use_jit"]:
        jit_model = torch.jit.trace(model, input_example)

        # Replace model with jit_model
        del model
        model = jit_model

    if not opts["requires_grad"]:
        model.requires_grad_(False)
        model.eval()
    if opts["use_dquant"]:
        torch.backends.quantized.engine = "qnnpack"
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    return model
