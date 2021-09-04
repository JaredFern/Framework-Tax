import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import yaml
from timm import create_model
from torchvision import datasets, models
from torchvision import transforms as T

from metrics import Benchmark
from utils import _get_logger

NAME2MODEL = {
    "vit32": partial(create_model, "vit_base_patch32_224"),
    "efficientnet": partial(create_model, "efficientnetv2_m"),
    "efficientnet_lite": partial(create_model, "efficientnet_lite1"),
    "gernet": partial(create_model, "gernet_m"),
    "resnet18": models.resnet18,
    "alexnet": models.alexnet,
    "squeezenet": models.squeezenet1_0,
    "vgg16": models.vgg16,
    "densenet": models.densenet161,
    "inception": models.inception_v3,
    "googlenet": models.googlenet,
    "shufflenet": models.shufflenet_v2_x1_0,
    "mobilenet_v2": models.mobilenet_v2,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0,
}


def main(opts, device_name, model_name, metrics, results_dir, logger):
    results_fname = f"{results_dir}/{device_name}.csv"
    dataframe = (
        pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()
    )

    # Load model and inputs
    logger.info(f"Loading {model_name} model")
    model = NAME2MODEL[model_name](pretrained=True)
    model.requires_grad_(opts["requires_grad"])
    if not opts["requires_grad"]:
        model.eval()

    # Set CUDA Devices
    if opts["use_cuda"]:
        model.to("cuda")
        device = torch.device(f"{opts['device']}:{opts['device_idx']}")
    else:
        device = torch.device("cpu")

    img_sizes = [224, 384, 448, 512]
    for batch_size in opts["batch_size"]:
        for img_size in img_sizes:
            if model_name in "vit32":
                model = NAME2MODEL[model_name](pretrained=True, img_size=img_size)
                model.requires_grad_(opts["requires_grad"])
                if not opts["requires_grad"]:
                    model.eval()
                if opts["use_cuda"]:
                    model.to("cuda")

            data = {
                "device": device_name,
                "model": model_name,
                "accelerator": opts["use_cuda"],
                "requires_grad": opts["requires_grad"],
                "use_torchscript": opts["use_jit"],
                "batch_size": batch_size,
                "img_size": img_size,
            }

            img_constructor = partial(
                torch.rand, device=device, size=(batch_size, 3, img_size, img_size)
            )

            benchmarker = Benchmark(
                model,
                img_constructor,
                opts["num_threads"],
                opts["use_cuda"],
                device_idx=opts["device_idx"],
            )
            memory_usage = benchmarker.get_memory()
            data["avg_memory"] = np.mean(memory_usage)
            data["max_memory"] = np.max(memory_usage)
            data["macs"] = benchmarker.get_flops_count()
            data["total_params"] = benchmarker.get_param_count(False)
            data["trainable_params"] = benchmarker.get_param_count(True)
            data["mean"] = benchmarker.get_wallclock(opts["use_jit"], opts["iters"])

            # Combine the run params with the observed metrics
            dataframe = dataframe.append(data, ignore_index=True)
            dataframe.to_csv(results_fname, index=False)

    # Model teardown
    del model
    if opts["use_cuda"]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--metrics",
        choices=["wallclock", "flops", "memory", "params"],
        nargs="+",
        default=["wallclock", "flops", "memory", "params"],
    )
    args = parser.parse_args()
    logger = _get_logger(args.results_dir, args.device)

    # Load config
    with open(args.config_file, "r") as config_file:
        params = yaml.safe_load(config_file)

    main(params, args.device, args.model, args.metrics, args.results_dir, logger)
