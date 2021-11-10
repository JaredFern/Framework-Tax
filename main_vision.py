import argparse
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import yaml
from timm import create_model
from torchvision import models

from metrics import Benchmark
from utils import prepare_model, setup_logger

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
    "mobilenet_v2": models.quantization.mobilenet_v2,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0,
}


def main(opts, device_name, model_name, results_dir):
    logger = logging.getLogger(device_name)
    results_file = f"{results_dir}/{device_name}.csv"
    dataframe = (
        pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
    )

    # Load model and inputs
    logger.info(f"Loading {model_name} model")
    device = torch.device(opts["device"])
    # Lazy loading of MobileNet as quantized bc it is the only one impled with QNNPack:
    if model_name == "mobilenet_v2":
        model = NAME2MODEL[model_name](pretrained=True, quantize=opts['use_dynamic_quant'])
    else:
        model = NAME2MODEL[model_name](pretrained=True)
    model = prepare_model(model, device, opts["requires_grad"], opts['use_dynamic_quant'])

    for batch_size in opts["batch_size"]:
        for img_size in opts["img_sizes"]:
            if model_name in "vit32":
                model = NAME2MODEL[model_name](pretrained=True, img_size=img_size)
                model = prepare_model(model, device, opts["requires_grad"])

            data = {
                "device": device_name,
                "model": model_name,
                "accelerator": opts["use_cuda"],
                "requires_grad": opts["requires_grad"],
                "use_torchscript": opts["use_jit"],
                "num_threads": opts["num_threads"],
                "batch_size": batch_size,
                "img_size": img_size,
            }

            img_constructor = partial(
                torch.rand, device=device, size=(batch_size, 3, img_size, img_size)
            )

            if opts["use_jit"]:
                model = torch.jit.trace(model, img_constructor())

            benchmarker = Benchmark(
                model,
                img_constructor,
                opts["num_threads"],
                opts["use_cuda"],
                device_idx=opts["device_idx"],
            )
            # Not sure why but dquant causes OOM when doing memory profiling
            if not opts["use_dynamic_quant"]:
                memory_usage = benchmarker.get_memory()
                data["avg_memory"] = np.mean(memory_usage)
                data["max_memory"] = np.max(memory_usage)
                data["latency"] = benchmarker.get_wallclock(opts["iters"])
            if not opts["use_jit"]:
                data["macs"] = benchmarker.get_flops_count()
                data["total_params"] = benchmarker.get_param_count(False)
                data["trainable_params"] = benchmarker.get_param_count(True)

            # Combine the run params with the observed metrics
            dataframe = dataframe.append(data, ignore_index=True)
            dataframe.to_csv(results_file, index=False)

    # Teardown model to (hopefully) avoid OOM on next run
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
    args = parser.parse_args()

    # Load config
    with open(args.config_file, "r") as config_file:
        params = yaml.safe_load(config_file)

    setup_logger(args.results_dir, args.device)
    main(params, args.device, args.model, args.results_dir)
