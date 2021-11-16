import argparse
import logging
import os
from functools import partial

import pandas as pd
import torch
import yaml
from timm import create_model
from torchvision import models

from metrics import Benchmark
from utils import fill_metadata, prepare_model, setup_logger

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


# TODO: Cleanup model loading for ViT and Mobilenet
def main(opts, model_name, device_name, results_dir):
    logger = logging.getLogger(device_name)
    results_file = f"{results_dir}/{device_name}.csv"
    dataframe = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    # Load model and inputs
    logger.info(f"Loading {model_name} model")
    device = torch.device(opts["device"])

    for batch_size in opts["batch_size"]:
        for img_size in opts["img_sizes"]:
            metadata = {
                "model": model_name,
                "batch_size": batch_size,
                "img_size": img_size,
                **fill_metadata(opts),
            }

            img_constructor = partial(
                torch.rand, device=device, size=(batch_size, 3, img_size, img_size)
            )

            if model_name in "vit32":
                model = NAME2MODEL[model_name](pretrained=True, img_size=img_size)
            elif model_name == "mobilenet_v2":
                model = NAME2MODEL[model_name](pretrained=True, quantize=opts["use_dquant"])
            else:
                model = NAME2MODEL[model_name](pretrained=True)
            model = prepare_model(model, img_constructor(), opts)

            benchmarker = Benchmark(
                model,
                img_constructor,
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
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--device_config", type=str)
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
