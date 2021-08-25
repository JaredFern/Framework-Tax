import argparse
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torchvision import datasets, models
from torchvision import transforms as T
from tqdm import tqdm

# TODO: Fix visualizations to handle expanded metrics

NAME2MODEL = {
    "resnet18": models.resnet18,
    "alexnet": models.alexnet,
    "squeezenet": models.squeezenet1_0,
    "vgg16": models.vgg16,
    "densenet": models.densenet161,
    "inception": models.inception_v3,
    "googlenet": models.googlenet,
    "shufflenet": models.shufflenet_v2_x1_0,
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0,
}


def _get_logger(results_dir, device):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    log_fname = f"{results_dir}/{device}.log"
    file_handler = logging.FileHandler(filename=log_fname)
    file_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(log_fname)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def main(opts, device, model_name, results_dir):
    init_time = time.time()
    logger = _get_logger(results_dir, device)

    # Load Model & Tokenizer
    results_fname = f"{results_dir}/{device}.csv"
    dataframe = (
        pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()
    )

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    cifar_data = datasets.CIFAR10("data/", False, transform, download=True)
    data_loader = torch.utils.data.DataLoader(
        cifar_data, batch_size=opts["batch_size"], shuffle=True
    )

    model = NAME2MODEL[model_name](pretrained=True)
    model.requires_grad_(opts["requires_grad"])
    model.eval()

    data = {
        "device": device,
        "model": model_name,
        "accelerator": opts["use_cuda"],
        "batch_size": opts["batch_size"],
    }

    time_per_example = []
    init_image, _ = next(iter(data_loader))
    if opts["use_cuda"]:
        init_image = init_image.to("cuda")
        model.to("cuda")

    # Dry run a single example
    _ = model(init_image)

    for idx, (input_image, _) in enumerate(data_loader):
        if opts["use_cuda"]:
            input_image = input_image.to("cuda")
            model.to("cuda")

        init_example_time = time.time()
        _ = model(input_image)
        time_per_example.append(time.time() - init_example_time)
        if idx >= opts["iters"]:
            break

    # Compute statistics over individual wallclock times
    data["median"] = np.median(time_per_example)
    data["mean"] = np.mean(time_per_example)
    data["std"] = np.std(time_per_example)
    data["min"] = np.min(time_per_example)
    data["max"] = np.max(time_per_example)
    data["5_pct"] = np.percentile(time_per_example, 5)
    data["95_pct"] = np.percentile(time_per_example, 95)
    dataframe = dataframe.append(data, ignore_index=True)

    logger.info(f"Time per example: {data['mean']}")

    logger.info(f"Total Runtime: {time.time() - init_time}")
    dataframe.to_csv(results_fname, index=False)
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

    if args.model == "all":
        for model in NAME2MODEL.keys():
            main(params, args.device, model, args.results_dir)
    else:
        main(params, args.device, args.model, args.results_dir)
