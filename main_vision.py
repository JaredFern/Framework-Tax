import argparse
import os
from functools import partial

import pandas as pd
import torch
import yaml
from timm import create_model
from torchvision import datasets, models
from torchvision import transforms as T

from metrics import gather_metrics
from utils import _get_logger

NAME2MODEL = {
    "vit32": create_model("vit_base_patch32_224", pretrained=True),
    "efficientnet": create_model("efficientnetv2_m", pretrained=True),
    "efficientnet_lite": create_model("efficientnet_lite1", pretrained=True),
    "gernet": create_model("gernet_m", pretrained=True),
    "resnet18": models.resnet18(pretrained=True),
    "alexnet": models.alexnet(pretrained=True),
    "squeezenet": models.squeezenet1_0(pretrained=True),
    "vgg16": models.vgg16(pretrained=True),
    "densenet": models.densenet161(pretrained=True),
    "inception": models.inception_v3(pretrained=True),
    "googlenet": models.googlenet(pretrained=True),
    "shufflenet": models.shufflenet_v2_x1_0(pretrained=True),
    "mobilenet_v2": models.mobilenet_v2(pretrained=True),
    "resnext50_32x4d": models.resnext50_32x4d(pretrained=True),
    "wide_resnet50_2": models.wide_resnet50_2(pretrained=True),
    "mnasnet": models.mnasnet1_0(pretrained=True),
}


def main(opts, device, model_name, metrics, results_dir, logger):
    results_fname = f"{results_dir}/{device}.csv"
    dataframe = (
        pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()
    )

    # Load model and inputs
    logger.info(f"Loading {model_name} model")
    model = NAME2MODEL[model_name]
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
                ckpt = "vit_base_patch32_224"
                model = create_model(ckpt, pretrained=True, img_size=img_size)
                model.requires_grad_(opts["requires_grad"])
                if not opts["requires_grad"]:
                    model.eval()
                if opts["use_cuda"]:
                    model.to("cuda")

            data = {
                "device": device,
                "model": model_name,
                "accelerator": opts["use_cuda"],
                "requires_grad": opts["requires_grad"],
                "batch_size": batch_size,
                "img_size": img_size,
            }

            img_constructor = partial(
                torch.rand, device=device, size=(batch_size, 3, img_size, img_size)
            )
            results = gather_metrics(opts, model, img_constructor, metrics, logger)

            # Combine the run params with the observed metrics
            data.update(results)
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

    if args.model == "all":
        for model in NAME2MODEL.keys():
            main(params, args.device, model, args.metrics, args.results_dir, logger)
    else:
        main(params, args.device, args.model, args.metrics, args.results_dir, logger)
