import argparse
import os

import pandas as pd
import torch
import yaml
from functools import partial
from torchvision import datasets, models
from torchvision import transforms as T

from metrics import gather_metrics
from utils import _get_logger

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
    # "mobilenet_v3_large": models.mobilenet_v3_large,
    # "mobilenet_v3_small": models.mobilenet_v3_small,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0,
}


def main(opts, device, model_name, metrics, results_dir, logger):
    results_fname = f"{results_dir}/{device}.csv"
    dataframe = (
        pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()
    )

    # Load model and inputs
    logger.info(f"Loading {model_name} model")
    model = NAME2MODEL[model_name](pretrained=True)
    model.requires_grad_(opts["requires_grad"])
    if not opts['requires_grad']:
        model.eval()

    # Set CUDA Devices
    if opts["use_cuda"]:
        model.to("cuda")
        device = torch.device(f"{opts['device']}:{opts['device_idx']}")
    else:
        device = torch.device("cpu")


    img_sizes = [224, 384]
    for batch_size in opts['batch_size']:
        for img_size in img_sizes:
            data = {
                "device": device,
                "model": model_name,
                "accelerator": opts["use_cuda"],
                "requires_grad": opts['requires_grad'],
                "batch_size": batch_size,
                "img_size": img_size
            }

            img_constructor = partial(torch.rand, device=device,
                                      size=(batch_size, 3, img_size, img_size))
            results = gather_metrics(opts, model, img_constructor, metrics, logger)

            # Combine the run params with the observed metrics
            data.update(results)
            dataframe = dataframe.append(data, ignore_index=True)
    del model
    dataframe.to_csv(results_fname, index=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--metrics", choices=["wallclock", "flops", "memory", "params"], nargs="+",
        default=["wallclock", "flops", "memory", "params"]
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
