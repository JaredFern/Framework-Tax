import argparse
import os

import pandas as pd
import torch
import yaml
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
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0,
}


def main(opts, device, model_name, metrics, results_dir):
    logger = _get_logger(results_dir, device)
    results_fname = f"{results_dir}/{device}.csv"
    dataframe = (
        pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()
    )

    # Load model and inputs
    model = NAME2MODEL[model_name](pretrained=True)
    model.requires_grad_(opts["requires_grad"])
    model.eval()

    # Load image transform and dataloader
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    data = datasets.CIFAR10("data/", False, transform, download=True)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=opts["batch_size"], shuffle=True
    )

    input_image, _ = next(iter(data_loader))
    if opts["use_cuda"]:
        input_image = input_image.to("cuda")
        model.to("cuda")

    # Dry run a single example
    _ = model(input_image)

    # Compute statistics over individual wallclock times
    results = gather_metrics(opts, model, input_image, metrics, logger)

    # Combine the run params with the observed metrics
    data = {
        "device": device,
        "model": model_name,
        "accelerator": opts["use_cuda"],
        "batch_size": opts["batch_size"],
    }
    data.update(results)

    # Log results to file, teardown models
    dataframe = dataframe.append(data, ignore_index=True)
    dataframe.to_csv(results_fname, index=False)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--results_dir", type=str)  # experiments/MMDDYY_name/
    parser.add_argument("--device", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--metrics", choices=["wallclock", "flops", "memory", "params"], nargs="+"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config_file, "r") as config_file:
        params = yaml.safe_load(config_file)

    if args.model == "all":
        for model in NAME2MODEL.keys():
            main(params, args.device, model, args.metrics, args.results_dir)
    else:
        main(params, args.device, args.model, args.metrics, args.results_dir)
