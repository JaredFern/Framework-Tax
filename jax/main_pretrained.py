import argparse
import logging
import os
from functools import partial
from itertools import product

import pandas as pd
import yaml


# TODO: No
def main(opts, model_name, device_name, results_dir):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Base Configs
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--use_jit", type=bool, default=False)
    parser.add_argument("--use_dquant", type=bool, default=False)
    # Input Configs
    parser.add_argument("--batch_size", type=list, default=[1])
    parser.add_argument("--input_size", type=list, default=[128])
    # Load from Config Files
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
    main(all_params, args.model, args.device, args.results_dir)
