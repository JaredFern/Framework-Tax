import argparse
import numpy as np
import os

import onnx 
import torch
from config import NAME2MODEL_VISION

def export_onnx_vision_models(opts):
    pt_model = NAME2MODEL_VISION[opts.model_name]().to('cuda')
    inputs = torch.randn((1,3,224,224)).to('cuda')
    
    if opts.use_fp16:
        pt_model = pt_model.half()
        inputs = inputs.half()
    
    precision = "fp16" if opts.use_fp16 else "fp32"
    model_path = os.path.join(
        opts.model_dir, f"{opts.model_name}-{precision}.onnx",)

    torch.onnx.export(
        pt_model, inputs, model_path, export_params=True, opset_version=opts.opset_version, do_constant_folding=True,
        input_names=['inputs'], output_names=['outputs'],
        dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}})
                      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--opset_version", type=int, default=10)
    parser.add_argument("--use_fp16", action="store_true")

    # Load from Config Files
    args = parser.parse_args()

    export_onnx_vision_models(args)