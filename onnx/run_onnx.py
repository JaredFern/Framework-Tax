import argparse
import datetime
import logging
import os
from pathlib import Path

import pandas as pd
import onnxruntime as ort

import torch
from torch.utils import benchmark
from thop import profile as thop_profile


def create_and_bind_rgb(io_binding, bs, device, device_idx, dtype):
    pixels_cpu = torch.randn((bs,3,224,224), dtype=dtype).cpu().numpy()
    out = torch.empty((bs,1000), dtype=dtype).numpy()
    
    pixels_ortvalue = ort.OrtValue.ortvalue_from_numpy(pixels_cpu, device, 0)
    out_ortvalue = ort.OrtValue.ortvalue_from_numpy(out, device, device_idx)
    io_binding.bind_ortvalue_input('inputs', pixels_ortvalue)
    io_binding.bind_ortvalue_output('outputs', out_ortvalue)
    return pixels_ortvalue, out_ortvalue 

def create_and_bind_ids(io_binding, bs, device):
    ids = torch.randint(0, 1000, (1, 128)).numpy()
    ids_ortvalue = ort.OrtValue.ortvalue_from_numpy(ids, device, 0)
    io_binding.bind_ortvalue_input('ids', ids_ortvalue)


def eval_latency(
    model_name, providers, iters=100, batch_sizes=[1], 
    device="cuda", device_idx=0, dtype=torch.float32, 
    model_dir="onnx_models/"
):
    latency = []
    results = pd.DataFrame()
    for bs in batch_sizes:
        print(f"Evaluating {model_name} with batch size {bs}")        
        data = {"model": model_name, "batch_size": bs}
        
        precision = "fp16" if dtype == torch.float16 else "fp32"
        onnx_fname = f"{model_name}-{precision}.onnx"
        onnx_fpath = os.path.join(model_dir, onnx_fname)

        session = ort.InferenceSession(onnx_fpath, providers=providers)
        io_binding = session.io_binding()
        create_and_bind_rgb(io_binding, bs, device, device_idx, dtype)

        # Warmup Run
        session.run_with_iobinding(io_binding)
        
        # Measure ONNX latency
        timer = benchmark.Timer(
            stmt="session.run_with_iobinding(io_binding)",
            globals={"session": session, "io_binding": io_binding}
        )
        batch_latency = timer.timeit(iters).mean 
        print(f"Mean latency for {iters} runs: {batch_latency}")
        data['latency'] = batch_latency
        results = results.append(data, ignore_index=True)
    print(results)
    return results
        
        
def main(args):
    batch_sizes = [1,2,3,4,6,8,16,32,64]
    dtype = torch.float16 if args.use_fp16 else torch.float32 
    
    results_dir = os.path.join(
        args.results_dir, 
        f"{datetime.datetime.now().strftime('%Y_%m%d')}_{args.exp_name}"
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if args.device == "cuda":
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': args.device_idx,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
    else:
        providers = ['CPUExecutionProvider']

    results_fpath = os.path.join(args.results_dir, f"{args.exp_name}.csv")
    results = eval_latency(
            args.model_name, providers, args.iters, batch_sizes, args.device, args.device_idx, dtype)
    print(f"Logging results to {results_fpath}")
    results.to_csv(results_fpath, index=False)       
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_dir", type=str)                
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--platform", type=str, default="")
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--use_fp16", action="store_true")

    # Load from Config Files
    args = parser.parse_args()

    main(args)