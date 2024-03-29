import argparse
import datetime
import logging
import os
import yaml
from functools import partial
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from optimum.onnxruntime import AutoOptimizationConfig, ORTOptimizer, ORTModelForCausalLM, ORTModelForSequenceClassification

from benchmark import PyTorchBenchmark
from config import NAME2MODEL_LANGUAGE, NAME2MODEL_SPEECH, NAME2MODEL_VISION
from utils import setup_logger


def _build_language_model(model_name, device, use_jit=False, use_onnxrt=False):
    model_fn, _, checkpoint = NAME2MODEL_LANGUAGE[model_name]
    if use_jit:
        model = model_fn.from_pretrained(checkpoint, torchscript=use_jit).to(device)
    if use_onnxrt:
        if model_name == "gpt2":
            model = ORTModelForCausalLM.from_pretrained("gpt2_onnx")
        else:
            model = ORTModelForSequenceClassification.from_pretrained("./tmp", file_name="model_optimized.onnx")

    else:
        model = model_fn.from_pretrained(checkpoint, torchscript=use_jit).to(device)

    return model


def _build_vision_model(
    model_name, device, img_size=224, use_dquant=False, pretrained=False
):
    if model_name in "vit32":
        model = NAME2MODEL_VISION[model_name](
            pretrained=pretrained, img_size=img_size
        ).to(device)
    elif model_name == "mobilenet_v2":
        model = NAME2MODEL_VISION[model_name](
            pretrained=pretrained, quantize=use_dquant
        ).to(device)
    else:
        model = NAME2MODEL_VISION[model_name](pretrained=pretrained).to(device)
    return model


def _build_speech_model(model_name, device, use_jit):
    model_fn, _, model_name = NAME2MODEL_SPEECH[model_name]
    model = model_fn.from_pretrained(model_name, torchscript=use_jit).to(device)
    return model


def _build_language_input(
    model_name, batch_size, seq_len, std_len=0, return_attention_mask=False, device=None):
    _, tokenizer_fn, model_name = NAME2MODEL_LANGUAGE[model_name]
    tokenizer = tokenizer_fn.from_pretrained(model_name)

    def generate_padded_inputs(batch_size, seq_len, std_len, device):
        sampled_seqlens = torch.round(
            torch.normal(mean=seq_len, std=std_len, size=(batch_size,))
        ).to(torch.int32)
        sampled_seqlens[sampled_seqlens < 0] = 0
        input_ids = [
           torch.randint(
               low=0, high=tokenizer.vocab_size, size=tuple([seqlen.item()]), device=device)
            for seqlen in sampled_seqlens
        ]

        attn_mask = [
           torch.ones(size=tuple([seqlen.item()]), device=device)
            for seqlen in sampled_seqlens
        ]
        attn_mask = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True).to(device).to(torch.int64)
        padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(device).to(torch.int64)

        return {"input_ids": padded_ids, "attention_mask": attn_mask, "token_type_ids": attn_mask}

    # if True:
    #     return partial(tokenizer, ["<|endoftext|>"] * batch_size, return_tensors="pt")
    if return_attention_mask:
        id_constructor = partial(generate_padded_inputs, batch_size, seq_len, std_len, device)
    else:
        id_constructor = partial(
            torch.randint,
            low=0,
            high=tokenizer.vocab_size,
            size=(batch_size, seq_len),
            device=device,
        )
    return id_constructor


def _build_speech_input(
    model_name,
    batch_size,
    input_size,
    sampling_rate=16e3,
    device=None,
    dtype=torch.float32,
):
    # Assume 2 second utterances sampled at 16kHz
    return partial(
        torch.rand,
        device=device,
        dtype=dtype,
        size=(batch_size, int(input_size * sampling_rate)),
    )


def _build_vision_input(
    batch_size,
    img_size,
    device=None,
    dtype=torch.float32,
    memory_format=torch.channels_last,
):
    return partial(
        torch.empty,
        device=device,
        dtype=dtype,
        size=(batch_size, 3, img_size, img_size),
        memory_format=memory_format,
    )


def build_model(opts, batch_size, input_size, dtype, device):
    if opts.model_name in NAME2MODEL_LANGUAGE:
        input_constructor = _build_language_input(
            opts.model_name, batch_size, input_size,
            return_attention_mask=not opts.use_jit,
            device=device
        )
        model = _build_language_model(opts.model_name, opts.device, opts.use_jit, opts.use_onnxrt)
    elif opts.model_name in NAME2MODEL_SPEECH:
        input_constructor = _build_speech_input(
            opts.model_name, batch_size, input_size, device=device, dtype=dtype
        )
        model = _build_speech_model(opts.model_name, opts.device, opts.use_jit)
    elif opts.model_name in NAME2MODEL_VISION:
        if opts.use_channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        input_constructor = _build_vision_input(
            batch_size, input_size, device, dtype, memory_format
        )
        model = _build_vision_model(opts.model_name, device, input_size)
    else:
        raise ValueError(f"Model {opts.model_name} not supported.")
    return model, input_constructor


def main(opts):
    device = torch.device(opts.device)

    if opts.use_fp16:
        dtype = torch.float16
    elif opts.use_bfloat16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Setup Logging
    results_dir = os.path.join(
        opts.results_dir,
        f"{datetime.datetime.now().strftime('%Y_%m%d')}_{opts.exp_name}",
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(results_dir, opts.platform)
    results_file = f"{results_dir}/{opts.platform}.csv"
    dataframe = (
        pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()
    )

    # Load model and inputs
    logger.info(f"Loading {opts.model_name} model")

    for batch_size in opts.batch_size:
        logger.info(f"Run Parameters: {opts}")
        for input_size in opts.input_size:
            logger.info(
                f"Benchmarking {opts.model_name} -- Batch: {batch_size}; Input: {input_size}"
            )
            metadata = {}
            metadata["model"] = opts.model_name
            metadata["batch_size"] = batch_size
            metadata["input_size"] = input_size
            metadata['dtype'] = dtype
            metadata["use_jit"] = opts.use_jit
            metadata["use_onnxrt"] = opts.use_onnxrt

            model, input_constructor = build_model(
                opts, batch_size, input_size, dtype, device
            )
            benchmarker = PyTorchBenchmark(model, input_constructor, opts)
            data = benchmarker.aggregate_metrics()
            results = {**data, **metadata}

            # Combine the run params with the observed metrics
            dataframe = pd.concat([dataframe, pd.DataFrame(results, index=[0])], ignore_index=True)
            dataframe.to_csv(results_file, index=False)

    # Teardown model to (hopefully) avoid OOM on next run
    del model
    if opts.use_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--device_config", type=str)
    args, remain_args = parser.parse_known_args()
    with open(args.model_config, "r") as model_config:
        parser.set_defaults(**yaml.safe_load(model_config))

    with open(args.device_config, "r") as device_config:
        parser.set_defaults(**yaml.safe_load(device_config))

    # Base Configs
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--platform", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)

    # Optimization Params
    parser.add_argument("--requires_grad", action="store_true")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--use_cuda_graphs", action="store_true")
    parser.add_argument("--use_channels_last", action="store_true")
    parser.add_argument(
        "--use_fp16", action="store_true"
    )
    parser.add_argument("--use_bfloat16", action="store_true")

    parser.add_argument("--use_jit", action="store_true")
    parser.add_argument("--use_onnxrt", action="store_true")
    parser.add_argument("--use_tensorrt", action="store_true")
    parser.add_argument("--use_better_tf", action="store_true")
    parser.add_argument("--use_ipex", action="store_true")
    parser.add_argument("--use_dquant", action="store_true")
    parser.add_argument("--use_ptcompile", action="store_true")
    parser.add_argument("--run_dlprof", action="store_true")
    # Load from Config Files
    parser.add_argument("--results_dir", type=str, default="experiments/")
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()

    main(args)
