import importlib
import pickle
import time
import timeit

import numpy as np
from memory_profiler import memory_usage
from optimum.bettertransformer import BetterTransformer
from thop import profile as thop_profile

import torch
from torch.cuda.nvtx import range_push, range_pop
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils import benchmark

if importlib.util.find_spec("py3nvml"):
    from py3nvml import py3nvml as nvml


class PyTorchBenchmark(object):
    def __init__(self, model, input_constructor, config, logdir=""):
        self.model = model
        self.logdir = logdir
        self.input_constructor = input_constructor
        self.config = config

        if self.config.use_cuda:
            self.profile_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        else:
            self.profile_activities = [ProfilerActivity.CPU]

    def _prepare_model(self):
        precisions = {torch.float}
        self.model = self.model.to(torch.device(self.config.device))
        if not self.config.requires_grad:
            self.model.requires_grad_(False)
            self.model.eval()
        if self.config.use_ipex:
            self.model = ipex.optimize(self.model)
        if self.config.use_fp16:
            precisions = {torch.float, torch.half}
            self.model = self.model.half()

    def _optimize_model(self):
        if self.config.use_dquant:
            torch.backends.quantized.engine = "qnnpack"
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        if self.config.use_better_tf:
            self.model = BetterTransformer.transform(self.model)
        if self.config.use_jit:
            input_example = self.input_constructor()
            jit_model = torch.jit.trace(self.model, input_example)
            del self.model
            self.model = jit_model
        if self.config.use_tensorrt:
            input_example = self.input_constructor()
            jit_model = torch.jit.trace(self.model, (input_example,))
            trt_model = torch_tensorrt.compile(
                jit_model, inputs=[input_example], enabled_precisions=precisions
            )
            del self.model, jit_model
            torch.cuda.empty_cache()
            self.model = trt_model
        if self.config.use_ptcompile:
            self.model = torch.compile(self.model)
        if self.config.use_cuda_graphs:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(stream):
                for warmup_step in range(10):
                    _ = self.model(self.input_constructor())
            torch.cuda.current_stream().wait_stream(stream)


            self.cuda_graph = torch.cuda.CUDAGraph()
            self.graph_input = self.input_constructor()
            with torch.cuda.graph(self.cuda_graph):
                _ = self.model(self.graph_input)


    def get_wallclock(self, iters=10):
        """Compute mean runtime for model execution averaged over iter runs

        Args:
            iters (int, optional): Number of model excecutions. Defaults to 10.

        Returns:
            wallclock_mean: Average runtime
        """
        if self.config.use_cuda_graphs:
            timer = benchmark.Timer(
                stmt="""
                    graph.replay()
                    """,
                setup="""
                    graph_input.copy_(input_constructor())
                    """,
                num_threads=self.config.num_threads,
                globals={
                    "graph": self.cuda_graph,
                    "graph_input": self.graph_input,
                    "input_constructor": self.input_constructor,
                },
            )
        else:
            timer = benchmark.Timer(
                stmt="""
                model(input_tensor)
                """,
                setup="""
                input_tensor=input_constructor(); model(input_tensor)
                """,
                num_threads=self.config.num_threads,
                globals={
                    "model": self.model,
                    "input_constructor": self.input_constructor,
                },
            )

        # with torch.autograd.profiler.emit_nvtx():
        wallclock_mean = timer.timeit(iters).mean
        return wallclock_mean

    def get_profile(self, iters=10, warmup=10, logdir="."):
        input_tensor = self.input_constructor()
        for iter in range(iters + warmup):
            if iter == warmup:
                torch.cuda.cudart().cudaProfilerStart()
            if self.config.use_cuda_graphs:
                self.graph_input.copy_(self.input_constructor())
                range_push("forward")
                self.cuda_graph.replay()
                torch.cuda.synchronize()
                range_pop()
            else:
               input_tensor = self.input_constructor()
               range_push("forward")
               _ = self.model(input_tensor)
               torch.cuda.synchronize()
               range_pop()




    def get_memory(self):
        if "cuda" in self.config.device:
            nvml.nvmlInit()
            _ = self.model(self.input_constructor())
            handle = nvml.nvmlDeviceGetHandleByIndex(self.config.device_idx)
            meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_bytes = meminfo.used
        else:
            memory_bytes = memory_usage((self.model, self.input_constructor().unsqueeze(0)))
        return memory_bytes

    def get_flops_count(self):
        macs, _, macs_by_op, op_count = thop_profile(
            self.model, (self.input_constructor(),), verbose=False, ret_layer_info=False
        )
        return macs, macs_by_op, op_count

    def get_param_count(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())

    def aggregate_metrics(self, iters=10):
        data = {}
        # Model Architecture Metrics
        self._prepare_model()
        macs, macs_by_op, op_count = self.get_flops_count()
        data["macs"] = macs
        for op, macs in macs_by_op.items():
            data[f"{op}_macs"] = macs

        data["total_nn_calls"] = 0
        for op, count in op_count.items():
            data[f"{op}_calls"] = count
            data["total_nn_calls"] += count
        data["total_params"] = self.get_param_count(False)
        data["trainable_params"] = self.get_param_count(True)

        # Model Execution Metrics
        self._optimize_model()
        data["latency"] = self.get_wallclock(iters)
        _ = self.get_profile(iters)
        if not self.config.use_dquant:
            memory_usage = self.get_memory()
            data["avg_memory"] = np.mean(memory_usage)
            data["max_memory"] = np.max(memory_usage)
        return data
