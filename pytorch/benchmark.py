import importlib
import pickle
import time
import timeit

import intel_extension_for_pytorch as ipex
import numpy as np
import torch
import torch_tensorrt
from memory_profiler import memory_usage
from thop import profile as thop_profile
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

    def get_wallclock(self, iters=10):
        """Compute mean runtime for model execution averaged over iter runs

        Args:
            iters (int, optional): Number of model excecutions. Defaults to 10.

        Returns:
            wallclock_mean: Average runtime
        """
        # Warmup Run Just in Case
        out = self.model(self.input_constructor())
        del out

        if self.config.use_cuda:
            exec_stmt = "model(input_tensor); torch.cuda.synchronize();"
        else:
            exec_stmt = "model(input_tensor);"

        timer = benchmark.Timer(
            stmt="exec_stmt",
            setup="input_tensor=input_constructor()",
            num_threads=self.config.num_threads,
            globals={
                "model": self.model,
                "input_constructor": self.input_constructor,
                "exec_stmt": exec_stmt,
            },
        )
        wallclock_mean = timer.timeit(iters).mean
        return wallclock_mean

    def get_profile(self, iters=10, logdir="."):
        input_tensor = self.input_constructor()
        with profile(
            activities=self.profile_activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
            schedule=torch.profiler.schedule(
                skip_first=4, wait=0, warmup=2, active=iters, repeat=0
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_stack=True,
        ) as prof:
            with record_function("model_inference"):
                for _ in range(iters + 6):
                    self.model(input_tensor)
                    prof.step()
        pickle.dump(
            prof.key_averages(group_by_input_shape=True),
            open(f"{logdir}/groupbyshapes_profiler.p", "wb"),
        )
        pickle.dump(
            prof.key_averages(group_by_stack_n=True),
            open(f"{logdir}/groupbystack_profiler.p", "wb"),
        )

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
        if not self.config.use_dquant:
            memory_usage = self.get_memory()
            data["avg_memory"] = np.mean(memory_usage)
            data["max_memory"] = np.max(memory_usage)
        return data
