import importlib
import pickle
import time
import timeit

import numpy as np
import torch
from memory_profiler import memory_usage
from thop import profile as thop_profile
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils import benchmark

if importlib.util.find_spec("py3nvml"):
    from py3nvml import py3nvml as nvml


class PyTorchBenchmark(object):
    def __init__(
        self, model, input_constructor, num_threads=1, use_cuda=False, device_idx=0, logdir=""
    ):
        self.model = model
        self.logdir = logdir
        self.input_constructor = input_constructor
        self.num_threads = num_threads
        self.use_cuda = use_cuda
        self.device_idx = device_idx

        if use_cuda:
            self.profile_activities = [ProfilerActivity.CPU, ProfilerActivity.GPU]
        else:
            self.profile_activities = [ProfilerActivity.CPU]

    def get_wallclock(self, iters=10):
        timer = benchmark.Timer(
            stmt="model(input_tensor)",
            setup="input_tensor=input_constructor()",
            num_threads=self.num_threads,
            globals={"model": self.model, "input_constructor": self.input_constructor},
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
        if self.use_cuda:
            nvml.nvmlInit()
            _ = self.model(self.input_constructor())
            handle = nvml.nvmlDeviceGetHandleByIndex(self.device_idx)
            meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_bytes = meminfo.used
        else:
            memory_bytes = memory_usage((self.model, self.input_constructor().unsqueeze(0)))
        return memory_bytes

    def get_flops_count(self):
        macs, _, macs_by_op = thop_profile(
            self.model, (self.input_constructor(),), verbose=False, ret_layer_info=False
        )
        return macs, macs_by_op

    def get_param_count(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())

    def aggregate_metrics(self, use_dquant, use_jit, iters=100):
        data = {}
        data["latency"] = self.get_wallclock()
        if not use_dquant:
            memory_usage = self.get_memory()
            data["avg_memory"] = np.mean(memory_usage)
            data["max_memory"] = np.max(memory_usage)
        if not use_jit:
            macs, macs_by_op = self.get_flops_count()
            data["macs"] = macs
            for op, macs in macs_by_op.items():
                data[f"{op}_macs"] = macs
            data["total_params"] = self.get_param_count(False)
            data["trainable_params"] = self.get_param_count(True)
        return data
