import importlib
import pickle
import time
import timeit

from memory_profiler import memory_usage
from torch.utils import benchmark

if importlib.util.find_spec("py3nvml"):
    from py3nvml import py3nvml as nvml


class JaxBenchmark(object):
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
        if self.config.use_jit:
            self.model = jax.jit(model)


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

    def aggregate_metrics(self, iters=10):
        data = {}
        self._prepare_model()
        memory_usage = self.get_memory()
        data["avg_memory"] = np.mean(memory_usage)
        data["max_memory"] = np.max(memory_usage)
        data["latency"] = self.get_wallclock(iters)
        return data
