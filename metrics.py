import time
import timeit

import numpy as np
import py3nvml.py3nvml as nvml
from memory_profiler import memory_usage
from thop import profile as thop_profile
from torch.profiler import ProfilerActivity, profile, record_function

from utils import _get_parameter_count


def gather_metrics(opts, model, input_tensor, metrics, logger, iters=100):
    data = {}
    init_time = time.time()

    # TODO: Add torch operation profiling, analysis {Operation Type x Device}
    # TODO: Get Checkpoint Size
    # TODO: Add controls for number of threads (swap to torch utils benchmark)

    if "memory" in metrics:
        if opts["use_cuda"]:
            nvml.nvmlInit()
            _ = model(input_tensor)
            handle = nvml.nvmlDeviceGetHandleByIndex(opts["device_idx"])
            meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_bytes = meminfo.used
        else:
            # NOTE: memory_usage grabs RAM snapshots from the process every 0.2s
            memory_bytes = memory_usage((model, input_tensor.unsqueeze(0)))
        data["max_memory"] = np.max(memory_bytes)
        data["avg_memory"] = np.mean(memory_bytes)
    if "params" in metrics:
        data["trainable_params"] = _get_parameter_count(model, trainable_only=True)
        data["total_params"] = _get_parameter_count(model, trainable_only=False)
    if "flops" in metrics:
        # NOTE: thops param count does not count embedding layers
        macs, _ = thop_profile(model, (input_tensor,), verbose=False)
        data["macs"] = macs
    if "wallclock" in metrics:
        time_per_example = timeit.repeat(
            lambda: model(input_tensor), repeat=iters, number=1
        )
        data["median"] = np.median(time_per_example)
        data["mean"] = np.mean(time_per_example)
        data["std"] = np.std(time_per_example)
        data["min"] = np.min(time_per_example)
        data["max"] = np.max(time_per_example)
        data["5_pct"] = np.percentile(time_per_example, 5)
        data["95_pct"] = np.percentile(time_per_example, 95)
    if not metrics:
        raise ValueError("No metrics specified.")

    logger.info(f"Total Runtime: {time.time() - init_time}")
    return data
