from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import torch
import json
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SingleBenchmarkRunInput:
    x: Union[int, float]
    kernel_provider: str
    kernel_operation_mode: Optional[str] = ""
    extra_benchmark_config: Optional[Dict[str, Any]] = None
    device: str = "cuda"

@dataclass
class SingleBenchmarkRunOutput:
    # 20th percentile
    y_20: float
    # 50th percentile (median)
    y_50: float
    # 80th percentile
    y_80: float

def get_formatted_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super().default(self, obj)


@dataclass
class BenchmarkData:
    """
    BenchmarkData is a dataclass to store the benchmark data for a a completed benchmark
    run on all x-values for a given kernel/kernel operation mode/metric/extra_benchmark_config
    """

    kernel_name: str
    kernel_provider: str
    metric_name: str
    metric_unit: str
    gpu_name: str
    x_name: str
    x_label: str
    x_values: List[float]
    y_values_50: List[float]
    y_values_20: List[float]
    y_values_80: List[float]
    timestamp: str
    kernel_operation_mode: Optional[str] = None
    extra_benchmark_config_str: Optional[str] = None


QUANTILES = [0.5, 0.2, 0.8]

def _print_benchmarking_banner(metric_name: str, kernel_name: str):
    print("**************************************")
    print(f"     BENCHMARKING {metric_name.upper()} for {kernel_name.upper()}")
    print("**************************************")

def get_gpu_name():
    """
    Returns the current GPU name, formatted to serve as a directory name
    """
    torch_device = getattr(torch, device)
    if torch_device.is_available():
        gpu_name = torch_device.get_device_name(torch_device.current_device())
        return gpu_name
    else:
        raise Exception("Benchmarks can only be run on GPU.")
    
def print_benchmark_data(benchmark_data_list: List[BenchmarkData]) -> str:
    print("********** Benchmark Data **********")
    formatted_list = [obj.__dict__ for obj in benchmark_data_list]
    print(json.dumps(formatted_list, indent=2))

def run_benchmarks(
    bench_test_fn: Callable,
    kernel_name: str,
    metric_name: str,
    metric_unit: str,
    x_name: str,
    x_label: str,
    x_values: List[Union[float, int]],
    kernel_providers: List[str],
    kernel_operation_modes: Optional[List[str]] = [None],
    extra_benchmark_configs: Optional[List[Dict[str, Any]]] = None,
    overwrite: bool = False,
):
    """
    Run benchmarks given a bench_test_fn that takes in a SingleBenchmarkRunInput as input and
    saves data to the CSV file.

    Args:
        - bench_test_fn: The benchmark test function to run. This function should take in a
            SingleBenchmarkRunInput as input and return a SingleBenchmarkRunOutput.
        - kernel_name: The name of the kernel being benchmarked (e.g. "swiglu")
        - metric_name: The name of the metric being benchmarked (e.g. "speed" or "memory")
        - metric_unit: The unit of the metric being benchmarked (e.g. "ms" or "MB")
        - x_name: The name of the x-axis (e.g. "T" for sequence length)
        - x_label: The label of the x-axis (e.g. "sequence length")
        - x_values: The list of x-values to run the benchmark on (e.g. [2**i for i in range(10, 14)])
        - kernel_providers: The list of kernel providers to run the benchmark on (e.g. ["liger", "huggingface"])
        - kernel_operation_modes: The list of kernel operation modes to run the benchmark on (e.g. ["full", "backward"])
        - extra_benchmark_configs: The list of extra benchmark configurations to run the benchmark on.
        - overwrite: Whether to overwrite the existing benchmark data entry if it already exists.
    """

    assert len(kernel_operation_modes) >= 1
    assert len(kernel_providers) >= 1

    _print_benchmarking_banner(metric_name=metric_name, kernel_name=kernel_name)

    gpu_name = get_gpu_name()
    benchmark_data_list = []
    for extra_benchmark_config in extra_benchmark_configs:
        for kernel_operation_mode in kernel_operation_modes:
            for kernel_provider in kernel_providers:
                y_values_50 = []
                y_values_20 = []
                y_values_80 = []

                for x in x_values:
                    single_benchmark_run_input = SingleBenchmarkRunInput(
                        x=x,
                        kernel_provider=kernel_provider,
                        kernel_operation_mode=kernel_operation_mode,
                        extra_benchmark_config=extra_benchmark_config,
                    )
                    benchmark_result: SingleBenchmarkRunOutput = bench_test_fn(single_benchmark_run_input)
                    y_values_50.append(benchmark_result.y_50)
                    y_values_20.append(benchmark_result.y_20)
                    y_values_80.append(benchmark_result.y_80)

                benchmark_run_data = BenchmarkData(
                    kernel_name=kernel_name,
                    kernel_operation_mode=kernel_operation_mode,
                    kernel_provider=kernel_provider,
                    metric_name=metric_name,
                    metric_unit=metric_unit,
                    gpu_name=gpu_name,
                    x_name=x_name,
                    x_label=x_label,
                    x_values=x_values,
                    y_values_50=y_values_50,
                    y_values_20=y_values_20,
                    y_values_80=y_values_80,
                    extra_benchmark_config_str=json.dumps(extra_benchmark_config, cls=CustomEncoder),
                    timestamp=get_formatted_time(),
                )

                benchmark_data_list.append(benchmark_run_data)

    print_benchmark_data(benchmark_data_list)
