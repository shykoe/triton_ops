import torch
from triton_test.benchmark.utils import SingleBenchmarkRunOutput, QUANTILES, run_benchmarks, SingleBenchmarkRunInput
import triton

def bench_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]
    x_shape = (M, N)
    x = torch.randn(x_shape, dtype=dtype, device=input.device)
    x.requires_grad_(True)
    dy = torch.randn_like(x)
    
    rms_norm_torch = torch.nn.RMSNorm([N], eps=eps, dtype=dtype).to(input.device)
    rms_norm_torch_ref = torch.compile(rms_norm_torch)

    def y_fwd():
        if provider == "torch":
            return rms_norm_torch(x)
        if provider == "torch_compile":
            return rms_norm_torch_ref(x)
        
    def full():
        y = y_fwd()
        y.backward(dy, retain_graph=True)
    ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[x],
            rep=500,
            quantiles=QUANTILES,
        )
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )
if __name__ == "__main__":
    common_configs = {
        "kernel_name": "rms_norm",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 16)],
        "kernel_providers": ["torch", "torch_compile"],
        "extra_benchmark_configs": [{"M": 2048, "dtype": torch.bfloat16, "eps": 1e-6}],
        "overwrite": True,
    }
    run_benchmarks(
        bench_test_fn=bench_rms_norm,
        kernel_operation_modes=["full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )