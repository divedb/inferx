"""Stub for the deep_gemm wheel whose sdist cannot build here (vendored
CUTLASS submodule is not shipped). The benchmarked TokenSpeed operators
(silu_and_mul, rmsnorm, rope, gemm, attention decode, sampling, routing,
quantization, hadamard) do not route through deep_gemm on SM89; this stub
only satisfies the import-time names so the registry can load. Calling any
of them raises.
"""
def _unavailable(*_args, **_kwargs):
    raise NotImplementedError("deep_gemm stub: not installed in this environment")


def __getattr__(name):
    return _unavailable
