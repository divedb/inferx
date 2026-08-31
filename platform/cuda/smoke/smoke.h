// CUDA device/kernel smoke API (m0.md section 10.2). Intentionally private to
// the test subtree — this is not the M2 RAII/runtime design and may be
// replaced wholesale then.
#ifndef INFERX_PLATFORM_CUDA_SMOKE_SMOKE_H_
#define INFERX_PLATFORM_CUDA_SMOKE_SMOKE_H_

namespace inferx::cuda_smoke {

// Outcome of the smoke run.
enum class SmokeResult {
  kOk = 0,        // device written data validated
  kNoDevice = 1,  // no visible CUDA device (developer machines return exit 77)
  kFailed = 2,    // device present but a check failed
};

// Queries devices, launches the pattern kernel, copies back, and validates
// every element. All CUDA calls are checked; resources are released on every
// exit path. Output on success: device name and compute capability.
//
// `accepted_architectures` is the CMake-accepted list (e.g. "89;90"); a
// device outside it is a failure (kFailed), not a skip: the declared matrix
// must match reality on an owned GPU runner.
SmokeResult RunSmoke(const char* accepted_architectures);

}  // namespace inferx::cuda_smoke

#endif  // INFERX_PLATFORM_CUDA_SMOKE_SMOKE_H_
