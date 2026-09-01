// Provider vocabulary for the unified kernels layer (ADR 0031).
//
// The CUDA backend resolves every operator through a fixed provider
// preference chain: hpc-ops -> flashinfer -> custom CUTLASS -> vendor/owned
// tail. The first provider whose availability probe accepts the request
// (compute capability, dtype, shapes) wins; a forced provider bypasses the
// chain and fails with Unimplemented instead of silently falling back.
#ifndef INFERX_KERNELS_PROVIDER_H_
#define INFERX_KERNELS_PROVIDER_H_

#include <cstdint>
#include <span>
#include <string_view>

namespace inferx::kernels {

inline constexpr uint32_t kKernelProviderContractVersion = 1;

enum class ProviderId : uint8_t {
  kHpcOps = 0,      // Tencent hpc-ops ahead-of-time objects (SM90+ at the pin).
  kFlashInfer = 1,  // FlashInfer native C++ kernels at the qualified pin.
  kCutlass = 2,     // Custom CUTLASS implementations maintained by InferX.
  kCublasLt = 3,    // NVIDIA cuBLASLt vendor library (gemm/logits).
  kInferxOwned = 4, // InferX-owned fallback kernels (ADR 0030 style).
};

[[nodiscard]] std::string_view ProviderIdName(ProviderId value) noexcept;

// Preference chain consulted per operator, strongest first. Providers that do
// not implement an operator simply never register for it.
[[nodiscard]] std::span<const ProviderId> ProviderPreferenceChain() noexcept;

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_PROVIDER_H_
