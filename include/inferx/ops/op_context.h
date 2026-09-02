// Hardware-neutral operation vocabulary and execution limits.
#ifndef INFERX_OPS_OP_CONTEXT_H_
#define INFERX_OPS_OP_CONTEXT_H_

#include <cstdint>
#include <string_view>

#include "inferx/base/token.h"
#include "inferx/tensor/device.h"

namespace inferx::ops {

inline constexpr uint32_t kOperatorContractVersion = 1;

enum class OpKind : uint8_t {
  kEmbedding,
  kGemm,
  kRmsNorm,
  kRope,
  kSilu,
  kMultiply,
  kSiluMultiply,
  kResidual,
  kAttention,
  kLogits,
};

enum class ExecutionPhase : uint8_t { kGeneric, kPrefill, kDecode };
enum class BackendId : uint8_t { kReference, kCublasLt, kInferxCuda, kFlashInfer, kCutlass };
enum class MathMode : uint8_t { kFp32Accumulate };
enum class LayoutId : uint8_t {
  kRowMajorDense,
  kQkvTokenHeadDim,
  kContiguousKvBshd,
};
enum class AliasMode : uint8_t { kDisjoint, kExactLeft, kExactRight, kExactInPlace };
enum class GraphCompatibility : uint8_t { kNotGraphCaptured };

[[nodiscard]] std::string_view OpKindName(OpKind value) noexcept;
[[nodiscard]] std::string_view BackendIdName(BackendId value) noexcept;

struct OpContext {
  Device device = Device::Host();
  ByteCount workspace_limit = ByteCount(0);
};

}  // namespace inferx::ops

#endif  // INFERX_OPS_OP_CONTEXT_H_
