#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/base/token.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/parameter_spec.h"
#include "inferx/model/weight_plan.h"
#include "inferx/tensor/dtype.h"

namespace inferx::runtime {

// Deterministic device/host memory plan computed before any model allocation
// (m5.md section 9). Every offset is checked 64-bit arithmetic; the plan is
// content, not addresses: it carries no pointers, allocation IDs, or timing.
struct WeightPlacement {
  model::ParameterId parameter;
  uint64_t offset = 0;  // bytes into the weight arena
  uint64_t bytes = 0;
  uint64_t alignment = 0;
  uint64_t padding_after = 0;  // bytes inserted before the next placement
  bool alias = false;          // tied parameters allocate no bytes
};

struct ActivationBufferPlan {
  std::string name;
  uint64_t elements = 0;  // per maximum prefill tokens
};

struct ModelMemoryPlan {
  static constexpr uint32_t kSchemaVersion = 1;

  std::vector<WeightPlacement> weights;  // canonical ParameterId order
  uint64_t weight_arena_bytes = 0;
  uint64_t weight_padding_bytes = 0;
  uint64_t kv_bytes = 0;  // separate rotated K and V arenas, one backing range
  uint64_t kv_alignment = 0;
  std::vector<ActivationBufferPlan> activations;  // element counts, FP32 family
  uint64_t activation_bytes = 0;
  uint64_t logits_bytes = 0;  // final-row FP32 [vocab]
  uint64_t host_logits_bytes = 0;
  uint64_t total_device_bytes = 0;
  uint64_t max_prefill_tokens = 0;
  uint64_t context_capacity = 0;
  Dtype execution_dtype = Dtype::kFloat32;
};

struct MemoryPlanRequest {
  const model::ModelSpec* spec = nullptr;
  const model::WeightPlan* weight_plan = nullptr;
  uint64_t max_prefill_tokens = 512;
  uint64_t context_capacity = 4096;
  uint64_t weight_alignment = 256;  // schema-v1 floor
};

class ModelMemoryPlanner {
 public:
  // Fails before any allocation when geometry is inconsistent or arithmetic
  // overflows; alias parameters (tied LM head) place zero bytes.
  [[nodiscard]] static absl::StatusOr<ModelMemoryPlan> Plan(const MemoryPlanRequest& request);
};

}  // namespace inferx::runtime
