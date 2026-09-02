#pragma once

#include <cstdint>

#include "inferx/base/id.h"

namespace inferx::runtime {

struct ModelSlotTag {};
struct ModelGenerationTag {};

using ModelSlotId = StrongId<ModelSlotTag, uint32_t>;
using ModelGeneration = StrongId<ModelGenerationTag, uint32_t>;

// Hardware-neutral handle to one loaded model (ADR 0035). It carries strong
// slot/generation identity only: unloading the slot increments the
// generation, so stale tickets and handles cannot name a later model.
struct ModelHandle {
  ModelSlotId slot{0};
  ModelGeneration generation{0};

  [[nodiscard]] friend bool operator==(const ModelHandle&, const ModelHandle&) = default;
};

}  // namespace inferx::runtime
