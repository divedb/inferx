#ifndef INFERX_OPS_KERNEL_REGISTRY_H_
#define INFERX_OPS_KERNEL_REGISTRY_H_

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/ops/backend_capability.h"

namespace inferx::ops {

struct KernelSelection {
  BackendId backend = BackendId::kReference;
  std::string capability_id;
  size_t registry_index = 0;
  CapabilityMatch match;
};

class KernelRegistry {
 public:
  explicit KernelRegistry(size_t capacity);

  [[nodiscard]] absl::Status Register(BackendCapability capability);
  [[nodiscard]] absl::Status Freeze();
  [[nodiscard]] bool frozen() const noexcept { return frozen_; }
  [[nodiscard]] size_t size() const noexcept { return entries_.size(); }
  [[nodiscard]] absl::StatusOr<KernelSelection> Select(
      const KernelKey& key, std::optional<BackendId> forced_backend = std::nullopt) const;
  [[nodiscard]] const BackendCapability& capability(const KernelSelection& selection) const;

 private:
  size_t capacity_ = 0;
  bool frozen_ = false;
  std::vector<BackendCapability> entries_;
};

[[nodiscard]] absl::Status RegisterReferenceCapabilities(KernelRegistry& registry);

struct PreparedKernelPlan {
  KernelKey key;
  BackendId backend = BackendId::kReference;
  std::string capability_id;
  uint64_t workspace_bytes = 0;
  uint32_t workspace_alignment = 1;
};

class PreparedKernelCache {
 public:
  explicit PreparedKernelCache(size_t capacity);
  [[nodiscard]] absl::Status Insert(PreparedKernelPlan plan);
  [[nodiscard]] absl::Status Freeze();
  [[nodiscard]] const PreparedKernelPlan* Find(const KernelKey& key) const noexcept;
  [[nodiscard]] size_t size() const noexcept { return plans_.size(); }
  [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] bool frozen() const noexcept { return frozen_; }

 private:
  size_t capacity_ = 0;
  bool frozen_ = false;
  std::vector<PreparedKernelPlan> plans_;
};

}  // namespace inferx::ops

#endif  // INFERX_OPS_KERNEL_REGISTRY_H_
