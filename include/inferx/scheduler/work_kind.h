// Hardware-neutral scheduler work vocabulary.

#ifndef INFERX_SCHEDULER_WORK_KIND_H_
#define INFERX_SCHEDULER_WORK_KIND_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"

namespace inferx {

enum class WorkKind : uint8_t {
  kPrefill,
  kDecode,
};

[[nodiscard]] absl::string_view ToString(WorkKind kind);
[[nodiscard]] std::optional<WorkKind> WorkKindFromName(absl::string_view name);

}  // namespace inferx

#endif  // INFERX_SCHEDULER_WORK_KIND_H_
