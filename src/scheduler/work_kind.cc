#include "inferx/scheduler/work_kind.h"

namespace inferx {

absl::string_view ToString(WorkKind kind) {
  switch (kind) {
    case WorkKind::kPrefill:
      return "prefill";
    case WorkKind::kDecode:
      return "decode";
  }
  return "unknown";
}

std::optional<WorkKind> WorkKindFromName(absl::string_view name) {
  if (name == "prefill") {
    return WorkKind::kPrefill;
  }
  if (name == "decode") {
    return WorkKind::kDecode;
  }
  return std::nullopt;
}

}  // namespace inferx
