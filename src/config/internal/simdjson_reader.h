// Private simdjson adapter: simdjson types never leave this directory. Base
// fields remain flat integers; the schema additionally permits one `cuda` object,
// booleans in that object, and null for its optional device budget.

#ifndef INFERX_SRC_CONFIG_INTERNAL_SIMDJSON_READER_H_
#define INFERX_SRC_CONFIG_INTERNAL_SIMDJSON_READER_H_

#include <map>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace inferx::config::internal {

// Returns flattened field-name -> value (`cuda.enabled`, etc.); rejects
// duplicate keys, unsupported scalar values, nesting beyond the schema,
// more than kMaxObjectMembers members, and inputs whose byte size exceeds
// the caller-provided limit. simdjson exceptions (if any escape the C API)
// are caught here and translated.
absl::StatusOr<std::map<std::string, uint64_t>> ParseFlatIntegerObject(absl::string_view text,
                                                                       uint64_t max_bytes);

}  // namespace inferx::config::internal

#endif  // INFERX_SRC_CONFIG_INTERNAL_SIMDJSON_READER_H_
