// Private simdjson adapter (m1.md section 3): simdjson types never leave this
// directory. Parses one flat JSON object of integer fields with strict
// duplicate/unknown/UTF-8 handling.

#ifndef INFERX_SRC_CONFIG_INTERNAL_SIMDJSON_READER_H_
#define INFERX_SRC_CONFIG_INTERNAL_SIMDJSON_READER_H_

#include <map>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace inferx::config::internal {

// Returns field-name -> value for the object; rejects unknown keys,
// duplicate keys, non-integer values, nesting > kMaxJsonNestingDepth,
// more than kMaxObjectMembers members, and inputs whose byte size exceeds
// the caller-provided limit. simdjson exceptions (if any escape the C API)
// are caught here and translated.
absl::StatusOr<std::map<std::string, uint64_t>> ParseFlatIntegerObject(absl::string_view text,
                                                                       uint64_t max_bytes);

}  // namespace inferx::config::internal

#endif  // INFERX_SRC_CONFIG_INTERNAL_SIMDJSON_READER_H_
