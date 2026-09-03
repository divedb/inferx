// Shared internal helpers for the per-command implementation files under
// src/command/. Not installed and not part of the command API.

#ifndef INFERX_COMMAND_COMMON_H_
#define INFERX_COMMAND_COMMON_H_

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "inferx/artifacts/model_resolver.h"
#include "inferx/base/log.h"
#include "inferx/command/options.h"

namespace inferx::command {
namespace internal {

/// Command results go to stdout; logs and failure diagnostics to stderr
/// (or --log-file) via the process logger.
inline std::ostream& kOut = std::cout;

inline artifacts::ModelResolverOptions ResolverOptions(
    const command::ResolverOptions& options) {
  artifacts::ModelResolverOptions result;
  result.revision = options.revision;
  result.local_files_only = options.offline;
  if (!options.download_dir.empty())
    result.download_dir = std::filesystem::path(options.download_dir);
  return result;
}

inline void Unavailable(std::string_view command, std::string_view detail) {
  LOG(ERROR) << command << ": feature unavailable: " << detail;
}

inline void Fail(std::string_view scope, const absl::Status& status) {
  LOG(ERROR) << scope << ": " << status;
}

}  // namespace internal
}  // namespace inferx::command

#endif  // INFERX_COMMAND_COMMON_H_
