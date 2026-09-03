// CLI logging policy on top of Abseil logging. Emit logs with absl's
// LOG(INFO/WARNING/ERROR) and VLOG(n) directly; this header only decides
// where they go — stderr by default, --log-file to redirect — so stdout
// stays reserved for pipeable command output.

#ifndef INFERX_BASE_LOG_H_
#define INFERX_BASE_LOG_H_

#include <string>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"

namespace inferx::log {

// Initializes absl logging and routes info-and-above to stderr (absl's own
// default stderr threshold is kError, which would hide INFO/WARNING).
void Initialize();

// Applies the --log-level choice: `minimum` gates the LOG severities,
// `vlog_level` enables VLOG(n) statements up to that level
// (--log-level debug/trace map to 1/2).
void SetLevel(absl::LogSeverity minimum, int vlog_level);

// Redirects every log line to `path` (truncated) instead of stderr.
// Returns false and keeps stderr when the file cannot be opened.
[[nodiscard]] bool SetLogFile(const std::string& path);

// Releases a previously installed file sink; logs return to stderr.
void ClearLogFile();

}  // namespace inferx::log

#endif  // INFERX_BASE_LOG_H_
