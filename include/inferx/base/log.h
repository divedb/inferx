#ifndef INFERX_BASE_LOG_H_
#define INFERX_BASE_LOG_H_

#include <string>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"

namespace inferx::log {

/// \brief Initializes Abseil logging and enables INFO-and-above on stderr.
///
/// Abseil's default stderr threshold is ERROR, so the explicit threshold keeps
/// normal CLI diagnostics visible when no log file is configured.
void Initialize();

/// \brief Configures the process-wide severity and verbose-log filters.
///
/// \param minimum    Lowest enabled severity for `LOG` statements.
/// \param vlog_level Highest enabled verbosity for `VLOG` statements.
void SetLevel(absl::LogSeverity minimum, int vlog_level);

/// \brief Redirects logging from stderr to a file.
///
/// \param path Destination path, opened for output and truncated.
/// \return     `true` when the sink was installed. If opening fails, the current destination is
///             left unchanged and `false` is returned.
[[nodiscard]] bool SetLogFile(const std::string& path);

/// \brief Removes the active file sink and restores INFO-and-above on stderr.
///
/// This function is a no-op when no file sink is installed.
void ClearLogFile();

}  // namespace inferx::log

#endif  // INFERX_BASE_LOG_H_
