#ifndef INFERX_BASE_LOG_H_
#define INFERX_BASE_LOG_H_

#include <string>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"

namespace inferx::log {

/// @brief
void Initialize();

void SetLevel(absl::LogSeverity minimum, int vlog_level);

[[nodiscard]] bool SetLogFile(const std::string& path);

void ClearLogFile();

}  // namespace inferx::log

#endif  // INFERX_BASE_LOG_H_
