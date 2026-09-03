#include "inferx/base/log.h"

#include <fstream>
#include <memory>
#include <mutex>
#include <utility>

#include "absl/log/initialize.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"

namespace inferx::log {
namespace {

/// \brief File-backed sink used to implement the `--log-file` policy.
///
/// Each record keeps Abseil's standard prefix and trailing newline. Abseil may
/// call `Send` concurrently, so writes are serialized and flushed before the
/// call returns.
class LogFileSink final : public absl::LogSink {
 public:
  /// \brief Opens `path` for output, truncating an existing file.
  explicit LogFileSink(const std::string& path) : file_(path, std::ios::out | std::ios::trunc) {}

  /// \brief Reports whether the destination file was opened successfully.
  [[nodiscard]] bool IsOk() const { return file_.is_open(); }

  /// \brief Writes one fully formatted Abseil log entry to the destination file.
  void Send(const absl::LogEntry& entry) override {
    const std::lock_guard<std::mutex> lock(file_mutex_);

    if (file_.is_open()) {
      file_ << entry.text_message_with_prefix_and_newline();
      file_.flush();
    }
  }

 private:
  std::mutex file_mutex_;
  std::ofstream file_;
};

/// The sink currently registered with Abseil, or null while logs use stderr.
/// Logging configuration is process-wide and must not be changed concurrently.
LogFileSink* active_file_sink = nullptr;

}  // namespace

void Initialize() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
}

void SetLevel(absl::LogSeverity minimum, int vlog_level) {
  absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(minimum));
  absl::SetGlobalVLogLevel(vlog_level);
}

bool SetLogFile(const std::string& path) {
  auto sink = std::make_unique<LogFileSink>(path);

  if (!sink->IsOk()) return false;

  ClearLogFile();
  active_file_sink = sink.release();
  absl::AddLogSink(active_file_sink);

  // While the file sink is installed it is the sole destination.
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);

  return true;
}

void ClearLogFile() {
  if (active_file_sink == nullptr) return;

  absl::RemoveLogSink(active_file_sink);
  delete active_file_sink;
  active_file_sink = nullptr;
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
}

}  // namespace inferx::log
