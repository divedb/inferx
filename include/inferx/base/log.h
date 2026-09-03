// Minimal logging facade: severity-gated, process-wide, swappable sink.
// Logs are diagnostics — command results never route through here (the
// CLI contract keeps stdout pipeable; logs go to stderr or --log-file).

#ifndef INFERX_BASE_LOG_H_
#define INFERX_BASE_LOG_H_

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

namespace inferx::log {

enum class Severity : uint8_t { kTrace, kDebug, kInfo, kWarning, kError };

class Sink {
 public:
  virtual ~Sink() = default;
  virtual void Write(Severity severity, std::string_view message) = 0;
};

// One line per record: "error: <message>", "warning: <message>", or the
// bare message at info and below. The trailing newline is part of the line.
[[nodiscard]] std::string FormatLine(Severity severity, std::string_view message);

[[nodiscard]] bool Enabled(Severity severity);
[[nodiscard]] Severity MinSeverity();
void SetMinSeverity(Severity severity);

// Installs the process-wide sink; nullptr restores the default stderr sink.
void SetSink(std::shared_ptr<Sink> sink);

// Redirects logs to `path` (truncated). Keeps the current sink and returns
// false when the file cannot be opened.
[[nodiscard]] bool SetFileSink(const std::string& path);

// Emits one record to the active sink (newline handling is the sink's).
void Write(Severity severity, std::string_view message);

namespace detail {

// Severity accessors behind LOG(severity): the macro calls the function
// named by the token, so LOG(ERROR) resolves to detail::ERROR().
inline Severity TRACE() { return Severity::kTrace; }
inline Severity DEBUG() { return Severity::kDebug; }
inline Severity INFO() { return Severity::kInfo; }
inline Severity WARNING() { return Severity::kWarning; }
inline Severity ERROR() { return Severity::kError; }

// LOG(severity) statement: buffers the message, emits it on destruction
// when the severity is enabled.
class LogStatement {
 public:
  explicit LogStatement(Severity severity) : severity_(severity) {}
  ~LogStatement() {
    if (Enabled(severity_)) Write(severity_, message_.str());
  }
  LogStatement(const LogStatement&) = delete;
  LogStatement& operator=(const LogStatement&) = delete;
  [[nodiscard]] std::ostringstream& stream() { return message_; }

 private:
  Severity severity_;
  std::ostringstream message_;
};

}  // namespace detail
}  // namespace inferx::log

#define LOG(severity) \
  ::inferx::log::detail::LogStatement(::inferx::log::detail::severity()).stream()

#endif  // INFERX_BASE_LOG_H_
