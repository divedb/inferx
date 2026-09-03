#include "inferx/base/log.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <utility>

namespace inferx::log {
namespace {

class StderrSink final : public Sink {
 public:
  void Write(Severity severity, std::string_view message) override {
    std::cerr << FormatLine(severity, message);
  }
};

class FileSink final : public Sink {
 public:
  explicit FileSink(const std::string& path)
      : file_(path, std::ios::out | std::ios::trunc) {}

  [[nodiscard]] bool ok() const { return file_.is_open(); }

  void Write(Severity severity, std::string_view message) override {
    if (file_.is_open()) file_ << FormatLine(severity, message);
  }

 private:
  std::ofstream file_;
};

std::mutex& RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::atomic<Severity>& MinimumSeverity() {
  static std::atomic<Severity> severity{Severity::kInfo};
  return severity;
}

std::shared_ptr<Sink>& ActiveSink() {
  static std::shared_ptr<Sink> sink = std::make_shared<StderrSink>();
  return sink;
}

}  // namespace

std::string FormatLine(Severity severity, std::string_view message) {
  std::string line;
  line.reserve(message.size() + 12);
  switch (severity) {
    case Severity::kError:
      line += "error: ";
      break;
    case Severity::kWarning:
      line += "warning: ";
      break;
    case Severity::kTrace:
    case Severity::kDebug:
    case Severity::kInfo:
      break;
  }
  line += message;
  line += '\n';
  return line;
}

bool Enabled(Severity severity) { return severity >= MinimumSeverity().load(); }

Severity MinSeverity() { return MinimumSeverity().load(); }

void SetMinSeverity(Severity severity) { MinimumSeverity().store(severity); }

void SetSink(std::shared_ptr<Sink> sink) {
  const std::lock_guard<std::mutex> lock(RegistryMutex());
  ActiveSink() = sink != nullptr ? std::move(sink) : std::make_shared<StderrSink>();
}

bool SetFileSink(const std::string& path) {
  auto file = std::make_shared<FileSink>(path);
  if (!file->ok()) return false;
  SetSink(std::move(file));
  return true;
}

void Write(Severity severity, std::string_view message) {
  std::shared_ptr<Sink> sink;
  {
    const std::lock_guard<std::mutex> lock(RegistryMutex());
    sink = ActiveSink();
  }
  if (sink != nullptr) sink->Write(severity, message);
}

}  // namespace inferx::log
