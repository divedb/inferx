// Logger facade tests (inferx/base/log.h).
#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/base/log.h"

namespace {

using inferx::log::Severity;

class CaptureSink final : public inferx::log::Sink {
 public:
  struct Record {
    Severity severity;
    std::string message;
  };

  void Write(Severity severity, std::string_view message) override {
    records.push_back(Record{severity, std::string(message)});
  }

  std::vector<Record> records;
};

class LogTest : public ::testing::Test {
 protected:
  void TearDown() override {
    inferx::log::SetMinSeverity(Severity::kInfo);
    inferx::log::SetSink(nullptr);
  }
};

TEST_F(LogTest, StatementEmitsToInstalledSink) {
  auto sink = std::make_shared<CaptureSink>();
  inferx::log::SetSink(sink);
  LOG(INFO) << "hello " << 42;
  ASSERT_EQ(sink->records.size(), 1u);
  EXPECT_EQ(sink->records[0].severity, Severity::kInfo);
  EXPECT_EQ(sink->records[0].message, "hello 42");
}

TEST_F(LogTest, SeverityGateSuppressesBelowMinimum) {
  auto sink = std::make_shared<CaptureSink>();
  inferx::log::SetSink(sink);
  inferx::log::SetMinSeverity(Severity::kError);
  LOG(TRACE) << "dropped";
  LOG(DEBUG) << "dropped";
  LOG(INFO) << "dropped";
  LOG(WARNING) << "dropped";
  LOG(ERROR) << "kept";
  ASSERT_EQ(sink->records.size(), 1u);
  EXPECT_EQ(sink->records[0].severity, Severity::kError);
  EXPECT_EQ(sink->records[0].message, "kept");
}

TEST_F(LogTest, NullptrSinkRestoresDefault) {
  inferx::log::SetSink(std::make_shared<CaptureSink>());
  inferx::log::SetSink(nullptr);
  LOG(ERROR) << "back on the default stderr sink";
  SUCCEED();
}

TEST_F(LogTest, FormatLinePrefixesSeverityAndTerminates) {
  EXPECT_EQ(inferx::log::FormatLine(Severity::kError, "boom"), "error: boom\n");
  EXPECT_EQ(inferx::log::FormatLine(Severity::kWarning, "w"), "warning: w\n");
  EXPECT_EQ(inferx::log::FormatLine(Severity::kInfo, "plain"), "plain\n");
  EXPECT_EQ(inferx::log::FormatLine(Severity::kDebug, "d"), "d\n");
  EXPECT_EQ(inferx::log::FormatLine(Severity::kTrace, "t"), "t\n");
}

TEST_F(LogTest, FileSinkWritesFormattedLines) {
  const std::string path = std::string(::testing::TempDir()) + "/inferx_log_test.log";
  ASSERT_TRUE(inferx::log::SetFileSink(path));
  LOG(ERROR) << "to file";
  inferx::log::SetSink(nullptr);  // releases the file sink, flushing the line

  std::ifstream file(path);
  std::string line;
  std::getline(file, line);
  EXPECT_EQ(line, "error: to file");
}

TEST_F(LogTest, UnopenableFileSinkIsRejectedAndKeepsCurrentSink) {
  auto sink = std::make_shared<CaptureSink>();
  inferx::log::SetSink(sink);
  const std::string bad = std::string(::testing::TempDir()) + "/no_such_dir_xyz/log.txt";
  EXPECT_FALSE(inferx::log::SetFileSink(bad));
  LOG(WARNING) << "still captured";
  ASSERT_EQ(sink->records.size(), 1u);
  EXPECT_EQ(sink->records[0].message, "still captured");
}

}  // namespace
