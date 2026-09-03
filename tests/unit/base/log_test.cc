// CLI logging policy tests (inferx/base/log.h wrapping absl::log).
#include <fstream>
#include <iterator>
#include <string>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "gtest/gtest.h"
#include "inferx/base/log.h"

namespace {

class LogConfigTest : public ::testing::Test {
 protected:
  void TearDown() override {
    inferx::log::SetLevel(absl::LogSeverity::kInfo, 0);
    inferx::log::ClearLogFile();
  }

  static std::string FileContent(const std::string& path) {
    std::ifstream file(path);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  }
};

TEST_F(LogConfigTest, InitializeRoutesInfoAndAboveToStderr) {
  inferx::log::Initialize();
  EXPECT_EQ(absl::StderrThreshold(), absl::LogSeverityAtLeast::kInfo);
}

TEST_F(LogConfigTest, SetLevelGatesSeverities) {
  inferx::log::SetLevel(absl::LogSeverity::kError, 0);
  EXPECT_EQ(absl::MinLogLevel(), absl::LogSeverityAtLeast::kError);

  inferx::log::SetLevel(absl::LogSeverity::kWarning, 3);
  EXPECT_EQ(absl::MinLogLevel(), absl::LogSeverityAtLeast::kWarning);
}

TEST_F(LogConfigTest, FileSinkReceivesEnabledSeveritiesOnly) {
  const std::string path = std::string(::testing::TempDir()) + "/inferx_log_test.log";
  inferx::log::SetLevel(absl::LogSeverity::kError, 0);
  ASSERT_TRUE(inferx::log::SetLogFile(path));
  LOG(INFO) << "suppressed marker";
  LOG(ERROR) << "recorded marker";
  inferx::log::ClearLogFile();  // releases and flushes the sink

  const std::string content = FileContent(path);
  EXPECT_NE(content.find("recorded marker"), std::string::npos);
  EXPECT_EQ(content.find("suppressed marker"), std::string::npos);
}

TEST_F(LogConfigTest, VLogLevelEnablesVerboseStatements) {
  const std::string path = std::string(::testing::TempDir()) + "/inferx_log_vlog.log";
  inferx::log::SetLevel(absl::LogSeverity::kInfo, 1);
  ASSERT_TRUE(inferx::log::SetLogFile(path));
  VLOG(1) << "verbose one";
  VLOG(2) << "verbose two";
  LOG(WARNING) << "warned marker";
  inferx::log::ClearLogFile();

  const std::string content = FileContent(path);
  EXPECT_NE(content.find("verbose one"), std::string::npos);
  EXPECT_EQ(content.find("verbose two"), std::string::npos);
  EXPECT_NE(content.find("warned marker"), std::string::npos);
}

TEST_F(LogConfigTest, FileSinkIsTheSoleDestinationWhileInstalled) {
  const std::string path = std::string(::testing::TempDir()) + "/inferx_log_sole.log";
  ASSERT_TRUE(inferx::log::SetLogFile(path));
  EXPECT_EQ(absl::StderrThreshold(), absl::LogSeverityAtLeast::kInfinity);
  inferx::log::ClearLogFile();
  EXPECT_EQ(absl::StderrThreshold(), absl::LogSeverityAtLeast::kInfo);
}

TEST_F(LogConfigTest, UnopenableFileKeepsStderr) {
  inferx::log::Initialize();
  const std::string bad = std::string(::testing::TempDir()) + "/no_such_dir_xyz/log.txt";
  EXPECT_FALSE(inferx::log::SetLogFile(bad));
  EXPECT_EQ(absl::StderrThreshold(), absl::LogSeverityAtLeast::kInfo);
}

}  // namespace
