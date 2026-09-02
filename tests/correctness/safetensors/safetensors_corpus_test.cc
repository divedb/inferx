#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/artifacts/safetensors_reader.h"

namespace inferx {
namespace {

TEST(SafeTensorsCorpusTest, MatchesCommittedAcceptanceContract) {
  const std::filesystem::path corpus =
      std::filesystem::path(INFERX_ARTIFACT_FIXTURE_DIR) / "safetensors";
  auto session = artifacts::ModelLocator::OpenLocal(corpus);
  ASSERT_TRUE(session.ok()) << session.status();
  std::ifstream table(corpus / "corpus.tsv");
  ASSERT_TRUE(table.good());

  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(table, line)));
  ASSERT_EQ(line, "file\taccepted\treason");
  size_t cases = 0;
  while (std::getline(table, line)) {
    if (line.empty()) continue;
    std::istringstream row(line);
    std::string filename;
    std::string accepted;
    std::string reason;
    ASSERT_TRUE(static_cast<bool>(std::getline(row, filename, '\t')));
    ASSERT_TRUE(static_cast<bool>(std::getline(row, accepted, '\t')));
    ASSERT_TRUE(static_cast<bool>(std::getline(row, reason)));
    std::string trace = filename;
    trace.append(": ");
    trace.append(reason);
    SCOPED_TRACE(trace);

    auto relative = artifacts::SafeRelativePath::Parse(filename);
    ASSERT_TRUE(relative.ok()) << relative.status();
    auto file = session->OpenRegular(*relative);
    ASSERT_TRUE(file.ok()) << file.status();
    artifacts::SafeTensorReader reader;
    const auto metadata = reader.ReadHeader(*file);
    if (accepted == "true") {
      ASSERT_TRUE(metadata.ok()) << metadata.status();
    } else {
      ASSERT_EQ(accepted, "false");
      EXPECT_FALSE(metadata.ok());
    }
    ++cases;
  }
  EXPECT_EQ(cases, 8);
}

}  // namespace
}  // namespace inferx
