// The public header must be includable alone and first (self-containment is
// additionally enforced per header by tools/check_headers.py).
#include "inferx/base/version.h"

#include <gtest/gtest.h>

#include <string>

#include "version_second_tu.h"

namespace {

TEST(VersionTest, ValuesMatchCMakeProjectVersion) {
  const inferx::Version version = inferx::GetVersion();
  EXPECT_EQ(version.major, INFERX_EXPECTED_VERSION_MAJOR);
  EXPECT_EQ(version.minor, INFERX_EXPECTED_VERSION_MINOR);
  EXPECT_EQ(version.patch, INFERX_EXPECTED_VERSION_PATCH);
}

TEST(VersionTest, StringIsDottedTriple) {
  const std::string text(inferx::GetVersionString());
  EXPECT_EQ(text, std::to_string(INFERX_EXPECTED_VERSION_MAJOR) + "." +
                      std::to_string(INFERX_EXPECTED_VERSION_MINOR) + "." +
                      std::to_string(INFERX_EXPECTED_VERSION_PATCH));
}

TEST(VersionTest, StringAndStructAgree) {
  const inferx::Version version = inferx::GetVersion();
  const std::string text(inferx::GetVersionString());
  EXPECT_EQ(text, std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
                      std::to_string(version.patch));
}

TEST(VersionTest, UsableFromMultipleTranslationUnits) {
  // version_second_tu.cc includes the public header in its own TU; the
  // returned value must be identical (one library, one version).
  const inferx::Version here = inferx::GetVersion();
  const inferx::Version there = inferx::SecondTuGetVersion();
  EXPECT_EQ(here.major, there.major);
  EXPECT_EQ(here.minor, there.minor);
  EXPECT_EQ(here.patch, there.patch);
}

}  // namespace
