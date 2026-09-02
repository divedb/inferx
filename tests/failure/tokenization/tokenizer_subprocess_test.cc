// Malformed-input subprocess gate (ADR 0024): no Rust unwrap/panic/abort
// may be reachable from malformed tokenizer.json bytes. The test binary
// re-executes itself as a child for each mutated blob; a child that dies by
// signal (abort) fails the suite, a child that exits cleanly with any
// status passes.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/tokenization/tokenizer.h"

namespace inferx::tokenization {
namespace {

std::string ReadFile(const char* path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.good()) return std::string();
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

int RunChild(const std::string& blob, const char* self_exe) {
  const pid_t pid = fork();
  if (pid == 0) {
    // Child: write the blob to a temp file, re-exec self in child mode.
    char path[] = "/tmp/inferx_tok_fuzz_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) _exit(99);
    if (::write(fd, blob.data(), blob.size()) != static_cast<ssize_t>(blob.size())) {
      _exit(99);
    }
    close(fd);
    execl(self_exe, self_exe, "--tokenizer-child", path, nullptr);
    _exit(98);  // exec failed
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return status;
}

// Structured malformations: valid JSON framing with hostile payloads the
// engine must reject as data errors, never as aborts.
std::vector<std::string> StructuredMalformations() {
  return {
      "{}",
      "[]",
      "null",
      "42",
      "\"string\"",
      "{\"model\": null}",
      "{\"model\": 7}",
      "{\"model\": []}",
      "{\"model\": {\"type\": \"NoSuchModel\"}}",
      "{\"model\": {\"type\": \"BPE\"}}",
      "{\"model\": {\"type\": \"BPE\", \"vocab\": \"not-a-map\"}}",
      "{\"model\": {\"type\": \"BPE\", \"vocab\": {}, \"merges\": 3}}",
      "{\"model\": {\"type\": \"BPE\", \"vocab\": {\"a\": \"x\"}, \"merges\": []}}",
      "{\"model\": {\"type\": \"BPE\", \"vocab\": {\"a\": 1e999}, \"merges\": []}}",
      "{\"model\": {\"type\": \"WordPiece\", \"vocab\": null}}",
      "{\"model\": {\"type\": \"Unigram\", \"vocab\": [[\"a\", \"b\"]]}}",
      "{\"normalizer\": {\"type\": \"NoSuchNormalizer\"}}",
      "{\"pre_tokenizer\": {\"type\": \"NoSuchPreTokenizer\"}}",
      "{\"post_processor\": {\"type\": \"NoSuchPostProcessor\"}}",
      "{\"decoder\": {\"type\": \"NoSuchDecoder\"}}",
      "{\"added_tokens\": \"not-an-array\"}",
      "{\"added_tokens\": [{\"id\": -5, \"content\": 1}]}",
      "{\"added_tokens\": [{\"id\": 4294967295, \"content\": \"x\", \"special\": true}]}",
      "{\"truncation\": {\"max_length\": -1}}",
      "{\"padding\": {\"strategy\": \"bogus\"}}",
      "{\"normalizer\": {\"type\": \"Sequence\", \"normalizers\": null}}",
      "{\"version\": \"not-a-version\", \"model\": {\"type\": \"BPE\", \"vocab\": {}, \"merges\": "
      "[]}}",
      "{\n  \"model\": {\"type\": \"BPE\", \"vocab\": {\"" + std::string(200000, 'a') +
          "\": 0}, \"merges\": []}}",
      // Deeply nested arrays inside a component.
      [] {
        std::string deep = "{\"decoder\": [";
        for (int i = 0; i < 1000; ++i) deep += "[";
        for (int i = 0; i < 1000; ++i) deep += "]";
        deep += "]}";
        return deep;
      }(),
      // Truncated fragments of a valid object.
      "{\"model\": {\"type\": \"BPE\", \"vocab\": {",
      "{\"model\": {\"type\": \"BPE\"",
      "{\"model\":",
  };
}

TEST(TokenizerSubprocessTest, MalformedInputNeverAborts) {
  const char* fixture = TOKENIZER_JSON_PATH;
  const std::string valid = ReadFile(fixture);
  ASSERT_FALSE(valid.empty()) << "fixture missing: " << fixture;

  char self[4096];
  const ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
  ASSERT_GT(len, 0);
  self[len] = '\0';

  std::vector<std::string> blobs = StructuredMalformations();

  // Seeded byte-level mutations of the real blob (truncations, flips,
  // splices). The first 128 KiB contains the full model component of the
  // fixture, so mutations reach deep parser paths while children stay fast.
  const std::string prefix = valid.substr(0, std::min<size_t>(valid.size(), 128u << 10));
  std::mt19937 rng(20260902);
  for (int i = 0; i < 150; ++i) {
    std::string blob = prefix;
    const int kind = rng() % 4;
    if (kind == 0) {
      blob.resize(rng() % blob.size());
    } else if (kind == 1) {
      for (int f = 0; f < 8; ++f) {
        blob[rng() % blob.size()] = static_cast<char>(rng() % 256);
      }
    } else if (kind == 2) {
      const size_t at = rng() % blob.size();
      const size_t count = rng() % 256;
      blob.insert(at, count, static_cast<char>(rng() % 256));
    } else {
      const size_t at = rng() % blob.size();
      blob.erase(at, rng() % 4096);
    }
    blobs.push_back(std::move(blob));
  }

  for (const std::string& blob : blobs) {
    const int status = RunChild(blob, self);
    ASSERT_TRUE(WIFEXITED(status))
        << "child died by signal " << (WIFSIGNALED(status) ? WTERMSIG(status) : -1)
        << "; a malformed tokenizer reached an engine abort";
    ASSERT_EQ(WEXITSTATUS(status), 0) << "child exited nonzero (" << WEXITSTATUS(status) << ")";
  }
}

}  // namespace

// Child mode: load the blob at argv[2]; success is a clean exit regardless
// of the load result (any status, no signal). argv[1] is --tokenizer-child.
int ChildMain(const char* path) {
  const std::string blob = ReadFile(path);
  ::unlink(path);
  if (blob.empty()) return 0;
  TokenizerArtifacts artifacts;
  artifacts.tokenizer_json = blob;
  auto loaded = Tokenizer::Load(std::move(artifacts));
  if (!loaded.ok()) return 0;  // rejected as data: the required behavior
  // A malformed blob that claims to load: still no abort, still success.
  return 0;
}

}  // namespace inferx::tokenization

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--tokenizer-child") {
    return inferx::tokenization::ChildMain(argv[2]);
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
