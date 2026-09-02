#include "tokenizer/internal/adapters/default_adapter.h"

#include <array>

#include "tokenizer/internal/detail/loaders.h"
#include "tokenizer/tokenization/pretrained_tokenizer.h"

namespace tokenizer {
namespace {

/// The contractions Hugging Face's `clean_up_tokenization_spaces` joins back
/// up, alongside the punctuation it un-spaces. This lives in Python upstream,
/// so the Rust decoder never does it and a decoded string would otherwise
/// differ from the reference by a handful of spaces.
constexpr std::array<std::string_view, 7> kContractions = {" n't", " 'm",  " 's", " 've",
                                                           " 're", " 'll", " 'd"};

constexpr std::string_view kPunctuation = ".?!,'";

}  // namespace

std::string CleanUpTokenizationSpaces(std::string_view text) {
  std::string out;
  out.reserve(text.size());

  size_t i = 0;
  while (i < text.size()) {
    if (text[i] == ' ' && i + 1 < text.size()) {
      // " ." -> "."  and the same for the other punctuation marks.
      if (kPunctuation.find(text[i + 1]) != std::string_view::npos) {
        // An apostrophe only collapses as part of a contraction, which the
        // next branch handles; a bare " '" is left alone.
        if (text[i + 1] != '\'') {
          ++i;
          continue;
        }
      }
      bool matched = false;
      for (std::string_view contraction : kContractions) {
        if (text.compare(i, contraction.size(), contraction) == 0) {
          out.append(contraction.substr(1));
          i += contraction.size();
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }
    out.push_back(text[i]);
    ++i;
  }
  return out;
}

std::string_view DefaultAdapter::name() const { return "default"; }

void DefaultAdapter::PostDecode(std::string& text, const DecodeOptions& options,
                                const TokenizerConfig& config) const {
  // Hugging Face's default when the key is absent is true; every modern
  // checkpoint states false explicitly.
  const bool clean = options.clean_up_tokenization_spaces.value_or(
      config.clean_up_tokenization_spaces.value_or(true));
  if (!clean) return;
  text = CleanUpTokenizationSpaces(text);
}

}  // namespace tokenizer
