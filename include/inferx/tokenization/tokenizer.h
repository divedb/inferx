// The hardware- and backend-neutral tokenizer facade.
//
// Logical constness: encode/decode never change tokenizer semantics. The
// implementation may internally acquire a mutable, exclusive engine
// instance; public spans/views are consumed during the call and every result
// owns its memory. No dependency type appears here.

#ifndef INFERX_TOKENIZATION_TOKENIZER_H_
#define INFERX_TOKENIZATION_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/tokenization/incremental_decoder.h"
#include "inferx/tokenization/tokenizer_metadata.h"
#include "inferx/tokenization/tokenizer_options.h"

namespace inferx::tokenization {

// Checkpoint artifact bytes the qualified tokenizer consumes. The host owns
// discovery through its rooted artifact session; this layer performs no
// filesystem or network access. The input set is deliberately fixed.
struct TokenizerArtifacts {
  // Serialized tokenizer.json. Required.
  std::string tokenizer_json;
  std::optional<std::string> tokenizer_config;     // tokenizer_config.json
  std::optional<std::string> special_tokens_map;   // special_tokens_map.json
  std::optional<std::string> chat_template_jinja;  // chat_template.jinja
  std::optional<std::string> chat_template_json;   // chat_template.json
  // config.json, for the lowest-precedence special-token fallback and the
  // load-time cross-check.
  std::optional<std::string> model_config;
};

// Chat message for template rendering. Only text system, user, and assistant
// messages are supported; tools and multimodal content are not accepted.
struct ChatMessage {
  enum class Role : uint8_t { kSystem, kUser, kAssistant };
  Role role = Role::kUser;
  std::string content;
};

struct ChatRenderOptions {
  bool add_generation_prompt = true;
  // Only an explicitly named template or the checkpoint-declared default is
  // used; there is no fallback template and no inferred one.
  std::optional<std::string> template_name;
};

// How many exclusive engine instances the facade keeps. The engine handles
// are thread-affine; the facade lends them out one caller at a time.
struct TokenizerInstancePolicy {
  // 1..32; validated at load.
  uint32_t instance_count = 1;
  // Fail load when the checkpoint produced cross-check warnings.
  bool strict = false;
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  // Encodes UTF-8 text. Invalid UTF-8 is InvalidArgument; an empty string is
  // valid input (whether an empty prompt is admissible is the caller's
  // product rule).
  virtual absl::StatusOr<std::vector<TokenId>> Encode(std::string_view utf8,
                                                      const EncodeOptions& options) const = 0;

  // Decodes ids. Negative or out-of-vocabulary ids are InvalidArgument
  // before any state changes.
  virtual absl::StatusOr<std::string> Decode(std::span<const TokenId> ids,
                                             const DecodeOptions& options) const = 0;

  // Streaming decode driven by the upstream engine's state machine; see
  // incremental_decoder.h for the exact contract.
  virtual absl::StatusOr<std::unique_ptr<IncrementalDecoder>> NewIncrementalDecoder(
      const DecodeOptions& options) const = 0;

  // Renders a conversation with the checkpoint's chat template through the
  // qualified template engine. FailedPrecondition when the checkpoint ships
  // no template; InvalidArgument for invalid message shapes. The rendered
  // prompt must be encoded with add_special_tokens=false: the template owns
  // its special tokens.
  virtual absl::StatusOr<std::string> RenderChatPrompt(std::span<const ChatMessage> messages,
                                                       const ChatRenderOptions& options) const = 0;

  virtual const TokenizerMetadata& metadata() const = 0;

  // Loads and qualifies one checkpoint. The returned facade owns its engine
  // instances; metadata is derived once from the same engine resolution.
  static absl::StatusOr<std::unique_ptr<Tokenizer>> Load(
      TokenizerArtifacts artifacts, const TokenizerInstancePolicy& policy = {});
};

}  // namespace inferx::tokenization

#endif  // INFERX_TOKENIZATION_TOKENIZER_H_
