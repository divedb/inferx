// Request-confined streaming decoder contract.
//
// The implementation runs the upstream engine's `step_decode_stream` state
// machine: a Push may return no bytes while a split UTF-8/byte-fallback
// sequence is incomplete, and an owned valid UTF-8 chunk is returned only
// when the upstream algorithm commits it. Never repeated one-token decode,
// never a retraction of prior output.

#ifndef INFERX_TOKENIZATION_INCREMENTAL_DECODER_H_
#define INFERX_TOKENIZATION_INCREMENTAL_DECODER_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "inferx/base/id.h"

namespace inferx::tokenization {

class IncrementalDecoder {
 public:
  virtual ~IncrementalDecoder() = default;

  // Feeds one token id. Returns the committed chunk, or nullopt when the id
  // alone commits nothing yet. Invalid/out-of-vocabulary ids fail before any
  // stream state mutates. After a failure or Finish, another Push is
  // FailedPrecondition.
  virtual absl::StatusOr<std::optional<std::string>> Push(TokenId id) = 0;

  // Flushes any valid remaining bytes and ends the stream. DataLoss when the
  // trailing state is an irrecoverably incomplete sequence. Idempotent as a
  // zero-byte observation: exactly one terminal result is produced.
  virtual absl::StatusOr<std::string> Finish() = 0;

  // True once Finish succeeded or the decoder failed terminally.
  virtual bool finished() const = 0;
};

}  // namespace inferx::tokenization

#endif  // INFERX_TOKENIZATION_INCREMENTAL_DECODER_H_
