// Owned C ABI of the InferX tokenizer engine shim (ADR 0024).
//
// Declares exactly the entry points of third_party/tokenizer/rust/src/lib.rs.
// Every fallible call returns one of the IX_* status codes below and, on
// failure, writes an IxError record whose message the caller frees with
// ixtok_free_error. Every returned buffer is owned by the caller and freed
// explicitly. No entry point may abort, panic, or unwind into C++.

#ifndef TOKENIZER_BACKEND_INFERX_ABI_H_
#define TOKENIZER_BACKEND_INFERX_ABI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IX_OK 0
/* Stream step succeeded but no chunk is committed yet. */
#define IX_PENDING 1
/* Token/id lookup missed. */
#define IX_NOT_FOUND 2
#define IX_INVALID_ARGUMENT 3
#define IX_DATA_LOSS 4
#define IX_INTERNAL 5
/* The engine panicked; an engine or shim bug, never a data error. */
#define IX_PANIC 6

/* Opaque engine handle. Single-owner, thread-affine; never called
 * concurrently. */
typedef struct IxTokenizer IxTokenizer;
/* Externalized upstream streaming-decode state; borrows a tokenizer only
 * for the duration of one step call. */
typedef struct IxDecodeStream IxDecodeStream;

typedef struct {
  int32_t code;
  uint8_t* message;
  uintptr_t message_len;
} IxError;

IxTokenizer* ixtok_new(const uint8_t* json, uintptr_t json_len, IxError* err);
void ixtok_free(IxTokenizer* handle);
void ixtok_free_error(IxError* err);
void ixtok_free_bytes(uint8_t* ptr, uintptr_t len);
void ixtok_free_ids(uint32_t* ptr, uintptr_t len);
void ixtok_free_offsets(uint64_t* ptr, uintptr_t len);

int32_t ixtok_encode(IxTokenizer* handle, const uint8_t* text, uintptr_t text_len,
                     int32_t add_special_tokens, uint32_t** out_ids, uintptr_t* out_len,
                     IxError* err);
int32_t ixtok_decode(IxTokenizer* handle, const uint32_t* ids, uintptr_t ids_len,
                     int32_t skip_special_tokens, uint8_t** out_bytes, uintptr_t* out_len,
                     IxError* err);

uintptr_t ixtok_vocab_size(IxTokenizer* handle);
int32_t ixtok_id_to_token(IxTokenizer* handle, uint32_t id, uint8_t** out_bytes, uintptr_t* out_len,
                          IxError* err);
int32_t ixtok_token_to_id(IxTokenizer* handle, const uint8_t* token, uintptr_t token_len,
                          uint32_t* out_id, IxError* err);
/* One-shot vocabulary dump: `count` sorted ids, packed token bytes, and
 * count+1 offsets delimiting each token's byte range. */
int32_t ixtok_vocab_dump(IxTokenizer* handle, uint32_t** out_ids, uint64_t** out_offsets,
                         uint8_t** out_chars, uintptr_t* out_chars_len, uintptr_t* out_count,
                         IxError* err);
/* Sorted ids the engine's added vocabulary marks special. */
int32_t ixtok_special_ids(IxTokenizer* handle, uint32_t** out_ids, uintptr_t* out_len,
                          IxError* err);

IxDecodeStream* ixtok_stream_new(int32_t skip_special_tokens);
void ixtok_stream_free(IxDecodeStream* stream);
/* IX_OK with an owned chunk, or IX_PENDING when no chunk commits yet. */
int32_t ixtok_stream_step(IxTokenizer* handle, IxDecodeStream* stream, uint32_t id,
                          uint8_t** out_bytes, uintptr_t* out_len, IxError* err);
/* Flushes withheld bytes; IX_DATA_LOSS on an irrecoverably incomplete
 * trailing sequence. */
int32_t ixtok_stream_finish(IxTokenizer* handle, IxDecodeStream* stream, uint8_t** out_bytes,
                            uintptr_t* out_len, IxError* err);

void ixtok_set_parallelism(int32_t enabled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TOKENIZER_BACKEND_INFERX_ABI_H_
