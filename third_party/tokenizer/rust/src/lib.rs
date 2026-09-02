//! Owned InferX tokenizer FFI shim (ADR 0024).
//!
//! A thin, error-returning C ABI over the official Hugging Face Rust
//! `tokenizers` engine. Contract, enforced by every entry point:
//!
//! * no `unwrap()`, `expect()`, or panic reaches the FFI: every entry is
//!   wrapped in `catch_unwind` and a panic becomes `IX_PANIC` plus an owned
//!   error message;
//! * every fallible result is returned as a status code plus an owned
//!   `IxError` buffer the caller frees with [`ixtok_free_error`];
//! * every byte or id buffer returned to the caller is owned by the caller
//!   and freed explicitly (`ixtok_free_bytes`, `ixtok_free_ids`) -- no
//!   returned pointer aliases handle-owned scratch;
//! * streaming decode state is externalized: an [`IxDecodeStream`] holds
//!   exactly the state variables of the upstream `step_decode_stream`
//!   free function and borrows a tokenizer only for the duration of one
//!   step call, so any pool instance can serve any decoder;
//! * input text is validated as UTF-8 here and rejected with
//!   `IX_INVALID_ARGUMENT` rather than unwrapped.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::str::{from_utf8, FromStr};

use tokenizers::tokenizer::step_decode_stream;
use tokenizers::Tokenizer;

/// Status codes shared with `src/backend/inferx_abi.h`.
pub const IX_OK: i32 = 0;
/// Stream step succeeded but no chunk is committed yet (split code point).
pub const IX_PENDING: i32 = 1;
/// Token/id lookup missed; not an error for probing calls.
pub const IX_NOT_FOUND: i32 = 2;
pub const IX_INVALID_ARGUMENT: i32 = 3;
pub const IX_DATA_LOSS: i32 = 4;
pub const IX_INTERNAL: i32 = 5;
/// The engine panicked; an engine or shim bug, never a data error.
pub const IX_PANIC: i32 = 6;

/// Owned error record. `message` is heap bytes the caller frees with
/// [`ixtok_free_error`]; a null message with code `IX_OK` marks success.
#[repr(C)]
pub struct IxError {
    pub code: i32,
    pub message: *mut u8,
    pub message_len: usize,
}

impl IxError {
    fn ok() -> Self {
        IxError { code: IX_OK, message: std::ptr::null_mut(), message_len: 0 }
    }
}

/// The tokenizer handle. Not thread-safe: the C++ layer gives each handle
/// exactly one owner, matching the engine's `&mut self` surface.
pub struct IxTokenizer {
    tokenizer: Tokenizer,
}

/// Externalized upstream streaming-decode state: exactly the state of
/// `step_decode_stream` (pending ids, committed prefix, prefix index) plus
/// the fixed decode options. No tokenizer pointer is stored; each step
/// borrows the caller's handle for one call only.
pub struct IxDecodeStream {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    skip_special_tokens: bool,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn set_error(err: *mut IxError, code: i32, message: &str) {
    if err.is_null() {
        return;
    }
    let owned = message.as_bytes().to_vec().into_boxed_slice();
    let len = owned.len();
    let ptr = Box::into_raw(owned) as *mut u8;
    unsafe {
        (*err).code = code;
        (*err).message = ptr;
        (*err).message_len = len;
    }
}

/// Writes the error record and the `Err(code)` that unwinds `guarded` to a
/// failure; the entry point returns the record's code.
fn fail<T>(err: *mut IxError, code: i32, message: &str) -> Result<T, i32> {
    set_error(err, code, message);
    Err(code)
}

fn into_raw_bytes(bytes: Box<[u8]>) -> (*mut u8, usize) {
    let len = bytes.len();
    (Box::into_raw(bytes) as *mut u8, len)
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Runs `f` under `catch_unwind`. On success `err` is reset; on failure the
/// record holds the status written by `f` (or `IX_PANIC`).
fn guarded<T>(err: *mut IxError, what: &str, f: impl FnOnce() -> Result<T, i32>) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => {
            if !err.is_null() {
                unsafe { *err = IxError::ok() };
            }
            Some(value)
        }
        Ok(Err(code)) => {
            debug_assert_ne!(code, IX_OK);
            None
        }
        Err(panic) => {
            let detail = panic_message(&panic);
            set_error(err, IX_PANIC, &format!("{what} panicked: {detail}"));
            None
        }
    }
}

/// The i32 an entry point returns after `guarded` failed: the error record's
/// code, or `IX_INTERNAL` when no record was written (a shim bug).
fn exit_code(err: *mut IxError) -> i32 {
    if err.is_null() {
        return IX_INTERNAL;
    }
    let code = unsafe { (*err).code };
    if code == IX_OK {
        IX_INTERNAL
    } else {
        code
    }
}

macro_rules! abi {
    ($err:ident, $what:literal, $body:expr) => {
        match guarded($err, $what, $body) {
            Some(_) => IX_OK,
            None => exit_code($err),
        }
    };
}

// ---------------------------------------------------------------------------
// Construction and teardown
// ---------------------------------------------------------------------------

/// Parses a serialized `tokenizer.json`. Returns null and an owned error on
/// malformed input; never aborts.
#[no_mangle]
pub extern "C" fn ixtok_new(
    json_ptr: *const u8,
    json_len: usize,
    err: *mut IxError,
) -> *mut IxTokenizer {
    match guarded(err, "tokenizer construction", || {
        if json_ptr.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "tokenizer.json buffer is null");
        }
        let bytes = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
        let text = match from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => {
                return fail(err, IX_DATA_LOSS, "tokenizer.json is not valid UTF-8");
            }
        };
        match Tokenizer::from_str(text) {
            Ok(tokenizer) => Ok(Box::into_raw(Box::new(IxTokenizer { tokenizer }))),
            Err(parse) => Err({
                set_error(
                    err,
                    IX_DATA_LOSS,
                    &format!("tokenizer.json cannot be parsed: {parse}"),
                );
                IX_DATA_LOSS
            }),
        }
    }) {
        Some(handle) => handle,
        None => std::ptr::null_mut(),
    }
}

/// Frees a tokenizer handle. Null is accepted.
#[no_mangle]
pub extern "C" fn ixtok_free(handle: *mut IxTokenizer) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Frees an owned error message and resets the record to success.
#[no_mangle]
pub extern "C" fn ixtok_free_error(err: *mut IxError) {
    if err.is_null() {
        return;
    }
    unsafe {
        let record = std::ptr::read(err);
        if !record.message.is_null() {
            drop(Box::from_raw(
                std::slice::from_raw_parts_mut(record.message, record.message_len),
            ));
        }
        *err = IxError::ok();
    }
}

/// Frees an owned byte buffer returned by this ABI.
#[no_mangle]
pub extern "C" fn ixtok_free_bytes(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(std::slice::from_raw_parts_mut(ptr, len)) });
    }
}

/// Frees an owned id buffer returned by this ABI.
#[no_mangle]
pub extern "C" fn ixtok_free_ids(ptr: *mut u32, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(std::slice::from_raw_parts_mut(ptr, len)) });
    }
}

/// Frees an owned u64 offsets buffer returned by this ABI.
#[no_mangle]
pub extern "C" fn ixtok_free_offsets(ptr: *mut u64, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(std::slice::from_raw_parts_mut(ptr, len)) });
    }
}

// ---------------------------------------------------------------------------
// Encode and decode
// ---------------------------------------------------------------------------

/// Encodes UTF-8 text into owned ids. Invalid UTF-8 is `IX_INVALID_ARGUMENT`.
#[no_mangle]
pub extern "C" fn ixtok_encode(
    handle: *mut IxTokenizer,
    text_ptr: *const u8,
    text_len: usize,
    add_special_tokens: i32,
    out_ids: *mut *mut u32,
    out_len: *mut usize,
    err: *mut IxError,
) -> i32 {
    abi!(err, "encode", || {
        if handle.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "tokenizer handle is null");
        }
        if text_ptr.is_null() && text_len != 0 {
            return fail(err, IX_INVALID_ARGUMENT, "text buffer is null");
        }
        if out_ids.is_null() || out_len.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "encode output slots are null");
        }
        let bytes = unsafe { std::slice::from_raw_parts(text_ptr, text_len) };
        let text = match from_utf8(bytes) {
            Ok(text) => text,
            Err(_) => {
                return fail(
                    err,
                    IX_INVALID_ARGUMENT,
                    "input is not valid UTF-8; the tokenizer engine cannot accept it",
                );
            }
        };
        let tokenizer = &unsafe { &*handle }.tokenizer;
        match tokenizer.encode(text, add_special_tokens != 0) {
            Ok(encoding) => {
                let ids = encoding.get_ids().to_vec().into_boxed_slice();
                let len = ids.len();
                unsafe {
                    *out_ids = Box::into_raw(ids) as *mut u32;
                    *out_len = len;
                }
                Ok(())
            }
            Err(e) => fail(err, IX_INTERNAL, &format!("encode failed: {e}")),
        }
    })
}

/// Decodes ids into an owned UTF-8 buffer.
#[no_mangle]
pub extern "C" fn ixtok_decode(
    handle: *mut IxTokenizer,
    ids_ptr: *const u32,
    ids_len: usize,
    skip_special_tokens: i32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    err: *mut IxError,
) -> i32 {
    abi!(err, "decode", || {
        if handle.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "tokenizer handle is null");
        }
        if out_ptr.is_null() || out_len.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "decode output slots are null");
        }
        let ids = unsafe { std::slice::from_raw_parts(ids_ptr, ids_len) };
        let tokenizer = &unsafe { &*handle }.tokenizer;
        match tokenizer.decode(ids, skip_special_tokens != 0) {
            Ok(text) => {
                let (ptr, len) = into_raw_bytes(text.into_bytes().into_boxed_slice());
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                }
                Ok(())
            }
            Err(e) => fail(err, IX_INTERNAL, &format!("decode failed: {e}")),
        }
    })
}

// ---------------------------------------------------------------------------
// Vocabulary and metadata
// ---------------------------------------------------------------------------

/// Vocabulary size including added tokens.
#[no_mangle]
pub extern "C" fn ixtok_vocab_size(handle: *mut IxTokenizer) -> usize {
    if handle.is_null() {
        return 0;
    }
    let tokenizer = &unsafe { &*handle }.tokenizer;
    tokenizer.get_vocab_size(true)
}

/// Resolves one id to its token string into an owned buffer.
/// `IX_NOT_FOUND` when the id is not in the vocabulary.
#[no_mangle]
pub extern "C" fn ixtok_id_to_token(
    handle: *mut IxTokenizer,
    id: u32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    err: *mut IxError,
) -> i32 {
    abi!(err, "id_to_token", || {
        if handle.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "tokenizer handle is null");
        }
        if out_ptr.is_null() || out_len.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "id_to_token output slots are null");
        }
        let tokenizer = &unsafe { &*handle }.tokenizer;
        match tokenizer.id_to_token(id) {
            Some(token) => {
                let (ptr, len) = into_raw_bytes(token.into_bytes().into_boxed_slice());
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                }
                Ok(())
            }
            None => fail(err, IX_NOT_FOUND, "id is not in the vocabulary"),
        }
    })
}

/// Resolves one token string to its id. `IX_NOT_FOUND` when absent.
#[no_mangle]
pub extern "C" fn ixtok_token_to_id(
    handle: *mut IxTokenizer,
    token_ptr: *const u8,
    token_len: usize,
    out_id: *mut u32,
    err: *mut IxError,
) -> i32 {
    abi!(err, "token_to_id", || {
        if handle.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "tokenizer handle is null");
        }
        if out_id.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "token_to_id output slot is null");
        }
        let bytes = unsafe { std::slice::from_raw_parts(token_ptr, token_len) };
        let token = match from_utf8(bytes) {
            Ok(token) => token,
            Err(_) => {
                return fail(err, IX_INVALID_ARGUMENT, "token is not valid UTF-8");
            }
        };
        let tokenizer = &unsafe { &*handle }.tokenizer;
        match tokenizer.token_to_id(token) {
            Some(id) => {
                unsafe { *out_id = id };
                Ok(())
            }
            None => fail(err, IX_NOT_FOUND, "token is not in the vocabulary"),
        }
    })
}

/// Dumps the full vocabulary in one crossing: `count` ids in `out_ids`, the
/// packed token strings in `out_chars` (`*out_chars_len` bytes), delimited by
/// `out_offsets` with `count + 1` entries. Everything is caller-owned, so
/// metadata construction does not cross the FFI once per id.
#[no_mangle]
pub extern "C" fn ixtok_vocab_dump(
    handle: *mut IxTokenizer,
    out_ids: *mut *mut u32,
    out_offsets: *mut *mut u64,
    out_chars: *mut *mut u8,
    out_chars_len: *mut usize,
    out_count: *mut usize,
    err: *mut IxError,
) -> i32 {
    abi!(err, "vocab_dump", || {
        if handle.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "tokenizer handle is null");
        }
        if out_ids.is_null()
            || out_offsets.is_null()
            || out_chars.is_null()
            || out_chars_len.is_null()
            || out_count.is_null()
        {
            return fail(err, IX_INVALID_ARGUMENT, "vocab dump output slots are null");
        }
        let tokenizer = &unsafe { &*handle }.tokenizer;
        let mut vocab: Vec<(u32, String)> = tokenizer
            .get_vocab(true)
            .into_iter()
            .map(|(token, id)| (id, token))
            .collect();
        vocab.sort_unstable_by_key(|(id, _)| *id);

        let count = vocab.len();
        let mut ids = Vec::with_capacity(count);
        let mut offsets = Vec::with_capacity(count + 1);
        let mut chars = Vec::new();
        offsets.push(0u64);
        for (id, token) in &vocab {
            ids.push(*id);
            chars.extend_from_slice(token.as_bytes());
            offsets.push(chars.len() as u64);
        }
        let chars_len = chars.len();

        unsafe {
            *out_ids = Box::into_raw(ids.into_boxed_slice()) as *mut u32;
            *out_offsets = Box::into_raw(offsets.into_boxed_slice()) as *mut u64;
            let (chars_ptr, _) = into_raw_bytes(chars.into_boxed_slice());
            *out_chars = chars_ptr;
            *out_chars_len = chars_len;
            *out_count = count;
        }
        Ok(())
    })
}

/// Dumps the ids the engine's added vocabulary marks special into an owned
/// id buffer, so callers can reproduce `skip_special_tokens` decisions.
#[no_mangle]
pub extern "C" fn ixtok_special_ids(
    handle: *mut IxTokenizer,
    out_ids: *mut *mut u32,
    out_len: *mut usize,
    err: *mut IxError,
) -> i32 {
    abi!(err, "special_ids", || {
        if handle.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "tokenizer handle is null");
        }
        if out_ids.is_null() || out_len.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "special ids output slots are null");
        }
        let tokenizer = &unsafe { &*handle }.tokenizer;
        let mut ids: Vec<u32> = tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, token)| token.special)
            .map(|(id, _)| *id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let len = ids.len();
        unsafe {
            *out_ids = Box::into_raw(ids.into_boxed_slice()) as *mut u32;
            *out_len = len;
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Streaming decode (externalized upstream state)
// ---------------------------------------------------------------------------

/// Creates stream state. `skip_special_tokens` is fixed for the stream
/// lifetime, matching `Tokenizer::decode_stream`.
#[no_mangle]
pub extern "C" fn ixtok_stream_new(skip_special_tokens: i32) -> *mut IxDecodeStream {
    Box::into_raw(Box::new(IxDecodeStream {
        ids: Vec::new(),
        prefix: String::new(),
        prefix_index: 0,
        skip_special_tokens: skip_special_tokens != 0,
    }))
}

/// Frees stream state. Null is accepted. A stream never owns a tokenizer.
#[no_mangle]
pub extern "C" fn ixtok_stream_free(stream: *mut IxDecodeStream) {
    if !stream.is_null() {
        drop(unsafe { Box::from_raw(stream) });
    }
}

/// Steps the stream by one id using the upstream `step_decode_stream`
/// algorithm. Returns `IX_OK` with an owned chunk, or `IX_PENDING` when the
/// id alone does not commit a chunk (split code point or byte-fallback
/// sequence).
#[no_mangle]
pub extern "C" fn ixtok_stream_step(
    handle: *mut IxTokenizer,
    stream: *mut IxDecodeStream,
    id: u32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    err: *mut IxError,
) -> i32 {
    abi!(err, "stream_step", || {
        if handle.is_null() || stream.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "stream step needs a live handle and stream");
        }
        if out_ptr.is_null() || out_len.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "stream step output slots are null");
        }
        let tokenizer = &unsafe { &*handle }.tokenizer;
        let state = unsafe { &mut *stream };
        match step_decode_stream(
            tokenizer,
            id,
            state.skip_special_tokens,
            &mut state.ids,
            &mut state.prefix,
            &mut state.prefix_index,
        ) {
            Ok(Some(chunk)) => {
                let (ptr, len) = into_raw_bytes(chunk.into_bytes().into_boxed_slice());
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                }
                Ok(())
            }
            Ok(None) => {
                set_error(err, IX_PENDING, "no chunk committed yet");
                Err(IX_PENDING)
            }
            Err(e) => fail(err, IX_INTERNAL, &format!("stream step failed: {e}")),
        }
    })
}

/// Flushes a stream: emits any bytes the engine is still withholding.
/// Returns `IX_OK` with an owned final chunk (possibly empty), or
/// `IX_DATA_LOSS` when the buffered ids decode to an irrecoverably
/// incomplete sequence.
#[no_mangle]
pub extern "C" fn ixtok_stream_finish(
    handle: *mut IxTokenizer,
    stream: *mut IxDecodeStream,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    err: *mut IxError,
) -> i32 {
    abi!(err, "stream_finish", || {
        if handle.is_null() || stream.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "stream finish needs a live handle and stream");
        }
        if out_ptr.is_null() || out_len.is_null() {
            return fail(err, IX_INVALID_ARGUMENT, "stream finish output slots are null");
        }
        let tokenizer = &unsafe { &*handle }.tokenizer;
        let state = unsafe { &mut *stream };
        let whole = match tokenizer.decode(&state.ids, state.skip_special_tokens) {
            Ok(text) => text,
            Err(e) => {
                return fail(err, IX_INTERNAL, &format!("stream finish decode failed: {e}"));
            }
        };
        if whole.ends_with('\u{FFFD}') {
            return fail(
                err,
                IX_DATA_LOSS,
                "stream ends with an incomplete byte sequence; no valid text remains to flush",
            );
        }
        let rest = match whole.strip_prefix(state.prefix.as_str()) {
            Some(rest) => rest.to_string(),
            None => {
                return fail(
                    err,
                    IX_DATA_LOSS,
                    "stream prefix no longer matches the buffered ids",
                );
            }
        };
        let (ptr, len) = into_raw_bytes(rest.into_bytes().into_boxed_slice());
        unsafe {
            *out_ptr = ptr;
            *out_len = len;
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Engine configuration
// ---------------------------------------------------------------------------

/// Configures engine-internal parallelism once. InferX tokenization workers
/// own exclusive handles and the serving path encodes single strings, so the
/// engine's internal Rayon pool must not oversubscribe the workers.
#[no_mangle]
pub extern "C" fn ixtok_set_parallelism(enabled: i32) {
    tokenizers::utils::parallelism::set_parallelism(enabled != 0);
}
