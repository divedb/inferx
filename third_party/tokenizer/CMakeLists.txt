# InferX tokenizer vendor package (ADR 0024).
#
# An owned, local-only adaptation of divedb/tokenizer (MIT) over an owned
# error-returning Rust FFI shim on the Hugging Face `tokenizers` engine.
# Composition rules enforced here:
#
#   * parent-provided Abseil only -- no dependency is added or force-set;
#   * no Hub/cache/curl/OpenSSL/SentencePiece anywhere in the target;
#   * the Rust crate builds offline from the vendored crates.io closure;
#   * nlohmann/json and minja (chat templates) are pinned submodules of the
#     repository root and stay private to this target's sources.

set(_tokenizer_root "${CMAKE_CURRENT_LIST_DIR}")

# --- Rust static library ----------------------------------------------------

set(_tokenizer_rust_dir "${_tokenizer_root}/rust")
set(_tokenizer_rust_stamp "${CMAKE_CURRENT_BINARY_DIR}/tokenizer_cargo.stamp")
set(_tokenizer_rust_lib
    "${_tokenizer_rust_dir}/target/release/libinferx_tokenizers_c.a")

add_custom_command(
  OUTPUT "${_tokenizer_rust_stamp}"
  COMMAND "${INFERX_CARGO_EXECUTABLE}" build --release --offline
          --target-dir "${_tokenizer_rust_dir}/target"
  COMMAND "${CMAKE_COMMAND}" -E touch "${_tokenizer_rust_stamp}"
  WORKING_DIRECTORY "${_tokenizer_rust_dir}"
  DEPENDS "${_tokenizer_rust_dir}/src/lib.rs"
          "${_tokenizer_rust_dir}/Cargo.toml"
          "${_tokenizer_rust_dir}/Cargo.lock"
  COMMENT "Building the InferX tokenizer engine (offline cargo build)"
  VERBATIM)

add_custom_target(inferx_tokenizer_cargo DEPENDS "${_tokenizer_rust_stamp}")

add_library(inferx_tokenizer_c STATIC IMPORTED GLOBAL)
set_target_properties(
  inferx_tokenizer_c PROPERTIES
  IMPORTED_LOCATION "${_tokenizer_rust_lib}"
  IMPORTED_NO_SONAME TRUE)
target_link_libraries(inferx_tokenizer_c INTERFACE Threads::Threads
                                              ${CMAKE_DL_LIBS})
add_dependencies(inferx_tokenizer_c inferx_tokenizer_cargo)

# --- C++ wrapper library ----------------------------------------------------

set(TOKENIZER_SOURCES
    src/adapters/default_adapter.cc
    src/adapters/llama_family_adapter.cc
    src/adapters/registry.cc
    src/backend/rust_backend.cc
    src/chat/chat_message_json.cc
    src/chat/chat_template.cc
    src/chat/template_source.cc
    src/json_bridge.cc
    src/json_value.cc
    src/pretrained_tokenizer.cc
    src/special_tokens.cc
    src/tokenizer_config.cc)

add_library(inferx_tokenizer_vendor STATIC ${TOKENIZER_SOURCES})
add_dependencies(inferx_tokenizer_vendor inferx_tokenizer_cargo)

target_include_directories(
  inferx_tokenizer_vendor
  PUBLIC $<BUILD_INTERFACE:${_tokenizer_root}/include>
         $<BUILD_INTERFACE:${_tokenizer_root}/src/backend>)
target_include_directories(
  inferx_tokenizer_vendor SYSTEM PUBLIC
  $<BUILD_INTERFACE:${INFERX_THIRD_PARTY_DIR}/nlohmann-json/single_include>
  $<BUILD_INTERFACE:${INFERX_THIRD_PARTY_DIR}/minja/include>)
target_link_libraries(
  inferx_tokenizer_vendor
  PUBLIC absl::core_headers
         $<BUILD_INTERFACE:absl::status>
         $<BUILD_INTERFACE:absl::statusor>
         $<BUILD_INTERFACE:absl::status_builder>
  PRIVATE inferx_tokenizer_c
          absl::flat_hash_map
          absl::str_format
          absl::strings
          absl::synchronization
          absl::time)

# The vendored sources are adapted third-party code: they are not subject to
# the project's warning/tidy policy. Consumers apply their own policy; this
# target must build clean with only the default toolchain flags.
