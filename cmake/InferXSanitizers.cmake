# InferX sanitizer configuration (ADR 0005/0006: CPU-only lanes in M0).
#
# Builds the target-scoped `inferx_sanitizers` INTERFACE library when any
# sanitizer is enabled, after rejecting invalid combinations at configure time.

if(INFERX_ENABLE_ASAN AND INFERX_ENABLE_TSAN)
  message(FATAL_ERROR
    "Incompatible sanitizer options: INFERX_ENABLE_ASAN and INFERX_ENABLE_TSAN "
    "are mutually exclusive. Configure one lane at a time (see CMakePresets.json: "
    "'asan-ubsan' or 'tsan').")
endif()

if((INFERX_ENABLE_ASAN OR INFERX_ENABLE_UBSAN OR INFERX_ENABLE_TSAN) AND INFERX_ENABLE_CUDA)
  message(FATAL_ERROR
    "Incompatible options: CPU sanitizers (ASan/UBSan/TSan) cannot be combined "
    "with INFERX_ENABLE_CUDA in M0; this would produce a partially sanitized "
    "binary. Use the 'cuda-release' preset for CUDA and a sanitizer preset for "
    "CPU code.")
endif()

set(_INFERX_SANITIZER_FLAGS "")
if(INFERX_ENABLE_ASAN)
  list(APPEND _INFERX_SANITIZER_FLAGS
       -fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope)
endif()
if(INFERX_ENABLE_UBSAN)
  list(APPEND _INFERX_SANITIZER_FLAGS
       -fsanitize=undefined -fno-sanitize-recover=undefined)
endif()
if(INFERX_ENABLE_TSAN)
  list(APPEND _INFERX_SANITIZER_FLAGS
       -fsanitize=thread -fno-omit-frame-pointer)
endif()

if(_INFERX_SANITIZER_FLAGS)
  add_library(inferx_sanitizers INTERFACE)
  add_library(inferx::sanitizers ALIAS inferx_sanitizers)
  # Sanitizers are CPU-only in M0; compile flags apply to CXX sources only.
  target_compile_options(inferx_sanitizers INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${_INFERX_SANITIZER_FLAGS}>)
  target_link_options(inferx_sanitizers INTERFACE ${_INFERX_SANITIZER_FLAGS})
endif()
