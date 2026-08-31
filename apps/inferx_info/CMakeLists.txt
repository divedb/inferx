# inferx-info: build/toolchain metadata smoke CLI (m0.md section 7.1).

add_executable(inferx-info main.cc)
inferx_apply_project_options(inferx-info)
target_link_libraries(inferx-info PRIVATE inferx::base)

set(_INFERX_INFO_FEATURES "base")
if(INFERX_ENABLE_CUDA)
  string(APPEND _INFERX_INFO_FEATURES ",cuda")
endif()
if(INFERX_ENABLE_ASAN)
  string(APPEND _INFERX_INFO_FEATURES ",asan")
endif()
if(INFERX_ENABLE_UBSAN)
  string(APPEND _INFERX_INFO_FEATURES ",ubsan")
endif()
if(INFERX_ENABLE_TSAN)
  string(APPEND _INFERX_INFO_FEATURES ",tsan")
endif()

target_compile_definitions(inferx-info PRIVATE
  INFERX_INFO_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  INFERX_INFO_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  INFERX_INFO_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
  INFERX_INFO_BUILD_TYPE="$<UPPER_CASE:${CMAKE_BUILD_TYPE}>"
  INFERX_INFO_FEATURES="${_INFERX_INFO_FEATURES}")

if(INFERX_ENABLE_CUDA)
  target_compile_definitions(inferx-info PRIVATE
    INFERX_INFO_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
endif()
