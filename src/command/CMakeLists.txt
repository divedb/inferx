add_library(inferx_command STATIC dispatcher.cc)
add_library(inferx::command ALIAS inferx_command)

if(INFERX_ENABLE_CUDA)
  target_sources(inferx_command PRIVATE
    ${PROJECT_SOURCE_DIR}/platform/cuda/diagnostics/cuda_environment.cc)
else()
  target_sources(inferx_command PRIVATE cuda_environment_stub.cc)
endif()

inferx_apply_project_options(inferx_command)
target_include_directories(inferx_command PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(inferx_command
  PUBLIC inferx::base
  PRIVATE inferx::artifacts inferx::config inferx::engine inferx::execution_runtime
          inferx::input inferx::lifecycle
          absl::status absl::statusor)

set(_INFERX_COMMAND_FEATURES "cpu")
if(INFERX_ENABLE_CUDA)
  string(APPEND _INFERX_COMMAND_FEATURES ",cuda")
  target_link_libraries(inferx_command PRIVATE CUDA::cudart)
  target_compile_definitions(inferx_command PRIVATE
    INFERX_COMMAND_ENABLE_CUDA=1
    INFERX_COMMAND_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
else()
  target_compile_definitions(inferx_command PRIVATE INFERX_COMMAND_ENABLE_CUDA=0)
endif()
if(INFERX_ENABLE_ASAN)
  string(APPEND _INFERX_COMMAND_FEATURES ",asan")
endif()
if(INFERX_ENABLE_UBSAN)
  string(APPEND _INFERX_COMMAND_FEATURES ",ubsan")
endif()
if(INFERX_ENABLE_TSAN)
  string(APPEND _INFERX_COMMAND_FEATURES ",tsan")
endif()

set(INFERX_GIT_REVISION "auto" CACHE STRING
    "Git revision embedded in inferx version output (auto, unknown, or an explicit revision).")
set(_INFERX_GIT_REVISION "${INFERX_GIT_REVISION}")
if(_INFERX_GIT_REVISION STREQUAL "auto")
  set(_INFERX_GIT_RESULT 1)
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RESULT_VARIABLE _INFERX_GIT_RESULT
      OUTPUT_VARIABLE _INFERX_GIT_REVISION
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(NOT Git_FOUND OR NOT _INFERX_GIT_RESULT EQUAL 0 OR NOT _INFERX_GIT_REVISION)
    set(_INFERX_GIT_REVISION "unknown")
  endif()
endif()
target_compile_definitions(inferx_command PRIVATE
  INFERX_COMMAND_GIT_REVISION="${_INFERX_GIT_REVISION}"
  INFERX_COMMAND_CXX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  INFERX_COMMAND_CXX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  INFERX_COMMAND_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
  INFERX_COMMAND_BUILD_TYPE="$<UPPER_CASE:${CMAKE_BUILD_TYPE}>"
  INFERX_COMMAND_FEATURES="${_INFERX_COMMAND_FEATURES}")
