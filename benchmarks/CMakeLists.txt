# Google Benchmark smoke (m0.md section 7.3): proves registration and JSON
# output; it claims no meaningful performance number. Not registered with
# CTest — CI's metrics job runs it explicitly with --benchmark_format=json.

add_executable(inferx_base_benchmark base_startup_benchmark.cc)
inferx_apply_project_options(inferx_base_benchmark)
target_link_libraries(inferx_base_benchmark PRIVATE inferx::base benchmark::benchmark)

add_executable(inferx_memory_pool_benchmark micro/memory_pool_benchmark.cc)
inferx_apply_project_options(inferx_memory_pool_benchmark)
target_link_libraries(inferx_memory_pool_benchmark PRIVATE
  inferx::runtime benchmark::benchmark)

if(INFERX_BUILD_M4_BENCHMARKS)
  add_executable(inferx_m4_dispatch_benchmark m4/dispatch_benchmark.cc)
  inferx_apply_project_options(inferx_m4_dispatch_benchmark)
  target_link_libraries(inferx_m4_dispatch_benchmark PRIVATE
    inferx::ops benchmark::benchmark)
  add_custom_target(inferx_m4_benchmarks DEPENDS inferx_m4_dispatch_benchmark)
endif()

if(INFERX_ENABLE_CUDA)
  target_compile_definitions(inferx_memory_pool_benchmark PRIVATE
    INFERX_MEMORY_POOL_BENCHMARK_CUDA=1)
  target_link_libraries(inferx_memory_pool_benchmark PRIVATE
    inferx_cuda CUDA::cudart)
  add_executable(inferx_cuda_copy_benchmark micro/cuda_copy_benchmark.cc)
  add_executable(inferx_cuda_event_benchmark micro/cuda_event_benchmark.cc)
  add_executable(inferx_cuda_launch_benchmark micro/cuda_launch_benchmark.cu)
  foreach(_target IN ITEMS
      inferx_cuda_copy_benchmark inferx_cuda_event_benchmark
      inferx_cuda_launch_benchmark)
    inferx_apply_project_options(${_target})
    target_link_libraries(${_target} PRIVATE
      inferx_cuda benchmark::benchmark CUDA::cudart)
  endforeach()
  set_target_properties(inferx_cuda_launch_benchmark PROPERTIES
    CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
    CUDA_STANDARD_REQUIRED ON)
  add_custom_target(inferx_cuda_benchmarks DEPENDS
    inferx_cuda_copy_benchmark inferx_cuda_event_benchmark
    inferx_cuda_launch_benchmark inferx_memory_pool_benchmark)
endif()

# The allocation gate interposes the global allocation operators. TSan supplies
# its own strong definitions for those symbols, so this benchmark is intentionally
# absent from TSan builds; the channel tests remain fully instrumented.
if(NOT INFERX_ENABLE_TSAN)
  add_executable(inferx_scheduler_benchmark scheduler/scheduler_benchmark.cc)
  inferx_apply_project_options(inferx_scheduler_benchmark)
  target_link_libraries(inferx_scheduler_benchmark PRIVATE
    inferx::scheduler benchmark::benchmark absl::status absl::strings)
  set_target_properties(inferx_scheduler_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

add_executable(inferx_simulator_benchmark scheduler/simulator_benchmark.cc)
inferx_apply_project_options(inferx_simulator_benchmark)
target_link_libraries(inferx_simulator_benchmark PRIVATE
  inferx::simulator benchmark::benchmark absl::status absl::strings)
set_target_properties(inferx_simulator_benchmark PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(INFERX_ENABLE_CUDA AND INFERX_BUILD_BENCHMARKS)
  add_executable(inferx_kernels_benchmark kernels/kernels_benchmark.cc)
  inferx_apply_project_options(inferx_kernels_benchmark)
  target_include_directories(inferx_kernels_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels/cuda/dispatch
    ${CMAKE_SOURCE_DIR}/kernels/cuda/include/inferx/kernels/cuda
    ${CMAKE_SOURCE_DIR}/kernels/cuda/include
    ${CMAKE_SOURCE_DIR}/platform/cuda/include)
  target_link_libraries(inferx_kernels_benchmark PRIVATE
    inferx_kernels inferx_kernels_cuda inferx::ops inferx::tensor CUDA::cudart
    absl::status absl::statusor absl::strings absl::synchronization)
endif()
