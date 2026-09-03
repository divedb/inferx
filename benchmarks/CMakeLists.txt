# Google Benchmark smoke: proves registration and JSON
# output; it claims no meaningful performance number. Not registered with
# CTest — CI's metrics job runs it explicitly with --benchmark_format=json.

add_executable(inferx_base_benchmark base_startup_benchmark.cc)
inferx_apply_project_options(inferx_base_benchmark)
target_link_libraries(inferx_base_benchmark PRIVATE inferx::base benchmark::benchmark)

if(INFERX_BUILD_OPERATOR_BENCHMARKS)
  add_executable(inferx_dispatch_benchmark dispatch/dispatch_benchmark.cc)
  inferx_apply_project_options(inferx_dispatch_benchmark)
  target_link_libraries(inferx_dispatch_benchmark PRIVATE
    inferx::ops benchmark::benchmark)
  add_custom_target(inferx_operator_benchmarks DEPENDS inferx_dispatch_benchmark)
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

if(INFERX_ENABLE_CUDA AND INFERX_BUILD_BENCHMARKS)
  add_executable(inferx_kernels_benchmark kernels/kernels_benchmark.cc)
  inferx_apply_project_options(inferx_kernels_benchmark)
  target_include_directories(inferx_kernels_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels/cuda/dispatch
    ${CMAKE_SOURCE_DIR}/kernels/cuda/include/inferx/kernels/cuda
    ${CMAKE_SOURCE_DIR}/kernels/cuda/include
  )
  target_link_libraries(inferx_kernels_benchmark PRIVATE
    inferx_kernels inferx_kernels_cuda inferx::ops inferx::tensor CUDA::cudart
    absl::status absl::statusor absl::strings absl::synchronization)
endif()
