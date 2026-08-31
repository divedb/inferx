# Google Benchmark smoke (m0.md section 7.3): proves registration and JSON
# output; it claims no meaningful performance number. Not registered with
# CTest — CI's metrics job runs it explicitly with --benchmark_format=json.

add_executable(inferx_base_benchmark base_startup_benchmark.cc)
inferx_apply_project_options(inferx_base_benchmark)
target_link_libraries(inferx_base_benchmark PRIVATE inferx::base benchmark::benchmark)
