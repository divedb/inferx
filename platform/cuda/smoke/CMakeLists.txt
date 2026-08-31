# CUDA smoke (m0.md section 10): object library + CTest executable, built only
# when INFERX_ENABLE_CUDA=ON. Host sources (smoke.cc, smoke_test_main.cc) get
# InferX policy through the helper; the .cu TU gets a deliberately minimal
# NVCC-specific warning set instead of forwarded host flags.

add_library(inferx_cuda_smoke OBJECT
  smoke.cc
  smoke_kernel.cu)
inferx_apply_project_options(inferx_cuda_smoke)
# CUDA::cudart carries the CUDA include path for the host sources and the
# runtime library to consumers of the object library.
target_link_libraries(inferx_cuda_smoke PRIVATE CUDA::cudart)
target_include_directories(inferx_cuda_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(inferx_cuda_smoke PRIVATE
  INFERX_CUDA_ACCEPTED_ARCHITECTURES="${INFERX_CUDA_ACCEPTED_ARCHITECTURES}")

# Host-warning forwarding through NVCC is fragile (m0.md section 6.5): forward
# only the two basic warning families for the host pass of .cu compilation.
target_compile_options(inferx_cuda_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

add_executable(inferx_cuda_smoke_test smoke_test_main.cc)
inferx_apply_project_options(inferx_cuda_smoke_test)
target_link_libraries(inferx_cuda_smoke_test PRIVATE inferx_cuda_smoke CUDA::cudart)
target_compile_definitions(inferx_cuda_smoke_test PRIVATE
  INFERX_CUDA_ACCEPTED_ARCHITECTURES="${INFERX_CUDA_ACCEPTED_ARCHITECTURES}")

add_test(NAME cuda_smoke COMMAND inferx_cuda_smoke_test)
set_tests_properties(cuda_smoke PROPERTIES LABELS "gpu" TIMEOUT 300)
