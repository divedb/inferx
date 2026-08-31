add_executable(inferx-device-info main.cc)
inferx_apply_project_options(inferx-device-info)
target_link_libraries(inferx-device-info PRIVATE
  inferx_cuda inferx_cuda_test_kernels CUDA::cudart absl::status absl::statusor)
target_include_directories(inferx-device-info PRIVATE
  ${CMAKE_SOURCE_DIR}/kernels/cuda/test)
set_target_properties(inferx-device-info PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(BUILD_TESTING AND INFERX_BUILD_GPU_TESTS)
  add_test(NAME m2_cuda_device_info_json COMMAND inferx-device-info --json)
  set_tests_properties(m2_cuda_device_info_json PROPERTIES
    LABELS "m2-unit" TIMEOUT 120)
  add_test(NAME m2_cuda_async_integration COMMAND inferx-device-info --self-test)
  set_tests_properties(m2_cuda_async_integration PROPERTIES
    LABELS "m2-integration" TIMEOUT 300)
  add_test(NAME m2_cuda_tensor_correctness COMMAND inferx-device-info --self-test)
  set_tests_properties(m2_cuda_tensor_correctness PROPERTIES
    LABELS "m2-correctness" TIMEOUT 300)
  add_test(NAME m2_cuda_invalid_argument COMMAND inferx-device-info --invalid)
  set_tests_properties(m2_cuda_invalid_argument PROPERTIES
    LABELS "m2-failure" TIMEOUT 120)
  add_test(NAME m2_cuda_resource_stress COMMAND inferx-device-info --stress)
  set_tests_properties(m2_cuda_resource_stress PROPERTIES
    LABELS "m2-stress" TIMEOUT 1800)
endif()
