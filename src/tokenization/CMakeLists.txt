add_library(inferx_tokenization STATIC
  qualified_tokenizer.cc
  tokenizer_pool.cc)
add_library(inferx::tokenization ALIAS inferx_tokenization)

set_target_properties(inferx_tokenization PROPERTIES
  EXPORT_NAME tokenization
  POSITION_INDEPENDENT_CODE ON)
target_compile_features(inferx_tokenization PUBLIC cxx_std_23)
target_include_directories(inferx_tokenization
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(inferx_tokenization
  PUBLIC inferx::base
         $<BUILD_INTERFACE:absl::status>
         $<BUILD_INTERFACE:absl::statusor>
  PRIVATE $<BUILD_INTERFACE:absl::strings>
          inferx_tokenizer_vendor)
inferx_apply_project_options(inferx_tokenization)
