add_library(inferx_input STATIC
  model_package.cc
  prompt_processor.cc)
add_library(inferx::input ALIAS inferx_input)

set_target_properties(inferx_input PROPERTIES
  EXPORT_NAME input
  POSITION_INDEPENDENT_CODE ON)
target_compile_features(inferx_input PUBLIC cxx_std_23)
target_include_directories(inferx_input
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(inferx_input
  PUBLIC inferx::base
         inferx::api
         inferx::model
         inferx::tokenization
  PRIVATE $<BUILD_INTERFACE:absl::strings>
          inferx::artifacts)
inferx_apply_project_options(inferx_input)

# Deliberately not installed: the static link closure
# includes the Rust engine archive; see the tokenization target note.
