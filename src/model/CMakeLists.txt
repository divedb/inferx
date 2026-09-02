add_library(inferx_model STATIC
  architecture_registry.cc
  llama_for_causal_lm.cc
  llama_model_factory.cc
  model_package.cc
  model_spec.cc
  model_weights.cc
  weight_planner.cc)
add_library(inferx::model ALIAS inferx_model)

set_target_properties(inferx_model PROPERTIES
  EXPORT_NAME model
  POSITION_INDEPENDENT_CODE ON)
target_compile_features(inferx_model PUBLIC cxx_std_23)
target_include_directories(inferx_model
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(inferx_model
  PUBLIC inferx::base inferx::artifacts inferx::ops inferx::scheduler
         $<BUILD_INTERFACE:absl::status>
         $<BUILD_INTERFACE:absl::statusor>
  PRIVATE $<BUILD_INTERFACE:absl::strings>
          simdjson::simdjson)
inferx_apply_project_options(inferx_model)

install(TARGETS inferx_model
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/model"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx"
        FILES_MATCHING PATTERN "*.h")
