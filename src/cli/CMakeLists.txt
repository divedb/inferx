add_library(inferx_cli STATIC app.cc)

inferx_apply_project_options(inferx_cli)
target_include_directories(inferx_cli PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>)
target_link_libraries(inferx_cli PUBLIC inferx::base PRIVATE CLI11::CLI11)
