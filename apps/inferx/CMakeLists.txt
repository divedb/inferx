add_executable(inferx main.cc)
inferx_apply_project_options(inferx)
target_link_libraries(inferx PRIVATE inferx_cli inferx::command)
set_target_properties(inferx PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

install(TARGETS inferx RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
