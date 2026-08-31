# inferx_base: the minimal installed static base library (m0.md section 6.4).

add_library(inferx_base STATIC version.cc)
add_library(inferx::base ALIAS inferx_base)

set_target_properties(inferx_base PROPERTIES
  EXPORT_NAME base
  POSITION_INDEPENDENT_CODE ON)

target_compile_features(inferx_base PUBLIC cxx_std_23)

target_include_directories(inferx_base
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(inferx_base PRIVATE
  INFERX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  INFERX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  INFERX_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  INFERX_VERSION_STRING="${PROJECT_VERSION}")

# Build-only policy (warnings, sanitizer flags) attaches through the helper;
# PRIVATE linking keeps these out of the installed interface.
inferx_apply_project_options(inferx_base)

install(TARGETS inferx_base
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.h")

install(EXPORT InferXTargets
        NAMESPACE inferx::
        DESTINATION "${INFERX_INSTALL_CMAKEDIR}")
