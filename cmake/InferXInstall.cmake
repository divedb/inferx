# InferX install/export rules (m0.md section 6.7; ADR 0004: static, no ABI
# promise). Included from the top-level CMakeLists after targets are ready —
# the actual install(TARGETS) call lives with the target that owns it
# (src/base/CMakeLists.txt); this file defines the package scaffolding.

include(CMakePackageConfigHelpers)

set(INFERX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/InferX")

# Package configuration: finds dependencies of the *installed* targets only.
# In M0 `inferx::base` has no public dependencies, so nothing beyond the
# targets file is required; when M1 exposes absl::Status, add
#   find_dependency(absl CONFIG)
# here and install the pinned Abseil into the same prefix (the strategy is
# proven by tests/consumer_absl).
configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/InferXConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/InferXConfig.cmake"
  INSTALL_DESTINATION "${INFERX_INSTALL_CMAKEDIR}")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/InferXConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/InferXConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/InferXConfigVersion.cmake"
        DESTINATION "${INFERX_INSTALL_CMAKEDIR}")

# Repository-owner distribution posture (ADR 0007): the internal-use notice
# ships as package metadata until the licensing ADR is accepted.
install(FILES "${CMAKE_CURRENT_LIST_DIR}/../INTERNAL_USE_ONLY.md"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
        RENAME INTERNAL_USE_ONLY.md)
