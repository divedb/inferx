# InferX install/export rules (ADR 0004: static, no ABI
# promise). Included from the top-level CMakeLists after targets are ready —
# the actual install(TARGETS) call lives with the target that owns it
# (src/base/CMakeLists.txt); this file defines the package scaffolding.

include(CMakePackageConfigHelpers)

set(INFERX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/InferX")

# Package configuration: finds dependencies of the *installed* targets only.
# `inferx::base` has public Abseil dependencies, so the package configuration
# resolves them before loading the targets file.
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
install(FILES
        "${CMAKE_CURRENT_LIST_DIR}/../third_party/notices/divedb-tokenizer-MIT.txt"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}/third-party-notices")
if(INFERX_ENABLE_HF_HUB AND INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
  install(FILES "${CMAKE_CURRENT_LIST_DIR}/../third_party/curl/COPYING"
          DESTINATION "${CMAKE_INSTALL_DOCDIR}/third-party-notices"
          RENAME curl-COPYING)
endif()
if(INFERX_BUILD_CLI AND INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
  install(FILES "${CMAKE_CURRENT_LIST_DIR}/../third_party/CLI11/LICENSE"
          DESTINATION "${CMAKE_INSTALL_DOCDIR}/third-party-notices"
          RENAME CLI11-BSD-3-Clause.txt)
endif()
