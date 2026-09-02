# Fixture: ASan and TSan enabled together. Configure must
# fail and identify the incompatible options.
cmake_minimum_required(VERSION 3.28)
project(asan_tsan_fixture LANGUAGES CXX)

set(INFERX_ENABLE_ASAN ON)
set(INFERX_ENABLE_TSAN ON)
include("${INFERX_ROOT}/cmake/InferXSanitizers.cmake")
