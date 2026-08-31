# Fixture: analysis requested but the tool is absent (m0.md 13.5 row 8).
# Must fail naming the required tool and version instead of silently skipping
# analysis.
cmake_minimum_required(VERSION 3.28)
project(analysis_tool_absent_fixture LANGUAGES CXX)

set(INFERX_ENABLE_CLANG_TIDY ON)
set(INFERX_CLANG_TIDY_BIN "/nonexistent/bin/clang-tidy")
include("${INFERX_ROOT}/cmake/InferXProjectOptions.cmake")
