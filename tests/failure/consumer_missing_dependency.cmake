# Fixture: consumer without a required installed dependency (m0.md 13.5 row
# 9). find_package must fail naming absl.
cmake_minimum_required(VERSION 3.28)
project(consumer_missing_dependency_fixture LANGUAGES CXX)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/empty_prefix")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/empty_prefix")
find_package(absl CONFIG REQUIRED)
