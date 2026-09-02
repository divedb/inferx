# Fixture: unsupported/too-old CMake. Must fail before
# compiling project sources, naming the required minimum.
cmake_minimum_required(VERSION 3.85)
project(cmake_too_old_fixture LANGUAGES CXX)
