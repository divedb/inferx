# inferx_simulator is a CPU-only behavioral oracle and is deliberately not
# installed as an SDK target (m1.md section 6).
add_library(inferx_simulator STATIC
  src/engine_simulator.cc
  src/fake_executor.cc
  src/invariant_checker.cc
  src/latency_model.cc
  src/replay.cc
  src/workload.cc)
add_library(inferx::simulator ALIAS inferx_simulator)

inferx_apply_project_options(inferx_simulator)
target_link_libraries(inferx_simulator
  PUBLIC inferx::base inferx::api inferx::config inferx::lifecycle
         inferx::scheduler inferx::engine
  PRIVATE $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::strings>
          $<BUILD_INTERFACE:simdjson::simdjson>)
target_include_directories(inferx_simulator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
