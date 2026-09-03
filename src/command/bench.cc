#include "common.h"

namespace inferx::command {

void Benchmark(const BenchmarkOptions&) {
  internal::Unavailable("benchmark", "the unified benchmark runner is not compiled in this milestone");
}

}  // namespace inferx::command
