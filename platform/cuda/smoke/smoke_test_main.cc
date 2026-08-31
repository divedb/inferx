// CTest entry for the CUDA smoke (label: gpu). Exit 77 means "no device" so
// developer machines without a GPU may skip; the owned GPU CI lane treats a
// skip as job failure (m0.md section 10.2).
#include <cstdlib>

#include "smoke.h"

#ifndef INFERX_CUDA_ACCEPTED_ARCHITECTURES
#define INFERX_CUDA_ACCEPTED_ARCHITECTURES "89"
#endif

int main() {
  switch (inferx::cuda_smoke::RunSmoke(INFERX_CUDA_ACCEPTED_ARCHITECTURES)) {
    case inferx::cuda_smoke::SmokeResult::kOk:
      return 0;
    case inferx::cuda_smoke::SmokeResult::kNoDevice:
      return 77;
    case inferx::cuda_smoke::SmokeResult::kFailed:
      return 1;
  }
  return 1;
}
