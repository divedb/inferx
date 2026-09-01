#ifndef INFERX_PLATFORM_CUDA_OPS_CUBLASLT_GEMM_H_
#define INFERX_PLATFORM_CUDA_OPS_CUBLASLT_GEMM_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/ops/gemm.h"
#include "inferx/ops/kernel_key.h"
#include "inferx/ops/logits.h"
#include "inferx/platform/cuda/cuda_error.h"
#include "inferx/platform/cuda/ops/cuda_op_context.h"
#include "inferx/tensor/buffer.h"

namespace inferx::cuda::ops {

class CublasLtGemmPlan {
 public:
  ~CublasLtGemmPlan() noexcept;
  CublasLtGemmPlan(CublasLtGemmPlan&&) noexcept;
  CublasLtGemmPlan& operator=(CublasLtGemmPlan&&) noexcept;
  CublasLtGemmPlan(const CublasLtGemmPlan&) = delete;
  CublasLtGemmPlan& operator=(const CublasLtGemmPlan&) = delete;

  [[nodiscard]] ByteCount workspace_bytes() const noexcept;
  [[nodiscard]] absl::Status Launch(const inferx::ops::GemmRequest& request,
                                    const CudaOpContext& context,
                                    MutableBufferView workspace) const;
  [[nodiscard]] absl::Status Launch(const inferx::ops::LogitsRequest& request,
                                    const CudaOpContext& context,
                                    MutableBufferView workspace) const;

 private:
  friend class CublasLtContext;
  struct Impl;
  explicit CublasLtGemmPlan(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

class CublasLtContext {
 public:
  [[nodiscard]] static absl::StatusOr<CublasLtContext> Create(DeviceId device,
                                                              CudaHealth* health = nullptr);
  ~CublasLtContext() noexcept;
  CublasLtContext(CublasLtContext&&) noexcept;
  CublasLtContext& operator=(CublasLtContext&&) noexcept;
  CublasLtContext(const CublasLtContext&) = delete;
  CublasLtContext& operator=(const CublasLtContext&) = delete;

  [[nodiscard]] absl::StatusOr<CublasLtGemmPlan> Prepare(const inferx::ops::KernelKey& key) const;
  [[nodiscard]] absl::Status Close();

 private:
  struct Impl;
  explicit CublasLtContext(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::cuda::ops

#endif  // INFERX_PLATFORM_CUDA_OPS_CUBLASLT_GEMM_H_
