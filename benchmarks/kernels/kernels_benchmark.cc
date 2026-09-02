// InferX kernels-layer benchmark (ADR 0031/0032): CUDA-event timing over
// the shared workload matrix (tools/bench/workloads.py), emitting JSON the
// plotting pipeline consumes. One launch per iteration through the unified
// dispatch surface — the provider chain runs exactly as production would.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cuda_kernel_backend.h"
#include "inferx/kernels/kernel_dispatch.h"
#include "inferx/kernels/ops/activation.h"
#include "inferx/kernels/ops/layernorm.h"
#include "inferx/kernels/ops/sampling.h"
#include "inferx/kernels/ops/attention.h"
#include "inferx/kernels/ops/embedding.h"
#include "inferx/kernels/ops/gemm.h"
#include "inferx/kernels/ops/transform.h"
#include "inferx/kernels/ops/model_fused.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace {

class BenchDomain final : public inferx::AllocationDomain {
 public:
  absl::Status Release(void* address, inferx::ByteCount, inferx::ByteCount,
                       inferx::AllocationId) override {
    cudaFree(address);
    return absl::OkStatus();
  }
  void Abandon(void* address, inferx::ByteCount, inferx::ByteCount,
               inferx::AllocationId) noexcept override {
    cudaFree(address);
  }
};

using inferx::kernels::KernelExecutionContext;

struct DevTensor {
  // Heap storage keeps DevTensor default-constructible; the underlying
  // buffers leak until exit by design (see MakeTensor).
  std::shared_ptr<inferx::Buffer> buffer;
  std::shared_ptr<inferx::MutableTensorView> view_storage;
  inferx::MutableTensorView& view() { return *view_storage; }
  const inferx::MutableTensorView& view() const { return *view_storage; }
};

// Benchmark tensors intentionally leak their device buffers until exit:
// interleaving cudaFree with large H2D copies produced nondeterministic
// WSL2 driver faults during bring-up, and the benchmark never needs the
// memory back.
DevTensor MakeTensor(std::vector<uint64_t> dims, inferx::DType dtype, uint64_t seed,
                     bool fill = true) {
  using namespace inferx;
  auto shape = Shape::Create(dims);
  auto strides = Strides::Contiguous(*shape);
  auto bytes = shape->Bytes(dtype);
  void* memory = nullptr;
  cudaError_t alloc_status = cudaMalloc(&memory, bytes->value());
  if (alloc_status != cudaSuccess || memory == nullptr) {
    fprintf(stderr, "cudaMalloc(%llu) failed: %s\n",
            static_cast<unsigned long long>(bytes->value()),
            cudaGetErrorString(alloc_status));
    exit(2);
  }
  if (fill && dtype != DType::kUInt8 && dtype != DType::kInt32) {
    const size_t element_bytes = (dtype == inferx::DType::kFloat32) ? 4 : 2;
    const size_t count = bytes->value() / element_bytes;
    std::vector<float> values(count);
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(-1.5F, 1.5F);
    for (float& value : values) value = dist(rng);
    if (dtype == DType::kFloat16) {
      std::vector<__half> halfs(count);
      for (size_t i = 0; i < count; ++i) halfs[i] = __float2half(values[i]);
      cudaError_t c = cudaMemcpy(memory, halfs.data(), bytes->value(), cudaMemcpyHostToDevice);
      if (c != cudaSuccess) { fprintf(stderr, "memcpy(fp16) failed: %s\n", cudaGetErrorString(c)); }
    } else if (dtype == DType::kBFloat16) {
      std::vector<__nv_bfloat16> bf16(count);
      for (size_t i = 0; i < count; ++i) bf16[i] = __float2bfloat16(values[i]);
      cudaError_t c = cudaMemcpy(memory, bf16.data(), bytes->value(), cudaMemcpyHostToDevice);
      if (c != cudaSuccess) { fprintf(stderr, "memcpy(bf16) failed: %s\n", cudaGetErrorString(c)); }
    } else if (dtype == DType::kFloat32) {
      cudaError_t c = cudaMemcpy(memory, values.data(), bytes->value(), cudaMemcpyHostToDevice);
      if (c != cudaSuccess) { fprintf(stderr, "memcpy(fp32) failed: %s\n", cudaGetErrorString(c)); }
    }
  }
  auto allocation_id = inferx::NextAllocationId();
  auto buffer = inferx::Buffer::Adopt(
      memory,
      inferx::AllocationRequest{Device::Cuda(DeviceId(0)), MemoryKind::kDevice, *bytes,
                                ByteCount(256), MemoryCategory::kTest},
      ByteCount(256), allocation_id.ok() ? *allocation_id : inferx::AllocationId(0),
      std::make_shared<BenchDomain>());
  auto view = buffer->MutableView(ByteRange{ByteCount(0), *bytes});
  auto tensor = inferx::MutableTensorView::Create(*view, dtype, *shape, *strides);
  DevTensor result;
  result.buffer = std::make_shared<inferx::Buffer>(std::move(*buffer));
  result.view_storage = std::make_shared<inferx::MutableTensorView>(*tensor);
  return result;
}

KernelExecutionContext Context89() {
  KernelExecutionContext context;
  context.device = inferx::Device::Cuda(inferx::DeviceId(0));
  context.stream_handle = 0;
  context.compute_capability = 89;
  return context;
}

struct Sample {
  std::string op;
  std::string workload;
  double micros = 0.0;
};

cudaEvent_t g_start;
cudaEvent_t g_stop;

template <typename Fn>
double TimeOnce(Fn&& launch, uint32_t iters, uint32_t warmup) {
  KernelExecutionContext context = Context89();
  for (uint32_t i = 0; i < warmup; ++i) {
    absl::Status warmup_status = launch(context);
    if (!warmup_status.ok()) { fprintf(stderr, "warmup failed: %s\n", warmup_status.ToString().c_str()); return -1.0; }
  }
  cudaDeviceSynchronize();
  std::vector<float> times(iters);
  for (uint32_t i = 0; i < iters; ++i) {
    cudaEventRecord(g_start);
    absl::Status status = launch(context);
    cudaEventRecord(g_stop);
    cudaEventSynchronize(g_stop);
    if (!status.ok()) { fprintf(stderr, "launch failed: %s\n", status.ToString().c_str()); return -1.0; }
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, g_start, g_stop);
    times[i] = ms * 1000.0F;  // microseconds
  }
  std::sort(times.begin(), times.end());
  return times[iters / 2];  // median
}

using inferx::kernels::ActMulKind;
using inferx::kernels::ActMulRequest;

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) != "--json") {
    fprintf(stderr, "usage: %s [--json]\n", argv[0]);
    return 1;
  }
  inferx::kernels::ResetKernelBackendsForTest();
  absl::Status registered = inferx::kernels::cuda::RegisterCudaKernelBackend();
  if (!registered.ok()) {
    fprintf(stderr, "backend registration failed: %s\n", registered.ToString().c_str());
    return 1;
  }
  constexpr uint32_t kIters = 50;
  constexpr uint32_t kWarmup = 10;
  std::vector<Sample> samples;
  cudaFree(0);  // establish the CUDA context before creating events
  cudaEventCreate(&g_start);
  cudaEventCreate(&g_stop);
  auto emit = [&](const char* op, const char* workload, double micros) {
    samples.push_back(Sample{op, workload, micros});
    cudaError_t sticky = cudaDeviceSynchronize();
    if (sticky != cudaSuccess) {
      fprintf(stderr, "phase %s/%s left device in error: %s\n", op, workload,
              cudaGetErrorString(sticky));
      cudaGetLastError();  // cannot recover a device fault; report and continue
    }
  };

  const char* stage_env = getenv("BENCH_STAGE");
  const int stage = stage_env ? atoi(stage_env) : 99;
  // act_mul (silu / gelu) [T, 2D] -> [T, D]
  if (stage >= 1)
  for (const auto& [tokens, dim, kind] :
       std::vector<std::tuple<uint64_t, uint64_t, const char*>>{
           {4096, 5120, "silu"}, {16384, 8192, "silu"}, {4096, 5120, "gelu"}}) {
    auto input = MakeTensor({tokens, 2 * dim}, inferx::DType::kBFloat16, 1);
    auto output = MakeTensor({tokens, dim}, inferx::DType::kBFloat16, 2, /*fill=*/false);
    ActMulKind k = std::string(kind) == "silu" ? ActMulKind::kSilu : ActMulKind::kGelu;
    ActMulRequest request{input.view().AsConst(), output.view(), k};
    std::string name = std::string(kind) == "silu"
                           ? (tokens == 4096 ? "T4096_D5120" : "T16384_D8192")
                           : "T4096_D5120";
    std::string opname = std::string(kind) == "silu" ? "act_mul_silu" : "act_mul_gelu";
    emit(opname.c_str(), name.c_str(),
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchActMul(request, c); }, kIters,
                  kWarmup));
  }

  // rmsnorm T4096 H5120 bf16
  if (stage >= 2)
  {
    auto input = MakeTensor({4096, 5120}, inferx::DType::kBFloat16, 3);
    auto weight = MakeTensor({5120}, inferx::DType::kBFloat16, 4);
    auto output = MakeTensor({4096, 5120}, inferx::DType::kBFloat16, 5, false);
    inferx::ops::RmsNormRequest request{input.view().AsConst(), weight.view().AsConst(), output.view(),
                                        1.0e-5F};
    emit("rmsnorm", "T4096_H5120",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchRmsNorm(request, c); }, kIters,
                  kWarmup));
  }

  // fused_add_rmsnorm T4096 H5120 bf16
  if (stage >= 3)
  {
    auto input = MakeTensor({4096, 5120}, inferx::DType::kBFloat16, 6);
    auto residual = MakeTensor({4096, 5120}, inferx::DType::kBFloat16, 7);
    auto weight = MakeTensor({5120}, inferx::DType::kBFloat16, 8);
    inferx::kernels::FusedAddRmsNormRequest request{input.view(), residual.view(),
                                                    weight.view().AsConst(), 1.0e-5F};
    emit("fused_add_rmsnorm", "T4096_H5120",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchFusedAddRmsNorm(request, c); },
                  kIters, kWarmup));
  }

  // qk_rmsnorm T4096 H32/KV8 D128 bf16
  {
    auto q = MakeTensor({4096, 32, 128}, inferx::DType::kBFloat16, 9);
    auto k = MakeTensor({4096, 8, 128}, inferx::DType::kBFloat16, 10);
    auto qw = MakeTensor({128}, inferx::DType::kBFloat16, 11);
    auto kw = MakeTensor({128}, inferx::DType::kBFloat16, 12);
    inferx::kernels::QkRmsNormRequest request{q.view(), k.view(), qw.view().AsConst(),
                                              kw.view().AsConst(), 1.0e-5F};
    emit("qk_rmsnorm", "T4096_H32_D128",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchQkRmsNorm(request, c); }, kIters,
                  kWarmup));
  }

  // rope T4096 H32/KV8 D128 bf16
  {
    auto q = MakeTensor({4096, 32, 128}, inferx::DType::kBFloat16, 13);
    auto k = MakeTensor({4096, 8, 128}, inferx::DType::kBFloat16, 14);
    auto qo = MakeTensor({4096, 32, 128}, inferx::DType::kBFloat16, 15, false);
    auto ko = MakeTensor({4096, 8, 128}, inferx::DType::kBFloat16, 16, false);
    std::vector<int32_t> positions(4096);
    for (uint64_t t = 0; t < 4096; ++t) positions[t] = static_cast<int32_t>(t);
    auto pos = MakeTensor({4096}, inferx::DType::kInt32, 17, false);
    cudaMemcpy(const_cast<void*>(inferx::cuda::BufferAccess::Address(pos.view().AsConst().buffer())),
               positions.data(), 4096 * 4, cudaMemcpyHostToDevice);
    std::vector<int32_t> host_positions = positions;
    inferx::ops::RopeRequest request{
        q.view().AsConst(), k.view().AsConst(), pos.view().AsConst(), qo.view(), ko.view(), 10000.0F, 8192,
        host_positions};
    emit("rope", "T4096_H32_D128",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchRope(request, c); }, kIters,
                  kWarmup));
  }

  // gemm M4096/M8 K4096 N4096 bf16
  for (uint64_t tokens : {4096ULL, 8ULL}) {
    auto input = MakeTensor({tokens, 4096}, inferx::DType::kBFloat16, 18);
    auto weight = MakeTensor({4096, 4096}, inferx::DType::kBFloat16, 19);
    auto output = MakeTensor({tokens, 4096}, inferx::DType::kBFloat16, 20, false);
    inferx::ops::GemmRequest request{input.view().AsConst(), weight.view().AsConst(), std::nullopt,
                                     output.view()};
    emit("gemm", tokens == 4096 ? "M4096_K4096_N4096" : "M8_K4096_N4096",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchGemm(request, c); }, kIters,
                  kWarmup));
  }

  // attention decode B1_S4096 / B8_S2048 (bf16)
  for (const auto& [batch, context_len] :
       std::vector<std::tuple<uint64_t, uint64_t>>{{1, 4096}, {8, 2048}}) {
    auto q = MakeTensor({batch, 32, 128}, inferx::DType::kBFloat16, 21);
    auto new_k = MakeTensor({batch, 8, 128}, inferx::DType::kBFloat16, 22);
    auto new_v = MakeTensor({batch, 8, 128}, inferx::DType::kBFloat16, 23);
    auto k_cache = MakeTensor({batch, context_len, 8, 128}, inferx::DType::kBFloat16, 24);
    auto v_cache = MakeTensor({batch, context_len, 8, 128}, inferx::DType::kBFloat16, 25);
    auto output = MakeTensor({batch, 32, 128}, inferx::DType::kBFloat16, 26, false);
    std::vector<int32_t> q_indptr(batch + 1), kv_indptr(batch + 1), positions, lengths(batch);
    for (uint64_t b = 0; b < batch; ++b) {
      q_indptr[b + 1] = static_cast<int32_t>(b + 1);
      kv_indptr[b + 1] = static_cast<int32_t>(b + 1);
      lengths[b] = static_cast<int32_t>(context_len - 1);
      positions.push_back(static_cast<int32_t>(context_len - 1));
    }
    // device mirrors
    auto d_q_indptr = MakeTensor({batch + 1}, inferx::DType::kInt32, 0, false);
    auto d_kv_indptr = MakeTensor({batch + 1}, inferx::DType::kInt32, 0, false);
    auto d_positions = MakeTensor({batch}, inferx::DType::kInt32, 0, false);
    auto d_lengths = MakeTensor({batch}, inferx::DType::kInt32, 0, false);
    auto upload = [&](DevTensor& t, const void* data, size_t bytes) {
      cudaMemcpy(const_cast<void*>(inferx::cuda::BufferAccess::Address(t.view().AsConst().buffer())),
                 data, bytes, cudaMemcpyHostToDevice);
    };
    upload(d_q_indptr, q_indptr.data(), q_indptr.size() * 4);
    upload(d_kv_indptr, kv_indptr.data(), kv_indptr.size() * 4);
    upload(d_positions, positions.data(), positions.size() * 4);
    upload(d_lengths, lengths.data(), lengths.size() * 4);
    inferx::kernels::DeviceAttentionMetadata metadata{
        reinterpret_cast<const int32_t*>(
            inferx::cuda::BufferAccess::Address(d_q_indptr.view().AsConst().buffer())),
        reinterpret_cast<const int32_t*>(
            inferx::cuda::BufferAccess::Address(d_kv_indptr.view().AsConst().buffer())),
        reinterpret_cast<const int32_t*>(
            inferx::cuda::BufferAccess::Address(d_positions.view().AsConst().buffer())),
        reinterpret_cast<const int32_t*>(
            inferx::cuda::BufferAccess::Address(d_lengths.view().AsConst().buffer()))};
    inferx::ops::AttentionRequest request{q.view().AsConst(),  new_k.view().AsConst(),
                                          new_v.view().AsConst(), q_indptr,
                                          kv_indptr,            positions,
                                          lengths,              k_cache.view(),
                                          v_cache.view(),         output.view(),
                                          inferx::ops::ExecutionPhase::kDecode};
    void* workspace = nullptr;
    cudaMalloc(&workspace, 64 * 1024);
    emit("attention_decode", batch == 1 ? "B1_S4096" : "B8_S2048",
         TimeOnce(
             [&](auto& c) {
               c.workspace = workspace;
               c.workspace_bytes = 64 * 1024;
               return inferx::kernels::LaunchAttention(request, metadata, c);
             },
             kIters, kWarmup));
    cudaFree(workspace);
  }

  // argmax / top_p_renorm B8 V131072 fp32
  {
    auto logits = MakeTensor({8, 131072}, inferx::DType::kFloat32, 27);
    auto ids = MakeTensor({8}, inferx::DType::kInt32, 0, false);
    inferx::kernels::ArgmaxRequest request{logits.view().AsConst(), ids.view()};
    emit("argmax", "B8_V131072",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchArgmax(request, c); }, kIters,
                  kWarmup));
  }
  {
    auto probs = MakeTensor({8, 131072}, inferx::DType::kFloat32, 28);
    inferx::kernels::TopPRenormRequest request{probs.view(), 0.95F};
    emit("top_p_renorm", "B8_V131072",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchTopPRenorm(request, c); }, kIters,
                  kWarmup));
  }

  // fp8 quant T4096 D7168 group 128 (fp32 in, uint8 out)
  {
    auto input = MakeTensor({4096, 7168}, inferx::DType::kFloat32, 29);
    auto output = MakeTensor({4096, 7168}, inferx::DType::kUInt8, 0, false);
    auto scales = MakeTensor({4096ULL * 56ULL}, inferx::DType::kFloat32, 0, false);
    inferx::kernels::Fp8QuantRequest request{
        input.view().AsConst(), output.view(), scales.view(),
        inferx::kernels::QuantGranularity::kTokenGroup, 128};
    emit("fp8_quant", "T4096_D7168",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchFp8Quant(request, c); }, kIters,
                  kWarmup));
  }

  // softmax_topk T512 E256 K8 fp32
  {
    auto logits = MakeTensor({512, 256}, inferx::DType::kFloat32, 30);
    auto weights = MakeTensor({512, 8}, inferx::DType::kFloat32, 0, false);
    auto ids = MakeTensor({512, 8}, inferx::DType::kInt32, 0, false);
    inferx::kernels::SoftmaxTopKRequest request{logits.view().AsConst(), weights.view(), ids.view(),
                                                true};
    emit("softmax_topk", "T512_E256_K8",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchSoftmaxTopK(request, c); }, kIters,
                  kWarmup));
  }

  // hadamard T4096 D128 bf16
  {
    auto input = MakeTensor({4096, 128}, inferx::DType::kBFloat16, 31);
    auto output = MakeTensor({4096, 128}, inferx::DType::kBFloat16, 32, false);
    inferx::kernels::HadamardTransformRequest request{input.view().AsConst(), output.view(), 0.044F};
    emit("hadamard", "T4096_D128",
         TimeOnce([&](auto& c) { return inferx::kernels::LaunchHadamardTransform(request, c); },
                  kIters, kWarmup));
  }

  // JSON output
  fprintf(stderr, "samples: %zu\n", samples.size());
  printf("[\n");
  for (size_t i = 0; i < samples.size(); ++i) {
    printf("  {\"impl\": \"inferx\", \"op\": \"%s\", \"workload\": \"%s\", \"latency_us\": %.3f}%s\n",
           samples[i].op.c_str(), samples[i].workload.c_str(), samples[i].micros,
           i + 1 == samples.size() ? "" : ",");
  }
  printf("]\n");
  inferx::kernels::ResetKernelBackendsForTest();
  return 0;
}
