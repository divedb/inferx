// CPU reference oracles: activation, layernorm variants, hadamard.
#include <cmath>

#include "absl/status/status.h"
#include "inferx/kernels/ops/activation.h"
#include "inferx/kernels/ops/layernorm.h"
#include "inferx/kernels/ops/transform.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {
namespace {}  // namespace

absl::Status ValidateActMul(const ActMulRequest& request) {
  if (request.input.shape().rank() != 2 || request.input.shape().dim(0) == 0 ||
      request.input.shape().dim(1) % 2 != 0 || request.input.dtype() != request.output.dtype() ||
      request.output.shape().dim(0) != request.input.shape().dim(0) ||
      request.output.shape().dim(1) * 2 != request.input.shape().dim(1)) {
    return absl::InvalidArgumentError("act_mul: input[tokens, 2d] -> output[tokens, d] required");
  }
  return absl::OkStatus();
}

absl::Status ReferenceActMul(const ActMulRequest& request) {
  absl::Status status = ValidateActMul(request);
  if (!status.ok()) return status;
  const uint64_t tokens = request.input.shape().dim(0);
  const uint64_t half = request.input.shape().dim(1) / 2;
  auto input_bytes = request.input.buffer().HostBytes();
  auto output_bytes = request.output.buffer().HostBytes();
  if (!input_bytes.ok() || !output_bytes.ok()) {
    return absl::InvalidArgumentError("act_mul: host tensors required for the reference");
  }
  const bool fp32 = request.input.dtype() == Dtype::kFloat32;
  if (request.input.dtype() != Dtype::kFloat32 && request.input.dtype() != Dtype::kFloat16 &&
      request.input.dtype() != Dtype::kBFloat16) {
    return absl::UnimplementedError("act_mul: FP32/FP16/BF16 required");
  }
  const auto read = [&](uint64_t index) {
    if (fp32) {
      return reinterpret_cast<const float*>(input_bytes->data())[index];
    }
    return 0.0F;  // narrow dtypes unsupported in this simplified reference
  };
  (void)read;
  if (!fp32) return absl::UnimplementedError("act_mul reference: FP32 tensors required");
  const float* input = reinterpret_cast<const float*>(input_bytes->data());
  float* output = reinterpret_cast<float*>(output_bytes->data());
  for (uint64_t token = 0; token < tokens; ++token) {
    for (uint64_t column = 0; column < half; ++column) {
      const float gate = input[token * 2 * half + column];
      const float up = input[token * 2 * half + half + column];
      float activated;
      switch (request.kind) {
        case ActMulKind::kSilu:
          activated = gate / (1.0F + std::exp(-gate));
          break;
        case ActMulKind::kGelu:
          activated = 0.5F * gate * (1.0F + std::erf(gate * 0.70710678118654752440F));
          break;
        case ActMulKind::kGeluTanh:
          activated =
              0.5F * gate *
              (1.0F + std::tanh(0.7978845608028654F * (gate + 0.044715F * gate * gate * gate)));
          break;
        default:
          return absl::InvalidArgumentError("act_mul: unknown activation kind");
      }
      output[token * half + column] = activated * up;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateAdd3(const Add3Request& request) {
  if (request.a.shape().rank() != 2 || request.a.shape() != request.b.shape() ||
      request.a.shape() != request.c.shape() || request.a.shape() != request.output.shape() ||
      request.a.dtype() != request.b.dtype() || request.a.dtype() != request.c.dtype() ||
      request.a.dtype() != request.output.dtype()) {
    return absl::InvalidArgumentError("add3: equal shape and dtype required");
  }
  return absl::OkStatus();
}

absl::Status ReferenceAdd3(const Add3Request& request) {
  absl::Status status = ValidateAdd3(request);
  if (!status.ok()) return status;
  if (request.a.dtype() != Dtype::kFloat32) {
    return absl::UnimplementedError("add3 reference: FP32 tensors required");
  }
  auto a = request.a.buffer().HostBytes();
  auto b = request.b.buffer().HostBytes();
  auto c = request.c.buffer().HostBytes();
  auto out = request.output.buffer().HostBytes();
  if (!a.ok() || !b.ok() || !c.ok() || !out.ok()) {
    return absl::InvalidArgumentError("add3: host tensors required for the reference");
  }
  const size_t count = a->size() / sizeof(float);
  const float* pa = reinterpret_cast<const float*>(a->data());
  const float* pb = reinterpret_cast<const float*>(b->data());
  const float* pc = reinterpret_cast<const float*>(c->data());
  float* po = reinterpret_cast<float*>(out->data());
  for (size_t index = 0; index < count; ++index) {
    po[index] = pa[index] + pb[index] + pc[index];
  }
  return absl::OkStatus();
}

absl::Status ValidateFusedAddRmsNorm(const FusedAddRmsNormRequest& request) {
  if (request.input.shape().rank() != 2 || request.input.shape() != request.residual.shape() ||
      request.input.dtype() != request.residual.dtype() || request.weight.shape().rank() != 1 ||
      request.weight.shape().dim(0) != request.input.shape().dim(1) ||
      request.weight.dtype() != request.input.dtype() || request.input.shape().dim(1) == 0 ||
      !std::isfinite(request.epsilon) || request.epsilon <= 0.0F) {
    return absl::InvalidArgumentError("fused_add_rmsnorm: incompatible tensors");
  }
  return absl::OkStatus();
}

absl::Status ReferenceFusedAddRmsNorm(const FusedAddRmsNormRequest& request) {
  absl::Status status = ValidateFusedAddRmsNorm(request);
  if (!status.ok()) return status;
  if (request.input.dtype() != Dtype::kFloat32) {
    return absl::UnimplementedError("fused_add_rmsnorm reference: FP32 tensors required");
  }
  auto input = request.input.buffer().HostBytes();
  auto residual = request.residual.buffer().HostBytes();
  auto weight = request.weight.buffer().HostBytes();
  if (!input.ok() || !residual.ok() || !weight.ok()) {
    return absl::InvalidArgumentError("fused_add_rmsnorm: host tensors required");
  }
  const uint64_t tokens = request.input.shape().dim(0);
  const uint64_t hidden = request.input.shape().dim(1);
  float* in = reinterpret_cast<float*>(input->data());
  float* res = reinterpret_cast<float*>(residual->data());
  const float* w = reinterpret_cast<const float*>(weight->data());
  for (uint64_t token = 0; token < tokens; ++token) {
    float sum = 0.0F;
    for (uint64_t index = 0; index < hidden; ++index) {
      res[token * hidden + index] += in[token * hidden + index];
      const float value = res[token * hidden + index];
      sum += value * value;
    }
    const float inverse_rms = 1.0F / std::sqrt(sum / static_cast<float>(hidden) + request.epsilon);
    for (uint64_t index = 0; index < hidden; ++index) {
      in[token * hidden + index] = res[token * hidden + index] * inverse_rms * w[index];
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateGemmaRmsNorm(const GemmaRmsNormRequest& request) {
  if (request.input.shape().rank() != 2 || request.input.shape() != request.output.shape() ||
      request.weight.shape().rank() != 1 ||
      request.weight.shape().dim(0) != request.input.shape().dim(1) ||
      request.input.dtype() != request.weight.dtype() ||
      request.input.dtype() != request.output.dtype() || request.input.shape().dim(1) == 0 ||
      !std::isfinite(request.epsilon) || request.epsilon <= 0.0F) {
    return absl::InvalidArgumentError("gemma_rmsnorm: incompatible tensors");
  }
  return absl::OkStatus();
}

absl::Status ReferenceGemmaRmsNorm(const GemmaRmsNormRequest& request) {
  absl::Status status = ValidateGemmaRmsNorm(request);
  if (!status.ok()) return status;
  if (request.input.dtype() != Dtype::kFloat32) {
    return absl::UnimplementedError("gemma_rmsnorm reference: FP32 tensors required");
  }
  auto input = request.input.buffer().HostBytes();
  auto weight = request.weight.buffer().HostBytes();
  auto output = request.output.buffer().HostBytes();
  if (!input.ok() || !weight.ok() || !output.ok()) {
    return absl::InvalidArgumentError("gemma_rmsnorm: host tensors required");
  }
  const uint64_t tokens = request.input.shape().dim(0);
  const uint64_t hidden = request.input.shape().dim(1);
  const float* in = reinterpret_cast<const float*>(input->data());
  const float* w = reinterpret_cast<const float*>(weight->data());
  float* out = reinterpret_cast<float*>(output->data());
  for (uint64_t token = 0; token < tokens; ++token) {
    float sum = 0.0F;
    for (uint64_t index = 0; index < hidden; ++index) {
      sum += in[token * hidden + index] * in[token * hidden + index];
    }
    const float inverse_rms = 1.0F / std::sqrt(sum / static_cast<float>(hidden) + request.epsilon);
    for (uint64_t index = 0; index < hidden; ++index) {
      out[token * hidden + index] = in[token * hidden + index] * inverse_rms * (1.0F + w[index]);
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateQkRmsNorm(const QkRmsNormRequest& request) {
  if (request.query.shape().rank() != 3 || request.key.shape().rank() != 3 ||
      request.query.shape().dim(0) != request.key.shape().dim(0) ||
      request.query.shape().dim(2) != request.key.shape().dim(2) ||
      request.query.shape().dim(2) == 0 ||
      request.query_weight.shape().dim(0) != request.query.shape().dim(2) ||
      request.key_weight.shape().dim(0) != request.key.shape().dim(2) ||
      request.query.dtype() != request.key.dtype() ||
      request.query.dtype() != request.query_weight.dtype()) {
    return absl::InvalidArgumentError("qk_rmsnorm: incompatible tensors");
  }
  return absl::OkStatus();
}

absl::Status ReferenceQkRmsNorm(const QkRmsNormRequest& request) {
  absl::Status status = ValidateQkRmsNorm(request);
  if (!status.ok()) return status;
  if (request.query.dtype() != Dtype::kFloat32) {
    return absl::UnimplementedError("qk_rmsnorm reference: FP32 tensors required");
  }
  auto q = request.query.buffer().HostBytes();
  auto k = request.key.buffer().HostBytes();
  auto qw = request.query_weight.buffer().HostBytes();
  auto kw = request.key_weight.buffer().HostBytes();
  if (!q.ok() || !k.ok() || !qw.ok() || !kw.ok()) {
    return absl::InvalidArgumentError("qk_rmsnorm: host tensors required");
  }
  const uint64_t tokens = request.query.shape().dim(0);
  const uint64_t head_dim = request.query.shape().dim(2);
  const float* q_weight = reinterpret_cast<const float*>(qw->data());
  const float* k_weight = reinterpret_cast<const float*>(kw->data());
  float* query = reinterpret_cast<float*>(q->data());
  float* key = reinterpret_cast<float*>(k->data());
  const uint64_t q_heads = request.query.shape().dim(1);
  const uint64_t k_heads = request.key.shape().dim(1);
  for (uint64_t token = 0; token < tokens; ++token) {
    for (uint64_t head = 0; head < q_heads; ++head) {
      float* row = query + (token * q_heads + head) * head_dim;
      float sum = 0.0F;
      for (uint64_t index = 0; index < head_dim; ++index) sum += row[index] * row[index];
      const float inverse_rms =
          1.0F / std::sqrt(sum / static_cast<float>(head_dim) + request.epsilon);
      for (uint64_t index = 0; index < head_dim; ++index) {
        row[index] = row[index] * inverse_rms * q_weight[index];
      }
    }
    for (uint64_t head = 0; head < k_heads; ++head) {
      float* row = key + (token * k_heads + head) * head_dim;
      float sum = 0.0F;
      for (uint64_t index = 0; index < head_dim; ++index) sum += row[index] * row[index];
      const float inverse_rms =
          1.0F / std::sqrt(sum / static_cast<float>(head_dim) + request.epsilon);
      for (uint64_t index = 0; index < head_dim; ++index) {
        row[index] = row[index] * inverse_rms * k_weight[index];
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateHadamardTransform(const HadamardTransformRequest& request) {
  if (request.input.shape().rank() < 1 || request.input.shape() != request.output.shape() ||
      request.input.shape().dim(request.input.shape().rank() - 1) != 128 ||
      request.input.dtype() != request.output.dtype() || !std::isfinite(request.scale)) {
    return absl::InvalidArgumentError("hadamard: last dim must be 128 and shapes must match");
  }
  return absl::OkStatus();
}

absl::Status ReferenceHadamardTransform(const HadamardTransformRequest& request) {
  absl::Status status = ValidateHadamardTransform(request);
  if (!status.ok()) return status;
  if (request.input.dtype() != Dtype::kFloat32) {
    return absl::UnimplementedError("hadamard reference: FP32 tensors required");
  }
  auto input = request.input.buffer().HostBytes();
  auto output = request.output.buffer().HostBytes();
  if (!input.ok() || !output.ok()) {
    return absl::InvalidArgumentError("hadamard: host tensors required");
  }
  const size_t rows = input->size() / (128 * sizeof(float));
  const float* in = reinterpret_cast<const float*>(input->data());
  float* out = reinterpret_cast<float*>(output->data());
  for (size_t row = 0; row < rows; ++row) {
    const float* src = in + row * 128;
    float* dst = out + row * 128;
    for (int i = 0; i < 128; ++i) {
      float sum = 0.0F;
      for (int j = 0; j < 128; ++j) {
        const float sign =
            ((__builtin_popcount(static_cast<unsigned int>(i) & static_cast<unsigned int>(j)) &
              1) != 0)
                ? -1.0F
                : 1.0F;
        sum += sign * src[j];
      }
      dst[i] = sum * request.scale;
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx::kernels
