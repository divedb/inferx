// CPU reference oracles: argmax, top-k/top-p renormalization, FP8
// quantization, MoE routing.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "inferx/kernels/ops/sampling.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {
namespace {

// Host float -> FP8 e4m3 bit encoder (round-to-nearest, NaN maps to 0x7f).
uint8_t FloatToFp8E4m3Bits(float value) {
  const uint32_t bits = __builtin_bit_cast(uint32_t, value);
  const uint32_t sign = (bits >> 24) & 0x80U;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 7;
  uint32_t mantissa = bits & 0x7FFFFFU;
  uint8_t result = 0;
  if (((bits >> 23) & 0xFFU) == 0xFFU) return static_cast<uint8_t>(sign | 0x7FU);
  if (((bits >> 23) & 0xFFU) == 0) return static_cast<uint8_t>(sign);  // zero
  if (exponent <= 0) {
    if (exponent < -3) return static_cast<uint8_t>(sign);
    // subnormal e4m3: mantissa carries the implicit leading zeros
    const uint32_t shifted = mantissa | 0x800000U;
    const int32_t shift = 1 - exponent;
    uint32_t sub = shifted >> shift;
    const uint32_t round_bit = (shifted >> (shift - 1)) & 1U;
    sub += round_bit;
    result = static_cast<uint8_t>(sub >> 20);
    return static_cast<uint8_t>(sign | result);
  }
  if (exponent >= 15) return static_cast<uint8_t>(sign | 0x7EU);  // saturate
  // round mantissa (23 -> 3 bits) with round-to-nearest-even
  uint32_t m3 = mantissa >> 20;
  const uint32_t remainder = mantissa & 0xFFFFFU;
  if (remainder > 0x7FFFFU || (remainder == 0x7FFFFU && (m3 & 1U) != 0U)) ++m3;
  if (m3 == 0x8U) {
    m3 = 0;
    ++exponent;
    if (exponent >= 15) return static_cast<uint8_t>(sign | 0x7EU);
  }
  result = static_cast<uint8_t>((static_cast<uint32_t>(exponent) << 3) | (m3 & 0x7U));
  return static_cast<uint8_t>(sign | result);
}

absl::Status RequireFp32Host(const TensorView& tensor, const char* field) {
  if (tensor.dtype() != DType::kFloat32) {
    return absl::UnimplementedError(std::string(field) + " reference: FP32 tensors required");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateArgmax(const ArgmaxRequest& request) {
  if (request.logits.shape().rank() != 2 || request.logits.shape().dim(0) == 0 ||
      request.logits.shape().dim(1) == 0 || request.output.shape().rank() != 1 ||
      request.output.shape().dim(0) != request.logits.shape().dim(0) ||
      (request.output.dtype() != DType::kInt32 && request.output.dtype() != DType::kInt64)) {
    return absl::InvalidArgumentError("argmax: logits[rows, vocab] -> ids[rows] required");
  }
  return absl::OkStatus();
}

absl::Status ReferenceArgmax(const ArgmaxRequest& request) {
  absl::Status status = ValidateArgmax(request);
  if (!status.ok()) return status;
  status = RequireFp32Host(request.logits, "argmax");
  if (!status.ok()) return status;
  auto logits = request.logits.buffer().HostBytes();
  auto output = request.output.buffer().HostBytes();
  if (!logits.ok() || !output.ok()) {
    return absl::InvalidArgumentError("argmax: host tensors required");
  }
  const uint64_t rows = request.logits.shape().dim(0);
  const uint64_t vocab = request.logits.shape().dim(1);
  const float* in = reinterpret_cast<const float*>(logits->data());
  int32_t* out = reinterpret_cast<int32_t*>(output->data());
  for (uint64_t row = 0; row < rows; ++row) {
    int32_t best = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    for (uint64_t column = 0; column < vocab; ++column) {
      const float value = in[row * vocab + column];
      if (std::isnan(value)) continue;
      if (value > best_value) {
        best_value = value;
        best = static_cast<int32_t>(column);
      }
    }
    out[row] = best;
  }
  return absl::OkStatus();
}

absl::Status ValidateTopPRenorm(const TopPRenormRequest& request) {
  if (request.probs.shape().rank() != 2 || !(request.top_p > 0.0F && request.top_p <= 1.0F)) {
    return absl::InvalidArgumentError("top_p_renorm: probs[rows, vocab] and 0 < p <= 1 required");
  }
  return absl::OkStatus();
}

absl::Status ReferenceTopPRenorm(const TopPRenormRequest& request) {
  absl::Status status = ValidateTopPRenorm(request);
  if (!status.ok()) return status;
  status = RequireFp32Host(request.probs.AsConst(), "top_p_renorm");
  if (!status.ok()) return status;
  auto bytes = request.probs.buffer().HostBytes();
  if (!bytes.ok()) return absl::InvalidArgumentError("top_p_renorm: host tensors required");
  const uint64_t rows = request.probs.shape().dim(0);
  const uint64_t vocab = request.probs.shape().dim(1);
  float* data = reinterpret_cast<float*>(bytes->data());
  std::vector<uint32_t> order(vocab);
  for (uint64_t row = 0; row < rows; ++row) {
    float* prow = data + row * vocab;
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return prow[a] > prow[b]; });
    float cumulative = 0.0F;
    uint64_t keep = 0;
    for (; keep < vocab; ++keep) {
      cumulative += prow[order[keep]];
      if (cumulative >= request.top_p) {
        ++keep;
        break;
      }
    }
    if (keep == 0) keep = 1;
    std::vector<float> kept(vocab, 0.0F);
    float total = 0.0F;
    for (uint64_t index = 0; index < keep; ++index) {
      kept[order[index]] = prow[order[index]];
      total += prow[order[index]];
    }
    for (uint64_t column = 0; column < vocab; ++column) {
      prow[column] = total > 0.0F ? kept[column] / total : 0.0F;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateTopKRenorm(const TopKRenormRequest& request) {
  if (request.probs.shape().rank() != 2 || request.top_k == 0) {
    return absl::InvalidArgumentError("top_k_renorm: probs[rows, vocab] and k > 0 required");
  }
  return absl::OkStatus();
}

absl::Status ReferenceTopKRenorm(const TopKRenormRequest& request) {
  absl::Status status = ValidateTopKRenorm(request);
  if (!status.ok()) return status;
  status = RequireFp32Host(request.probs.AsConst(), "top_k_renorm");
  if (!status.ok()) return status;
  auto bytes = request.probs.buffer().HostBytes();
  if (!bytes.ok()) return absl::InvalidArgumentError("top_k_renorm: host tensors required");
  const uint64_t rows = request.probs.shape().dim(0);
  const uint64_t vocab = request.probs.shape().dim(1);
  const uint64_t k = std::min<uint64_t>(request.top_k, vocab);
  float* data = reinterpret_cast<float*>(bytes->data());
  std::vector<uint32_t> order(vocab);
  for (uint64_t row = 0; row < rows; ++row) {
    float* prow = data + row * vocab;
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return prow[a] > prow[b]; });
    std::vector<float> kept(vocab, 0.0F);
    float total = 0.0F;
    for (uint64_t index = 0; index < k; ++index) {
      kept[order[index]] = prow[order[index]];
      total += prow[order[index]];
    }
    for (uint64_t column = 0; column < vocab; ++column) {
      prow[column] = total > 0.0F ? kept[column] / total : 0.0F;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateFp8Quant(const Fp8QuantRequest& request) {
  if (request.input.shape().rank() != 2 || request.input.shape().dim(1) == 0 ||
      request.output.shape() != request.input.shape() ||
      request.output.dtype() != DType::kUInt8 || request.input.dtype() != DType::kFloat32 ||
      request.scales.dtype() != DType::kFloat32 ||
      (request.granularity == QuantGranularity::kTokenGroup && request.group_size == 0)) {
    return absl::InvalidArgumentError(
        "fp8_quant: input[tokens,dim] FP32 -> uint8 e4m3 bits + FP32 scales required");
  }
  const uint64_t tokens = request.input.shape().dim(0);
  const uint64_t dim = request.input.shape().dim(1);
  const uint64_t expected = request.granularity == QuantGranularity::kTensor
                                ? 1
                                : (request.granularity == QuantGranularity::kToken
                                       ? tokens
                                       : tokens * ((dim + request.group_size - 1) /
                                                   request.group_size));
  if (request.scales.shape().rank() != 1 || request.scales.shape().dim(0) != expected) {
    return absl::InvalidArgumentError("fp8_quant: unexpected scales shape");
  }
  return absl::OkStatus();
}

absl::Status ReferenceFp8Quant(const Fp8QuantRequest& request) {
  absl::Status status = ValidateFp8Quant(request);
  if (!status.ok()) return status;
  auto input = request.input.buffer().HostBytes();
  auto output = request.output.buffer().HostBytes();
  auto scales = request.scales.buffer().HostBytes();
  if (!input.ok() || !output.ok() || !scales.ok()) {
    return absl::InvalidArgumentError("fp8_quant: host tensors required");
  }
  const uint64_t tokens = request.input.shape().dim(0);
  const uint64_t dim = request.input.shape().dim(1);
  const float* in = reinterpret_cast<const float*>(input->data());
  uint8_t* out = reinterpret_cast<uint8_t*>(output->data());
  float* scale = reinterpret_cast<float*>(scales->data());
  constexpr float kFp8Max = 448.0F;
  const auto quantize = [&](uint64_t begin, uint64_t count, float* scale_out, uint8_t* dst) {
    float amax = 0.0F;
    for (uint64_t index = begin; index < begin + count; ++index) {
      amax = std::max(amax, std::fabs(in[index]));
    }
    const float s = amax > 0.0F ? amax / kFp8Max : 1.0F;
    *scale_out = s;
    for (uint64_t index = begin; index < begin + count; ++index) {
      dst[index - begin] = FloatToFp8E4m3Bits(std::clamp(in[index] / s, -kFp8Max, kFp8Max));
    }
  };
  if (request.granularity == QuantGranularity::kTensor) {
    quantize(0, tokens * dim, scale, out);
  } else if (request.granularity == QuantGranularity::kToken) {
    for (uint64_t token = 0; token < tokens; ++token) {
      quantize(token * dim, dim, scale + token, out + token * dim);
    }
  } else {
    const uint64_t groups = (dim + request.group_size - 1) / request.group_size;
    for (uint64_t token = 0; token < tokens; ++token) {
      for (uint64_t group = 0; group < groups; ++group) {
        const uint64_t begin = token * dim + group * request.group_size;
        const uint64_t count =
            std::min<uint64_t>(request.group_size, dim - group * request.group_size);
        quantize(begin, count, scale + token * groups + group, out + begin);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateSoftmaxTopK(const SoftmaxTopKRequest& request) {
  if (request.logits.shape().rank() != 2 || request.logits.shape().dim(0) == 0 ||
      request.weights.shape().rank() != 2 || request.ids.shape().rank() != 2 ||
      request.weights.shape().dim(0) != request.logits.shape().dim(0) ||
      request.weights.shape().dim(1) != request.ids.shape().dim(1) ||
      request.ids.shape().dim(0) != request.logits.shape().dim(0) ||
      request.weights.dtype() != DType::kFloat32 || request.ids.dtype() != DType::kInt32 ||
      request.logits.dtype() != DType::kFloat32) {
    return absl::InvalidArgumentError(
        "softmax_topk: logits[tokens,experts] -> weights FP32 + ids int32 required");
  }
  return absl::OkStatus();
}

absl::Status ReferenceSoftmaxTopK(const SoftmaxTopKRequest& request) {
  absl::Status status = ValidateSoftmaxTopK(request);
  if (!status.ok()) return status;
  auto logits = request.logits.buffer().HostBytes();
  auto weights = request.weights.buffer().HostBytes();
  auto ids = request.ids.buffer().HostBytes();
  if (!logits.ok() || !weights.ok() || !ids.ok()) {
    return absl::InvalidArgumentError("softmax_topk: host tensors required");
  }
  const uint64_t tokens = request.logits.shape().dim(0);
  const uint64_t experts = request.logits.shape().dim(1);
  const uint64_t k = request.weights.shape().dim(1);
  const float* in = reinterpret_cast<const float*>(logits->data());
  float* w = reinterpret_cast<float*>(weights->data());
  int32_t* id = reinterpret_cast<int32_t*>(ids->data());
  std::vector<uint32_t> order(experts);
  for (uint64_t token = 0; token < tokens; ++token) {
    const float* row = in + token * experts;
    float maximum = *std::max_element(row, row + experts);
    float total = 0.0F;
    std::vector<float> probs(experts);
    for (uint64_t expert = 0; expert < experts; ++expert) {
      probs[expert] = std::exp(row[expert] - maximum);
      total += probs[expert];
    }
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return probs[a] > probs[b]; });
    float kept = 0.0F;
    for (uint64_t index = 0; index < k; ++index) {
      const float probability = probs[order[index]] / total;
      w[token * k + index] = probability;
      id[token * k + index] = static_cast<int32_t>(order[index]);
      kept += probability;
    }
    if (request.renormalize && kept > 0.0F) {
      for (uint64_t index = 0; index < k; ++index) w[token * k + index] /= kept;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateSigmoidBiasTopK(const SigmoidBiasTopKRequest& request) {
  if (request.logits.shape().rank() != 2 || request.bias.shape().rank() != 1 ||
      request.bias.shape().dim(0) != request.logits.shape().dim(1) ||
      request.weights.shape().dim(0) != request.logits.shape().dim(0) ||
      request.ids.shape().dim(0) != request.logits.shape().dim(0) ||
      request.weights.shape().dim(1) != request.ids.shape().dim(1) ||
      request.weights.dtype() != DType::kFloat32 || request.ids.dtype() != DType::kInt32 ||
      request.logits.dtype() != DType::kFloat32) {
    return absl::InvalidArgumentError(
        "sigmoid_bias_topk: logits[tokens,experts] + bias[experts] -> weights + ids required");
  }
  return absl::OkStatus();
}

absl::Status ReferenceSigmoidBiasTopK(const SigmoidBiasTopKRequest& request) {
  absl::Status status = ValidateSigmoidBiasTopK(request);
  if (!status.ok()) return status;
  auto logits = request.logits.buffer().HostBytes();
  auto bias = request.bias.buffer().HostBytes();
  auto weights = request.weights.buffer().HostBytes();
  auto ids = request.ids.buffer().HostBytes();
  if (!logits.ok() || !bias.ok() || !weights.ok() || !ids.ok()) {
    return absl::InvalidArgumentError("sigmoid_bias_topk: host tensors required");
  }
  const uint64_t tokens = request.logits.shape().dim(0);
  const uint64_t experts = request.logits.shape().dim(1);
  const uint64_t k = request.weights.shape().dim(1);
  const float* in = reinterpret_cast<const float*>(logits->data());
  const float* b = reinterpret_cast<const float*>(bias->data());
  float* w = reinterpret_cast<float*>(weights->data());
  int32_t* id = reinterpret_cast<int32_t*>(ids->data());
  std::vector<uint32_t> order(experts);
  for (uint64_t token = 0; token < tokens; ++token) {
    std::iota(order.begin(), order.end(), 0U);
    const float* row = in + token * experts;
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t c) { return row[a] + b[a] > row[c] + b[c]; });
    float total = 0.0F;
    for (uint64_t index = 0; index < k; ++index) {
      const float weight = request.routed_scaling_factor / (1.0F + std::exp(-row[order[index]]));
      w[token * k + index] = weight;
      id[token * k + index] = static_cast<int32_t>(order[index]);
      total += weight;
    }
    if (request.renormalize && total > 0.0F) {
      for (uint64_t index = 0; index < k; ++index) w[token * k + index] /= total;
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx::kernels
