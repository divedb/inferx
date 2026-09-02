// CPU reference oracles: attn_res, hyperconnection, mhc, ring sconv.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "inferx/kernels/ops/model_fused.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::kernels {
namespace {

absl::Status RequireFp32(const TensorView& tensor) {
  if (tensor.dtype() != DType::kFloat32) {
    return absl::UnimplementedError("model_fused reference: FP32 tensors required");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateAttnRes(const AttnResRequest& request) {
  if (request.layer_residual.shape().rank() != 2 ||
      request.block_residual.shape().rank() != 3 ||
      request.block_residual.shape().dim(1) != request.layer_residual.shape().dim(0) ||
      request.block_residual.shape().dim(2) != request.layer_residual.shape().dim(1) ||
      request.res_weight.shape().dim(0) != request.layer_residual.shape().dim(1) ||
      request.rms_weight.shape().dim(0) != request.layer_residual.shape().dim(1)) {
    return absl::InvalidArgumentError("attn_res: incompatible shapes");
  }
  return absl::OkStatus();
}

absl::Status ReferenceAttnRes(const AttnResRequest& request) {
  absl::Status status = ValidateAttnRes(request);
  if (!status.ok()) return status;
  status = RequireFp32(request.layer_residual.AsConst());
  if (!status.ok()) return status;
  const uint64_t tokens = request.layer_residual.shape().dim(0);
  const uint64_t hidden = request.layer_residual.shape().dim(1);
  const uint64_t blocks = request.block_residual.shape().dim(0);
  auto layer = request.layer_residual.buffer().HostBytes();
  auto block = request.block_residual.buffer().HostBytes();
  auto res_w = request.res_weight.buffer().HostBytes();
  auto rms_w = request.rms_weight.buffer().HostBytes();
  std::optional<std::span<const std::byte>> out_w;
  if (request.out_norm_weight.has_value()) {
    auto bytes = request.out_norm_weight->buffer().HostBytes();
    if (!bytes.ok()) return absl::InvalidArgumentError("attn_res: host tensors required");
    out_w = *bytes;
  }
  if (!layer.ok() || !block.ok() || !res_w.ok() || !rms_w.ok()) {
    return absl::InvalidArgumentError("attn_res: host tensors required");
  }
  float* layer_data = reinterpret_cast<float*>(layer->data());
  const float* block_data = reinterpret_cast<const float*>(block->data());
  const float* res_weight = reinterpret_cast<const float*>(res_w->data());
  const float* rms_weight = reinterpret_cast<const float*>(rms_w->data());
  const float* out_norm_weight =
      out_w.has_value() ? reinterpret_cast<const float*>(out_w->data()) : nullptr;
  std::vector<const float*> candidates(blocks + 1);
  for (uint64_t token = 0; token < tokens; ++token) {
    for (uint64_t b = 0; b < blocks; ++b) {
      candidates[b] = block_data + (b * tokens + token) * hidden;
    }
    candidates[blocks] = layer_data + token * hidden;
    const uint64_t count = blocks + 1;
    std::vector<float> logits(count);
    for (uint64_t n = 0; n < count; ++n) {
      float square_sum = 0.0F;
      for (uint64_t h = 0; h < hidden; ++h) {
        square_sum += candidates[n][h] * candidates[n][h];
      }
      const float inverse_rms =
          1.0F / std::sqrt(square_sum / static_cast<float>(hidden) + request.epsilon);
      float dot = 0.0F;
      for (uint64_t h = 0; h < hidden; ++h) {
        dot += candidates[n][h] * inverse_rms * rms_weight[h] * res_weight[h];
      }
      logits[n] = dot / std::sqrt(static_cast<float>(hidden));
    }
    const float maximum = *std::max_element(logits.begin(), logits.end());
    float total = 0.0F;
    for (uint64_t n = 0; n < count; ++n) {
      logits[n] = std::exp(logits[n] - maximum);
      total += logits[n];
    }
    std::vector<float> mixed(hidden, 0.0F);
    for (uint64_t n = 0; n < count; ++n) {
      const float probability = logits[n] / total;
      for (uint64_t h = 0; h < hidden; ++h) {
        mixed[h] += probability * candidates[n][h];
      }
    }
    if (out_norm_weight != nullptr) {
      float square_sum = 0.0F;
      for (uint64_t h = 0; h < hidden; ++h) square_sum += mixed[h] * mixed[h];
      const float inverse_rms =
          1.0F / std::sqrt(square_sum / static_cast<float>(hidden) + request.out_norm_epsilon);
      for (uint64_t h = 0; h < hidden; ++h) {
        mixed[h] = mixed[h] * inverse_rms * out_norm_weight[h];
      }
    }
    for (uint64_t h = 0; h < hidden; ++h) layer_data[token * hidden + h] = mixed[h];
  }
  return absl::OkStatus();
}

absl::Status ValidateHcMix(const HcMixRequest& request) {
  const uint64_t wide = static_cast<uint64_t>(request.hc_count) * request.hidden_size;
  if (request.normalized.shape().rank() != 2 || request.normalized.shape().dim(1) != wide ||
      request.projection_weight.shape().rank() != 2 ||
      request.projection_weight.shape().dim(1) != wide ||
      request.projection_weight.shape().dim(0) < request.lowrank + request.hc_count ||
      request.up_weight.shape().dim(0) != wide ||
      request.up_weight.shape().dim(1) != request.lowrank ||
      request.mixed.shape().dim(1) != request.hidden_size ||
      request.inject_logits.shape().dim(1) != request.hc_count) {
    return absl::InvalidArgumentError("hc_mix: incompatible shapes");
  }
  return absl::OkStatus();
}

absl::Status ReferenceHcMix(const HcMixRequest& request) {
  absl::Status status = ValidateHcMix(request);
  if (!status.ok()) return status;
  status = RequireFp32(request.normalized);
  if (!status.ok()) return status;
  const uint64_t tokens = request.normalized.shape().dim(0);
  const uint64_t wide = static_cast<uint64_t>(request.hc_count) * request.hidden_size;
  auto norm = request.normalized.buffer().HostBytes();
  auto down = request.projection_weight.buffer().HostBytes();
  auto up = request.up_weight.buffer().HostBytes();
  auto mixed = request.mixed.buffer().HostBytes();
  auto inject = request.inject_logits.buffer().HostBytes();
  if (!norm.ok() || !down.ok() || !up.ok() || !mixed.ok() || !inject.ok()) {
    return absl::InvalidArgumentError("hc_mix: host tensors required");
  }
  const float* x = reinterpret_cast<const float*>(norm->data());
  const float* w_down = reinterpret_cast<const float*>(down->data());
  const float* w_up = reinterpret_cast<const float*>(up->data());
  float* mixed_out = reinterpret_cast<float*>(mixed->data());
  float* inject_out = reinterpret_cast<float*>(inject->data());
  const uint64_t rank = request.lowrank;
  std::vector<float> projected(rank + request.hc_count);
  std::vector<float> gate(wide);
  for (uint64_t token = 0; token < tokens; ++token) {
    for (uint64_t r = 0; r < rank + request.hc_count; ++r) {
      float sum = 0.0F;
      for (uint64_t h = 0; h < wide; ++h) {
        sum += w_down[r * wide + h] * x[token * wide + h];
      }
      projected[r] = sum;
    }
    for (uint64_t h = 0; h < wide; ++h) {
      float sum = 0.0F;
      for (uint64_t r = 0; r < rank; ++r) {
        const float activated =
            (request.projection_scale * projected[r]) /
            (1.0F + std::exp(-request.projection_scale * projected[r]));
        sum += w_up[h * rank + r] * activated;
      }
      gate[h] = sum;
    }
    for (uint64_t h = 0; h < request.hidden_size; ++h) {
      float sum = 0.0F;
      for (uint32_t b = 0; b < request.hc_count; ++b) {
        const float sigmoid_gate =
            1.0F / (1.0F + std::exp(-gate[b * request.hidden_size + h]));
        sum += sigmoid_gate * x[token * wide + b * request.hidden_size + h];
      }
      mixed_out[token * request.hidden_size + h] =
          sum / static_cast<float>(request.hc_count);
    }
    for (uint32_t b = 0; b < request.hc_count; ++b) {
      inject_out[token * request.hc_count + b] =
          request.projection_scale * projected[rank + b];
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateHcCombine(const HcCombineRequest& request) {
  const uint64_t wide = static_cast<uint64_t>(request.hc_count) * request.hidden_size;
  if (request.block_output.shape().rank() != 2 ||
      request.block_output.shape().dim(1) != request.hidden_size ||
      request.residual.shape().dim(1) != wide ||
      request.inject_logits.shape().dim(1) != request.hc_count ||
      request.residual.shape().dim(0) != request.block_output.shape().dim(0)) {
    return absl::InvalidArgumentError("hc_combine: incompatible shapes");
  }
  return absl::OkStatus();
}

absl::Status ReferenceHcCombine(const HcCombineRequest& request) {
  absl::Status status = ValidateHcCombine(request);
  if (!status.ok()) return status;
  status = RequireFp32(request.block_output);
  if (!status.ok()) return status;
  const uint64_t tokens = request.block_output.shape().dim(0);
  auto out = request.block_output.buffer().HostBytes();
  auto res = request.residual.buffer().HostBytes();
  auto inject = request.inject_logits.buffer().HostBytes();
  if (!out.ok() || !res.ok() || !inject.ok()) {
    return absl::InvalidArgumentError("hc_combine: host tensors required");
  }
  const float* block = reinterpret_cast<const float*>(out->data());
  float* residual = reinterpret_cast<float*>(res->data());
  const float* inj = reinterpret_cast<const float*>(inject->data());
  const uint64_t wide = static_cast<uint64_t>(request.hc_count) * request.hidden_size;
  for (uint64_t token = 0; token < tokens; ++token) {
    for (uint32_t b = 0; b < request.hc_count; ++b) {
      const float scale = 2.0F / (1.0F + std::exp(-inj[token * request.hc_count + b]));
      for (uint64_t h = 0; h < request.hidden_size; ++h) {
        residual[token * wide + b * request.hidden_size + h] +=
            scale * block[token * request.hidden_size + h];
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateMhcPre(const MhcPreRequest& request) {
  const uint64_t m = request.residual.shape().dim(1);
  const uint64_t wide = m * request.residual.shape().dim(2);
  if (request.residual.shape().rank() != 3 || m == 0 ||
      request.fn.shape().dim(0) != 2 * m + m * m || request.fn.shape().dim(1) != wide ||
      request.hc_scale.shape().dim(0) < 3 || request.hc_base.shape().dim(0) < 2 * m + m * m ||
      request.layer_input.shape().dim(1) != request.residual.shape().dim(2) ||
      request.post.shape().dim(1) != m || request.comb.shape().dim(1) != m ||
      request.comb.shape().dim(2) != m) {
    return absl::InvalidArgumentError("mhc_pre: incompatible shapes");
  }
  return absl::OkStatus();
}

absl::Status ReferenceMhcPre(const MhcPreRequest& request) {
  absl::Status status = ValidateMhcPre(request);
  if (!status.ok()) return status;
  status = RequireFp32(request.residual);
  if (!status.ok()) return status;
  const uint64_t tokens = request.residual.shape().dim(0);
  const uint64_t m = request.residual.shape().dim(1);
  const uint64_t hidden = request.residual.shape().dim(2);
  const uint64_t wide = m * hidden;
  auto res = request.residual.buffer().HostBytes();
  auto fn = request.fn.buffer().HostBytes();
  auto scale = request.hc_scale.buffer().HostBytes();
  auto base = request.hc_base.buffer().HostBytes();
  auto layer = request.layer_input.buffer().HostBytes();
  auto post = request.post.buffer().HostBytes();
  auto comb = request.comb.buffer().HostBytes();
  if (!res.ok() || !fn.ok() || !scale.ok() || !base.ok() || !layer.ok() || !post.ok() ||
      !comb.ok()) {
    return absl::InvalidArgumentError("mhc_pre: host tensors required");
  }
  const float* residual = reinterpret_cast<const float*>(res->data());
  const float* fn_data = reinterpret_cast<const float*>(fn->data());
  const float* hc_scale = reinterpret_cast<const float*>(scale->data());
  const float* hc_base = reinterpret_cast<const float*>(base->data());
  float* layer_out = reinterpret_cast<float*>(layer->data());
  float* post_out = reinterpret_cast<float*>(post->data());
  float* comb_out = reinterpret_cast<float*>(comb->data());
  std::vector<float> mix(2 * m + m * m);
  for (uint64_t token = 0; token < tokens; ++token) {
    for (uint64_t row = 0; row < 2 * m + m * m; ++row) {
      float sum = 0.0F;
      for (uint64_t index = 0; index < wide; ++index) {
        sum += fn_data[row * wide + index] * residual[token * wide + index];
      }
      mix[row] = sum;
    }
    float square_sum = 0.0F;
    for (uint64_t index = 0; index < wide; ++index) {
      square_sum += residual[token * wide + index] * residual[token * wide + index];
    }
    const float rms =
        1.0F / std::sqrt(square_sum / static_cast<float>(wide) + request.rms_eps);
    for (uint64_t i = 0; i < m; ++i) {
      layer_out[token * hidden + i] = 0.0F;  // accumulated below
    }
    std::vector<float> pre(m);
    for (uint64_t i = 0; i < m; ++i) {
      pre[i] = 1.0F / (1.0F + std::exp(-(mix[i] * rms * hc_scale[0] + hc_base[i]))) + request.hc_eps;
      post_out[token * m + i] =
          2.0F / (1.0F + std::exp(-(mix[m + i] * rms * hc_scale[1] + hc_base[m + i])));
    }
    std::vector<float> comb_raw(m * m);
    for (uint64_t i = 0; i < m * m; ++i) {
      comb_raw[i] = mix[2 * m + i] * rms * hc_scale[2] + hc_base[2 * m + i];
    }
    // row softmax
    for (uint64_t i = 0; i < m; ++i) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (uint64_t j = 0; j < m; ++j) maximum = std::max(maximum, comb_raw[i * m + j]);
      float total = 0.0F;
      for (uint64_t j = 0; j < m; ++j) {
        comb_raw[i * m + j] = std::exp(comb_raw[i * m + j] - maximum);
        total += comb_raw[i * m + j];
      }
      for (uint64_t j = 0; j < m; ++j) comb_raw[i * m + j] /= total;
    }
    // sinkhorn iterations: row-normalize then column-normalize
    for (uint32_t iteration = 0; iteration < request.sinkhorn_iters; ++iteration) {
      for (uint64_t i = 0; i < m; ++i) {
        float row_sum = 0.0F;
        for (uint64_t j = 0; j < m; ++j) row_sum += comb_raw[i * m + j];
        for (uint64_t j = 0; j < m; ++j) {
          comb_raw[i * m + j] = comb_raw[i * m + j] / (row_sum + request.hc_eps);
        }
      }
      for (uint64_t j = 0; j < m; ++j) {
        float column_sum = 0.0F;
        for (uint64_t i = 0; i < m; ++i) column_sum += comb_raw[i * m + j];
        for (uint64_t i = 0; i < m; ++i) {
          comb_raw[i * m + j] = comb_raw[i * m + j] / (column_sum + request.hc_eps);
        }
      }
    }
    for (uint64_t i = 0; i < m; ++i) {
      for (uint64_t j = 0; j < m; ++j) {
        comb_out[token * m * m + i * m + j] = comb_raw[i * m + j];
      }
    }
    for (uint64_t h = 0; h < hidden; ++h) {
      float sum = 0.0F;
      for (uint64_t i = 0; i < m; ++i) {
        sum += pre[i] * residual[token * wide + i * hidden + h];
      }
      layer_out[token * hidden + h] = sum;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateMhcPost(const MhcPostRequest& request) {
  const uint64_t m = request.residual.shape().dim(1);
  if (request.hidden_states.shape().rank() != 2 ||
      request.residual.shape().rank() != 3 ||
      request.hidden_states.shape().dim(1) != request.residual.shape().dim(2) ||
      request.post.shape().dim(1) != m || request.comb.shape().dim(1) != m ||
      request.comb.shape().dim(2) != m) {
    return absl::InvalidArgumentError("mhc_post: incompatible shapes");
  }
  return absl::OkStatus();
}

absl::Status ReferenceMhcPost(const MhcPostRequest& request) {
  absl::Status status = ValidateMhcPost(request);
  if (!status.ok()) return status;
  status = RequireFp32(request.hidden_states);
  if (!status.ok()) return status;
  const uint64_t tokens = request.hidden_states.shape().dim(0);
  const uint64_t streams = request.residual.shape().dim(1);
  const uint64_t hidden = request.residual.shape().dim(2);
  auto hidden_bytes = request.hidden_states.buffer().HostBytes();
  auto res = request.residual.buffer().HostBytes();
  auto post = request.post.buffer().HostBytes();
  auto comb = request.comb.buffer().HostBytes();
  if (!hidden_bytes.ok() || !res.ok() || !post.ok() || !comb.ok()) {
    return absl::InvalidArgumentError("mhc_post: host tensors required");
  }
  const float* hidden_states = reinterpret_cast<const float*>(hidden_bytes->data());
  float* residual = reinterpret_cast<float*>(res->data());
  const float* post_data = reinterpret_cast<const float*>(post->data());
  const float* comb_data = reinterpret_cast<const float*>(comb->data());
  std::vector<float> updated(streams * hidden);
  for (uint64_t token = 0; token < tokens; ++token) {
    for (uint64_t j = 0; j < streams; ++j) {
      for (uint64_t h = 0; h < hidden; ++h) {
        float sum = post_data[token * streams + j] * hidden_states[token * hidden + h];
        for (uint64_t i = 0; i < streams; ++i) {
          sum += comb_data[token * streams * streams + i * streams + j] *
                 residual[token * streams * hidden + i * hidden + h];
        }
        updated[j * hidden + h] = sum;
      }
    }
    for (uint64_t index = 0; index < streams * hidden; ++index) {
      residual[token * streams * hidden + index] = updated[index];
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateRingSconv(const RingSconvRequest& request) {
  if (request.x.shape().rank() != 2 || request.weight.shape().rank() != 2 ||
      request.weight.shape().dim(1) != request.window ||
      request.conv_cache.shape().rank() != 3 ||
      request.conv_cache.shape().dim(2) != request.x.shape().dim(1) ||
      request.conv_cache.shape().dim(1) < request.window ||
      request.output.shape() != request.x.shape() ||
      request.seq_lens.shape().dim(1) != request.cache_indices.shape().dim(1)) {
    return absl::InvalidArgumentError("ring_sconv: incompatible shapes");
  }
  return absl::OkStatus();
}

absl::Status ReferenceRingSconv(const RingSconvRequest& request) {
  absl::Status status = ValidateRingSconv(request);
  if (!status.ok()) return status;
  status = RequireFp32(request.x);
  if (!status.ok()) return status;
  const uint64_t width = request.x.shape().dim(1);
  const uint64_t batch = request.seq_lens.shape().dim(0);
  const uint64_t ring = request.conv_cache.shape().dim(1);
  auto x_bytes = request.x.buffer().HostBytes();
  auto w_bytes = request.weight.buffer().HostBytes();
  auto c_bytes = request.conv_cache.buffer().HostBytes();
  auto l_bytes = request.seq_lens.buffer().HostBytes();
  auto i_bytes = request.cache_indices.buffer().HostBytes();
  auto o_bytes = request.output.buffer().HostBytes();
  if (!x_bytes.ok() || !w_bytes.ok() || !c_bytes.ok() || !l_bytes.ok() || !i_bytes.ok() ||
      !o_bytes.ok()) {
    return absl::InvalidArgumentError("ring_sconv: host tensors required");
  }
  const float* x = reinterpret_cast<const float*>(x_bytes->data());
  const float* weight = reinterpret_cast<const float*>(w_bytes->data());
  float* cache = reinterpret_cast<float*>(c_bytes->data());
  const int32_t* seq_lens = reinterpret_cast<const int32_t*>(l_bytes->data());
  const int32_t* cache_indices = reinterpret_cast<const int32_t*>(i_bytes->data());
  float* output = reinterpret_cast<float*>(o_bytes->data());
  uint64_t position = 0;
  for (uint64_t sequence = 0; sequence < batch; ++sequence) {
    const uint64_t slot = static_cast<uint64_t>(cache_indices[sequence]);
    float* ring_cache = cache + static_cast<uint64_t>(slot) * ring * width;
    for (int32_t step = 0; step < seq_lens[sequence]; ++step, ++position) {
      const uint64_t ustep = static_cast<uint64_t>(step);
      for (uint64_t channel = 0; channel < width; ++channel) {
        float sum = weight[channel * request.window + request.window - 1] *
                    x[position * width + channel];
        for (uint32_t lag = 1; lag < request.window; ++lag) {
          const uint64_t ring_position = (ustep + ring - lag) % ring;
          sum += weight[channel * request.window + request.window - 1 - lag] *
                 ring_cache[ring_position * width + channel];
        }
        output[position * width + channel] = sum;
      }
      const uint64_t write_position = ustep % ring;
      for (uint64_t channel = 0; channel < width; ++channel) {
        ring_cache[write_position * width + channel] = x[position * width + channel];
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx::kernels
