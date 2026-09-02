#include "inferx/runtime/cpu_execution_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/artifact_tensor.h"
#include "inferx/artifacts/mapped_region.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/base/checked_math.h"
#include "inferx/base/clock.h"
#include "inferx/base/id.h"
#include "inferx/base/status.h"
#include "inferx/base/token.h"
#include "inferx/input/model_package.h"
#include "inferx/model/forward_batch.h"
#include "inferx/model/llama_for_causal_lm.h"
#include "inferx/model/model_spec.h"
#include "inferx/model/model_weights.h"
#include "inferx/model/parameter_spec.h"
#include "inferx/model/weight_plan.h"
#include "inferx/ops/cpu_op_executor.h"
#include "inferx/runtime/contiguous_kv_cache.h"
#include "inferx/runtime/execution_backend.h"
#include "inferx/runtime/model_handle.h"
#include "inferx/runtime/model_memory_plan.h"
#include "inferx/scheduler/step_plan.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/device.h"
#include "inferx/tensor/dtype.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx::runtime {
namespace {

using model::ActivationBuffers;
using model::ForwardBatch;
using model::LayerWeights;

absl::Status BackendError(absl::StatusCode code, std::string_view detail) {
  return absl::Status(code, absl::StrCat("cpu_backend: ", detail));
}

// FP16 (IEEE 754 binary16) -> FP32.
float DecodeFp16(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
  const uint32_t exponent = (bits >> 10) & 0x1F;
  const uint32_t fraction = bits & 0x3FF;
  uint32_t result = 0;
  if (exponent == 0) {
    if (fraction != 0) {
      uint32_t fraction32 = fraction;
      uint32_t shift = 0;
      while ((fraction32 & 0x400) == 0) {
        fraction32 <<= 1;
        ++shift;
      }
      result = sign | ((112 - shift + 127) << 23) | ((fraction32 & 0x3FF) << 13);
    } else {
      result = sign;  // +/-0
    }
  } else if (exponent == 0x1F) {
    result = sign | 0x7F800000 | (fraction << 13);  // Inf / NaN
  } else {
    result = sign | ((exponent - 15 + 127) << 23) | (fraction << 13);
  }
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

// BF16 (truncated FP32) -> FP32.
float DecodeBf16(uint16_t bits) {
  const uint32_t result = static_cast<uint32_t>(bits) << 16;
  float value = 0.0F;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

void ConvertToFp32(std::span<const std::byte> source, artifacts::ArtifactDtype dtype,
                   std::span<float> destination) {
  const size_t count = destination.size();
  if (dtype == artifacts::ArtifactDtype::kF16 || dtype == artifacts::ArtifactDtype::kBf16) {
    const auto* raw = reinterpret_cast<const uint16_t*>(source.data());
    for (size_t index = 0; index < count; ++index) {
      destination[index] =
          dtype == artifacts::ArtifactDtype::kF16 ? DecodeFp16(raw[index]) : DecodeBf16(raw[index]);
    }
  } else {
    std::memcpy(destination.data(), source.data(), count * sizeof(float));
  }
}

struct HostTensor {
  Buffer buffer;
  MutableTensorView view;
};

absl::StatusOr<HostTensor> MakeHostTensor(uint64_t bytes, uint64_t alignment,
                                          MemoryCategory category, Dtype dtype,
                                          std::span<const uint64_t> shape) {
  CpuAllocator allocator;
  auto buffer = allocator.Allocate(AllocationRequest{
      Device::Host(), MemoryKind::kHost, ByteCount(bytes), ByteCount(alignment), category});
  if (!buffer.ok()) return buffer.status();
  const auto created_shape = Shape::Create(shape);
  if (!created_shape.ok()) return created_shape.status();
  const auto strides = Strides::Contiguous(*created_shape);
  if (!strides.ok()) return strides.status();
  auto window = buffer->MutableView(ByteRange{ByteCount(0), ByteCount(bytes)});
  if (!window.ok()) return window.status();
  auto view = MutableTensorView::Create(*window, dtype, *created_shape, *strides);
  if (!view.ok()) return view.status();
  return HostTensor{std::move(*buffer), *view};
}

absl::StatusOr<HostTensor> MakeHostMetadata(std::span<const int32_t> values) {
  auto tensor =
      MakeHostTensor(values.size() * sizeof(int32_t), 64, MemoryCategory::kExecutionMetadata,
                     Dtype::kInt32, std::array<uint64_t, 1>{values.size()});
  if (!tensor.ok()) return tensor.status();
  auto bytes = tensor->view.buffer().HostBytes();
  if (!bytes.ok()) return bytes.status();
  std::memcpy(bytes->data(), values.data(), values.size() * sizeof(int32_t));
  return tensor;
}

// Contiguous mutable view at a byte offset inside a host buffer.
absl::StatusOr<MutableTensorView> ViewAt(Buffer& buffer, uint64_t offset, uint64_t bytes,
                                         Dtype dtype, std::span<const uint64_t> shape) {
  auto window = buffer.MutableView(ByteRange{ByteCount(offset), ByteCount(bytes)});
  if (!window.ok()) return window.status();
  const auto created_shape = Shape::Create(shape);
  if (!created_shape.ok()) return created_shape.status();
  const auto strides = Strides::Contiguous(*created_shape);
  if (!strides.ok()) return strides.status();
  return MutableTensorView::Create(*window, dtype, *created_shape, *strides);
}

absl::StatusOr<float*> HostFloats(const MutableTensorView& view) {
  auto bytes = view.buffer().HostBytes();
  if (!bytes.ok()) return bytes.status();
  return reinterpret_cast<float*>(bytes->data() + view.byte_offset().value());
}

// Expected activation-plan name order (ModelMemoryPlanner emits exactly this
// sequence; drift is an internal error, never silent).
constexpr std::array<std::string_view, 12> kActivationOrder = {
    "hidden",    "normed", "query", "key",       "value",    "attn_out",
    "projected", "gate",   "up",    "activated", "down_out", "final_norm"};

}  // namespace

// One loaded model: weights, contiguous KV, activations, and the semantic
// composition. All storage is host FP32 (reference family).
struct CpuExecutionBackend::Instance {
  ModelHandle handle;
  std::optional<model::ModelSpec> spec;
  std::unique_ptr<model::LlamaForCausalLM> semantics;
  std::optional<model::ModelWeights> weights;
  ModelMemoryPlan memory_plan;
  std::optional<HostTensor> weights_arena;
  std::optional<HostTensor> kv_storage;
  std::optional<ContiguousKvCache> cache;
  std::optional<HostTensor> activations;
  std::optional<HostTensor> logits;  // [1, V] FP32
  ops::CpuOpExecutor executor;
  std::vector<int32_t> eos_token_ids;
  std::vector<uint64_t> activation_offsets;  // bytes, plan order
  uint64_t max_prefill_tokens = 0;

  [[nodiscard]] const model::LlamaSpec& llama() const { return spec->llama(); }
};

struct CpuExecutionBackend::Impl {
  std::unique_ptr<Instance> instance;
  uint64_t next_ticket = 1;

  struct InFlight {
    ExecutionTicket ticket;
    RequestId request;
    SequenceId sequence;
    RequestEpoch epoch;
    WorkKind kind;
    TokenRange range;
    scheduler::StepPlanLease lease;
    ContiguousKvCache::AppendTransaction transaction;
    bool completion_ready = false;
    ExecutionCompletion completion;
  };
  std::optional<InFlight> in_flight;
};

CpuExecutionBackend::CpuExecutionBackend() = default;

CpuExecutionBackend::~CpuExecutionBackend() = default;

BackendCapabilities CpuExecutionBackend::capabilities() const {
  return BackendCapabilities{"cpu-reference", DeviceKind::kHost, true};
}

absl::Status CpuExecutionBackend::BuildActivationBuffers(Instance& instance, uint64_t tokens,
                                                         ActivationBuffers& buffers) {
  if (instance.memory_plan.activations.size() != kActivationOrder.size()) {
    return BackendError(absl::StatusCode::kInternal, "activation plan order drift");
  }
  for (size_t index = 0; index < kActivationOrder.size(); ++index) {
    if (instance.memory_plan.activations[index].name != kActivationOrder[index]) {
      return BackendError(absl::StatusCode::kInternal, "activation plan order drift");
    }
  }
  const model::LlamaSpec& llama = instance.llama();
  const uint64_t hidden = llama.hidden_size;
  const uint64_t query = llama.num_attention_heads * llama.head_dim;
  const uint64_t qkv = llama.num_key_value_heads * llama.head_dim;
  const uint64_t intermediate = llama.intermediate_size;

  // Shapes in kActivationOrder order; `key` rides last in the plan.
  const std::array<std::pair<uint64_t, uint64_t>, 12> shapes = {
      std::make_pair(tokens, hidden),        // hidden
      std::make_pair(tokens, hidden),        // normed
      std::make_pair(tokens, query),         // query
      std::make_pair(tokens, qkv),           // key
      std::make_pair(tokens, qkv),           // value
      std::make_pair(tokens, query),         // attn_out
      std::make_pair(tokens, hidden),        // projected
      std::make_pair(tokens, intermediate),  // gate
      std::make_pair(tokens, intermediate),  // up
      std::make_pair(tokens, intermediate),  // activated
      std::make_pair(tokens, hidden),        // down_out
      std::make_pair(tokens, hidden),        // final_norm
  };
  std::array<MutableTensorView*, 12> destinations = {
      &buffers.hidden, &buffers.normed,    &buffers.query,     &buffers.key,
      &buffers.value,  &buffers.attn_out,  &buffers.projected, &buffers.gate,
      &buffers.up,     &buffers.activated, &buffers.down_out,  &buffers.final_norm,
  };
  // Head-major buffers (query/key/value/attn_out) are rank 3 [T, heads, D]
  // for RoPE/attention; the rest are rank 2 [T, width] (m5.md section 8.2).
  const std::array<bool, 12> head_major = {
      false, false, true, true, true, true, false, false, false, false, false, false,
  };
  for (size_t index = 0; index < shapes.size(); ++index) {
    const uint64_t rows = shapes[index].first;
    const uint64_t width = shapes[index].second;
    const uint64_t bytes = rows * width * 4;
    const uint64_t heads =
        (index == 2 || index == 5) ? llama.num_attention_heads : llama.num_key_value_heads;
    absl::StatusOr<MutableTensorView> view = [&]() -> absl::StatusOr<MutableTensorView> {
      if (head_major[index]) {
        const std::array<uint64_t, 3> dims{rows, heads, llama.head_dim};
        return ViewAt(instance.activations->buffer, instance.activation_offsets[index], bytes,
                      Dtype::kFloat32, dims);
      }
      const std::array<uint64_t, 2> dims{rows, width};
      return ViewAt(instance.activations->buffer, instance.activation_offsets[index], bytes,
                    Dtype::kFloat32, dims);
    }();
    if (!view.ok()) return view.status();
    *destinations[index] = *view;
  }
  buffers.logits = instance.logits->view;
  return absl::OkStatus();
}

absl::StatusOr<ModelHandle> CpuExecutionBackend::Load(const ModelLoadPlan& plan) {
  if (impl_ != nullptr && impl_->instance != nullptr) {
    return BackendError(absl::StatusCode::kFailedPrecondition,
                        "M5 supports one loaded model; unload first");
  }
  if (plan.package == nullptr || plan.model_root.empty()) {
    return BackendError(absl::StatusCode::kInvalidArgument,
                        "load requires a validated package and model root");
  }
  const input::ValidatedModelPackage& package = *plan.package;

  // Plan before allocate.
  MemoryPlanRequest request;
  request.spec = &package.model_spec();
  request.weight_plan = &package.weight_plan();
  request.max_prefill_tokens = plan.max_prefill_tokens;
  request.context_capacity = plan.context_capacity;
  request.weight_alignment = plan.weight_alignment_bytes;
  auto memory = ModelMemoryPlanner::Plan(request);
  if (!memory.ok()) return memory.status();
  const model::LlamaSpec& llama = package.model_spec().llama();

  auto session = artifacts::ModelLocator::OpenLocal(std::filesystem::path(plan.model_root));
  if (!session.ok()) return session.status();
  artifacts::ShardMappingPool mappings(artifacts::ArtifactLimits{});

  // Weight arena: one host allocation; every floating source dtype
  // materializes as FP32 at its planned offset (reference family).
  auto arena = MakeHostTensor(memory->weight_arena_bytes, plan.weight_alignment_bytes,
                              MemoryCategory::kModelWeights, Dtype::kFloat32,
                              std::array<uint64_t, 1>{memory->weight_arena_bytes / 4});
  if (!arena.ok()) return arena.status();

  std::vector<TensorView> placed_views(package.weight_plan().items.size());
  size_t placement_index = 0;
  for (const model::WeightPlanItem& item : package.weight_plan().items) {
    const WeightPlacement& placed = memory->weights.at(placement_index);
    auto file = session->OpenRegular(item.source.shard);
    if (!file.ok()) return file.status();
    auto mapping = mappings.Map(*file, item.source.file_range);
    if (!mapping.ok()) {
      return mapping.status();
    }
    uint64_t elements = 1;
    for (const uint64_t dim : item.source.shape) elements *= dim;

    auto destination =
        ViewAt(arena->buffer, placed.offset, placed.bytes, Dtype::kFloat32, item.source.shape);
    if (!destination.ok()) return destination.status();
    auto destination_data = HostFloats(*destination);
    if (!destination_data.ok()) return destination_data.status();
    ConvertToFp32(mapping->bytes(), item.source.dtype,
                  std::span<float>(*destination_data, static_cast<size_t>(elements)));
    placed_views[placement_index] = destination->AsConst();
    ++placement_index;
  }

  // Resolve the semantic weight table through the parameter catalog.
  const auto view_of = [&](model::ParameterId id) -> absl::StatusOr<TensorView> {
    for (size_t index = 0; index < package.weight_plan().items.size(); ++index) {
      if (package.weight_plan().items[index].parameter == id) return placed_views[index];
    }
    return BackendError(absl::StatusCode::kInternal, "parameter without a placement");
  };

  std::vector<LayerWeights> layers(llama.num_hidden_layers);
  TensorView embedding;
  TensorView final_norm;
  TensorView lm_head;
  for (const model::ParameterSpec& parameter : package.parameters()) {
    // Aliases share the target's storage exactly (tied LM head).
    absl::StatusOr<TensorView> view =
        view_of(parameter.alias_target.has_value() ? *parameter.alias_target : parameter.id);
    if (!view.ok()) return view.status();
    if (parameter.logical_layer.has_value()) {
      LayerWeights& layer = layers[*parameter.logical_layer];
      switch (parameter.role) {
        case model::ParameterRole::kAttentionNorm:
          layer.input_norm = *view;
          break;
        case model::ParameterRole::kAttentionQuery:
          layer.query = *view;
          break;
        case model::ParameterRole::kAttentionKey:
          layer.key = *view;
          break;
        case model::ParameterRole::kAttentionValue:
          layer.value = *view;
          break;
        case model::ParameterRole::kAttentionOutput:
          layer.attention_output = *view;
          break;
        case model::ParameterRole::kMlpNorm:
          layer.post_norm = *view;
          break;
        case model::ParameterRole::kMlpGate:
          layer.mlp_gate = *view;
          break;
        case model::ParameterRole::kMlpUp:
          layer.mlp_up = *view;
          break;
        case model::ParameterRole::kMlpDown:
          layer.mlp_down = *view;
          break;
        default:
          return BackendError(absl::StatusCode::kInvalidArgument,
                              "unexpected per-layer parameter role");
      }
    } else {
      switch (parameter.role) {
        case model::ParameterRole::kTokenEmbedding:
          embedding = *view;
          break;
        case model::ParameterRole::kFinalNorm:
          final_norm = *view;
          break;
        case model::ParameterRole::kLmHead:
          lm_head = *view;
          break;
        default:
          return BackendError(absl::StatusCode::kInvalidArgument,
                              "unexpected global parameter role");
      }
    }
  }
  auto weights =
      model::ModelWeights::Create(llama, embedding, final_norm, lm_head, std::move(layers));
  if (!weights.ok()) return weights.status();

  // Contiguous KV storage: one range, K arena then V arena, one view per
  // layer over each [1, C, Nkv, D] slice.
  const uint64_t kv_half = memory->kv_bytes / 2;
  auto kv = MakeHostTensor(memory->kv_bytes, memory->kv_alignment, MemoryCategory::kKvCache,
                           Dtype::kFloat32, std::array<uint64_t, 1>{memory->kv_bytes / 4});
  if (!kv.ok()) return kv.status();
  const uint64_t layer_bytes = kv_half / llama.num_hidden_layers;
  std::vector<MutableTensorView> keys(llama.num_hidden_layers);
  std::vector<MutableTensorView> values(llama.num_hidden_layers);
  for (uint32_t layer = 0; layer < llama.num_hidden_layers; ++layer) {
    const std::array<uint64_t, 4> dims{1, plan.context_capacity, llama.num_key_value_heads,
                                       llama.head_dim};
    const uint64_t k_begin = layer * layer_bytes;
    const uint64_t v_begin = kv_half + layer * layer_bytes;
    auto key = ViewAt(kv->buffer, k_begin, layer_bytes, Dtype::kFloat32, dims);
    auto value = ViewAt(kv->buffer, v_begin, layer_bytes, Dtype::kFloat32, dims);
    if (!key.ok()) return key.status();
    if (!value.ok()) return value.status();
    keys[layer] = *key;
    values[layer] = *value;
  }
  auto cache =
      ContiguousKvCache::Create(llama.num_hidden_layers, plan.context_capacity, keys, values);
  if (!cache.ok()) return cache.status();

  // Activations: one buffer with per-name offsets at maximum prefill tokens.
  auto activations =
      MakeHostTensor(memory->activation_bytes, 64, MemoryCategory::kWorkspace, Dtype::kFloat32,
                     std::array<uint64_t, 1>{memory->activation_bytes / 4});
  if (!activations.ok()) return activations.status();
  std::vector<uint64_t> offsets;
  offsets.reserve(memory->activations.size());
  uint64_t offset = 0;
  for (const ActivationBufferPlan& buffer : memory->activations) {
    offsets.push_back(offset);
    offset += buffer.elements * 4;
  }

  auto logits = MakeHostTensor(llama.vocab_size * 4, 64, MemoryCategory::kExecutionMetadata,
                               Dtype::kFloat32, std::array<uint64_t, 2>{1, llama.vocab_size});
  if (!logits.ok()) return logits.status();

  auto semantics = model::LlamaForCausalLM::Create(package.model_spec());
  if (!semantics.ok()) return semantics.status();

  auto instance = std::make_unique<Instance>();
  instance->handle = ModelHandle{ModelSlotId(0), ModelGeneration(1)};
  instance->spec = package.model_spec();
  instance->semantics = std::make_unique<model::LlamaForCausalLM>(std::move(*semantics));
  instance->weights = std::move(*weights);
  instance->memory_plan = std::move(*memory);
  instance->weights_arena = std::move(*arena);
  instance->kv_storage = std::move(*kv);
  instance->cache = std::move(*cache);
  instance->activations = std::move(*activations);
  instance->logits = std::move(*logits);
  instance->activation_offsets = std::move(offsets);
  instance->max_prefill_tokens = plan.max_prefill_tokens;

  // EOS ids: model-declared stop ids first, then the tokenizer's EOS.
  for (const int32_t eos : llama.eos_token_ids) instance->eos_token_ids.push_back(eos);
  const auto& metadata = package.tokenizer_metadata();
  if (metadata.eos_id.has_value() &&
      std::find(instance->eos_token_ids.begin(), instance->eos_token_ids.end(),
                metadata.eos_id->value()) == instance->eos_token_ids.end()) {
    instance->eos_token_ids.push_back(metadata.eos_id->value());
  }

  // Warm-up: one deterministic prefill token through the complete semantic
  // path with a dedicated cache sequence, then reset. Readiness publishes
  // only after it succeeds (m5.md section 10.3).
  {
    absl::Status status = instance->cache->Acquire(SequenceId(1));
    if (!status.ok()) return status;
    auto append = instance->cache->PrepareAppend(0, 1, ExecutionTicketId(0));
    if (!append.ok()) return append.status();
    ActivationBuffers buffers;
    status = BuildActivationBuffers(*instance, 1, buffers);
    if (!status.ok()) return status;
    const std::vector<int32_t> warm_token{0};
    const std::vector<int32_t> warm_positions{0};
    auto tokens_tensor = MakeHostMetadata(warm_token);
    auto positions_tensor = MakeHostMetadata(warm_positions);
    if (!tokens_tensor.ok() || !positions_tensor.ok()) {
      return BackendError(absl::StatusCode::kInternal, "warm-up metadata allocation");
    }
    ForwardBatch batch;
    batch.work = WorkKind::kPrefill;
    batch.num_tokens = 1;
    batch.token_ids = tokens_tensor->view.AsConst();
    batch.positions = positions_tensor->view.AsConst();
    batch.host_token_ids = warm_token;
    batch.host_positions = warm_positions;
    batch.key_cache = instance->cache->key_views();
    batch.value_cache = instance->cache->value_views();
    batch.kv_append_begin = 0;
    batch.activations = &buffers;
    status = instance->semantics->Forward(batch, *instance->weights, instance->executor);
    if (!status.ok()) return status;
    append->Commit();
    status = instance->cache->Reset();
    if (!status.ok()) return status;
  }

  impl_ = std::make_unique<Impl>();
  impl_->instance = std::move(instance);
  return impl_->instance->handle;
}

std::span<const int32_t> CpuExecutionBackend::stop_token_ids(ModelHandle handle) const {
  if (impl_ == nullptr || impl_->instance == nullptr || impl_->instance->handle != handle) {
    return {};
  }
  return impl_->instance->eos_token_ids;
}

absl::Status CpuExecutionBackend::Unload(ModelHandle handle) {
  if (impl_ == nullptr || impl_->instance == nullptr || impl_->instance->handle != handle) {
    return BackendError(absl::StatusCode::kNotFound, "unknown model handle");
  }
  if (impl_->in_flight.has_value()) {
    return BackendError(absl::StatusCode::kFailedPrecondition, "unload with an in-flight ticket");
  }
  absl::Status status = impl_->instance->cache->BeginClosing();
  if (!status.ok()) return status;
  impl_->instance.reset();
  impl_.reset();
  return absl::OkStatus();
}

absl::StatusOr<ExecutionTicket> CpuExecutionBackend::Submit(scheduler::StepPlanLease plan,
                                                            ModelHandle model,
                                                            std::span<const int32_t> step_tokens) {
  if (impl_ == nullptr || impl_->instance == nullptr) {
    return BackendError(absl::StatusCode::kFailedPrecondition, "no loaded model");
  }
  if (impl_->instance->handle != model) {
    return BackendError(absl::StatusCode::kNotFound, "stale model handle");
  }
  if (impl_->in_flight.has_value()) {
    return BackendError(absl::StatusCode::kFailedPrecondition, "one in-flight ticket at most");
  }
  Instance& instance = *impl_->instance;
  const scheduler::StepPlan& step = plan.plan();
  if (step.sequences.size() != 1) {
    return BackendError(absl::StatusCode::kUnimplemented, "exactly one sequence per step");
  }
  const scheduler::ScheduledSequence& item = step.sequences.front();
  const uint64_t begin = item.input_tokens.begin.value();
  const uint64_t count = item.input_tokens.end.value() - begin;
  if (step_tokens.size() != count) {
    return BackendError(absl::StatusCode::kInvalidArgument, "step token count mismatch");
  }
  for (const int32_t token : step_tokens) {
    if (token < 0 || static_cast<uint64_t>(token) >= instance.llama().vocab_size) {
      return BackendError(absl::StatusCode::kInvalidArgument, "token id outside the vocabulary");
    }
  }

  // Cache sequence transition: a fresh request prefills from zero length.
  absl::Status status = absl::OkStatus();
  if (item.kind == WorkKind::kPrefill) {
    if (count > instance.max_prefill_tokens) {
      return BackendError(absl::StatusCode::kInvalidArgument, "prefill exceeds the envelope");
    }
    const bool fresh_sequence = instance.cache->state() != ContiguousKvCache::State::kActive ||
                                instance.cache->sequence() != item.sequence;
    if (fresh_sequence || instance.cache->committed_length() != 0) {
      if (instance.cache->state() == ContiguousKvCache::State::kActive) {
        status = instance.cache->Reset();
        if (!status.ok()) return status;
      }
      status = instance.cache->Acquire(item.sequence);
      if (!status.ok()) return status;
    }
  } else if (count != 1) {
    return BackendError(absl::StatusCode::kInvalidArgument, "decode carries exactly one token");
  }
  if (instance.cache->sequence() != item.sequence) {
    return BackendError(absl::StatusCode::kFailedPrecondition, "cache sequence mismatch");
  }
  if (instance.cache->committed_length() != begin) {
    return BackendError(absl::StatusCode::kFailedPrecondition, "append boundary mismatch");
  }

  auto transaction = instance.cache->PrepareAppend(begin, count, ExecutionTicketId(0));
  if (!transaction.ok()) return transaction.status();

  ActivationBuffers buffers;
  status = BuildActivationBuffers(instance, count, buffers);
  if (!status.ok()) return status;

  // Ticket-owned token/position metadata (host for the reference backend).
  std::vector<int32_t> tokens(step_tokens.begin(), step_tokens.end());
  std::vector<int32_t> positions;
  positions.reserve(tokens.size());
  for (uint64_t index = 0; index < tokens.size(); ++index) {
    positions.push_back(static_cast<int32_t>(begin + index));
  }
  auto tokens_tensor = MakeHostMetadata(tokens);
  auto positions_tensor = MakeHostMetadata(positions);
  if (!tokens_tensor.ok() || !positions_tensor.ok()) {
    transaction->Rollback();
    return BackendError(absl::StatusCode::kInternal, "metadata allocation");
  }

  ForwardBatch batch;
  batch.work = item.kind;
  batch.num_tokens = count;
  batch.token_ids = tokens_tensor->view.AsConst();
  batch.positions = positions_tensor->view.AsConst();
  batch.host_token_ids = tokens;
  batch.host_positions = positions;
  batch.key_cache = instance.cache->key_views();
  batch.value_cache = instance.cache->value_views();
  batch.kv_append_begin = begin;
  batch.activations = &buffers;

  status = instance.semantics->Forward(batch, *instance.weights, instance.executor);
  if (!status.ok()) {
    transaction->Rollback();
    instance.cache->Poison();
    return status;
  }

  // Greedy argmax over the final-row logits: lowest id wins exact ties; a
  // NaN logit is a hard error with no output token (m5.md section 14.1).
  auto logits_data = HostFloats(instance.logits->view);
  if (!logits_data.ok()) {
    transaction->Rollback();
    return logits_data.status();
  }
  const float* logits = *logits_data;
  const uint64_t vocab = instance.llama().vocab_size;
  bool have_best = false;
  uint64_t best = 0;
  for (uint64_t id = 0; id < vocab; ++id) {
    if (std::isnan(logits[id])) {
      transaction->Rollback();
      return BackendError(absl::StatusCode::kDataLoss, "nonfinite logits");
    }
    if (!have_best || logits[id] > logits[best]) {
      best = id;
      have_best = true;
    }
  }
  transaction->Commit();

  const uint64_t ticket_value = impl_->next_ticket++;
  ExecutionTicket ticket{ExecutionTicketId(ticket_value), step.id, 1};

  ExecutionCompletion completion;
  completion.ticket = ticket.id;
  completion.step = ticket.step;
  completion.item_ordinal = 0;
  completion.item_count = 1;
  completion.request = item.request;
  completion.sequence = item.sequence;
  completion.epoch = item.epoch;
  completion.kind = item.kind;
  completion.scheduled_tokens = item.input_tokens;
  completion.status = absl::OkStatus();
  completion.error_reason = ErrorReason::kNone;
  completion.output_token = TokenId(static_cast<int32_t>(best));

  Impl::InFlight flight{ticket,    item.request,         item.sequence,   item.epoch,
                        item.kind, item.input_tokens,    std::move(plan), std::move(*transaction),
                        true,      std::move(completion)};
  impl_->in_flight = std::move(flight);
  return ticket;
}

absl::StatusOr<size_t> CpuExecutionBackend::CompleteReady(MonotonicTime /*now*/,
                                                          std::span<ExecutionCompletion> output) {
  if (impl_ == nullptr || !impl_->in_flight.has_value() || !impl_->in_flight->completion_ready) {
    return size_t{0};
  }
  if (output.empty()) {
    return BackendError(absl::StatusCode::kInvalidArgument, "empty completion span");
  }
  output[0] = impl_->in_flight->completion;
  impl_->in_flight->completion_ready = false;
  return size_t{1};
}

absl::Status CpuExecutionBackend::Acknowledge(ExecutionTicketId ticket) {
  if (impl_ == nullptr || !impl_->in_flight.has_value()) {
    return BackendError(absl::StatusCode::kFailedPrecondition, "no ticket to acknowledge");
  }
  if (impl_->in_flight->ticket.id != ticket) {
    return BackendError(absl::StatusCode::kFailedPrecondition, "ticket mismatch");
  }
  impl_->in_flight.reset();
  return absl::OkStatus();
}

}  // namespace inferx::runtime
