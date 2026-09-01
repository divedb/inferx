#include "inferx/simulator/replay.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/version.h"
#include "inferx/config/parser_limits.h"
#include "inferx/lifecycle/request_state.h"
#include "inferx/scheduler/work_kind.h"
#include "simdjson.h"

namespace inferx::simulator {
namespace {

absl::Status ReplayError(absl::StatusCode code, absl::string_view detail,
                         ErrorReason reason = ErrorReason::kReplayMismatch) {
  return WithErrorReason(absl::Status(code, absl::StrCat("replay: ", detail)), reason);
}

std::set<std::string> AllowedFields(absl::string_view type) {
  const std::set<std::string> common{"schema_version", "record_type", "ordinal"};
  std::set<std::string> fields = common;
  if (type == "header") {
    fields.insert({"inferx_version", "config", "fake_model", "workload_event_count"});
  } else if (type == "workload_event") {
    fields.insert({"at_ns", "event_type", "request_id", "model_id", "token_ids",
                   "max_output_tokens", "deadline_ns", "finish_after_tokens", "priority",
                   "tenant_scope", "target_kind", "target_id", "status_code", "error_reason",
                   "persistent", "mode"});
  } else if (type == "request_transition") {
    fields.insert({"at_ns", "request_id", "sequence_id", "epoch", "from", "event", "to",
                   "status_code", "error_reason"});
  } else if (type == "resource_change") {
    fields.insert({"at_ns", "action", "reservation_id", "request_id", "sequence_delta",
                   "kv_token_delta", "used_sequences", "used_kv_tokens"});
  } else if (type == "step_plan") {
    fields.insert({"at_ns", "step_id", "model_id", "buffer_slot", "buffer_generation",
                   "num_sequences", "num_tokens", "sequences"});
  } else if (type == "execution_completion") {
    fields.insert({"at_ns", "ticket_id", "step_id", "request_id", "sequence_id", "epoch", "kind",
                   "token_begin", "token_end", "status_code", "error_reason", "output_token"});
  } else if (type == "response_event") {
    fields.insert({"at_ns", "request_id", "event_kind", "output_position", "token_id",
                   "finish_reason", "status_code", "error_reason", "prompt_tokens",
                   "output_tokens"});
  } else if (type == "idle") {
    fields.insert({"at_ns", "idle_reason", "next_event_ns"});
  } else if (type == "footer") {
    fields.insert({"final_status_code", "error_reason", "events", "steps", "requests", "finished",
                   "cancelled", "failed", "used_sequences", "used_kv_tokens", "live_tickets",
                   "leased_plan_slots", "invariants_ok"});
  }
  return fields;
}

std::set<std::string> RequiredFields(absl::string_view type) {
  std::set<std::string> fields{"schema_version", "record_type", "ordinal"};
  if (type == "header") {
    fields.insert({"inferx_version", "config", "fake_model", "workload_event_count"});
  } else if (type == "workload_event") {
    fields.insert({"at_ns", "event_type"});
  } else if (type == "request_transition") {
    fields.insert({"at_ns", "request_id", "sequence_id", "epoch", "from", "event", "to",
                   "status_code", "error_reason"});
  } else if (type == "resource_change") {
    fields.insert({"at_ns", "action", "reservation_id", "request_id", "sequence_delta",
                   "kv_token_delta", "used_sequences", "used_kv_tokens"});
  } else if (type == "step_plan") {
    fields.insert({"at_ns", "step_id", "model_id", "buffer_slot", "buffer_generation",
                   "num_sequences", "num_tokens", "sequences"});
  } else if (type == "execution_completion") {
    fields.insert({"at_ns", "ticket_id", "step_id", "request_id", "sequence_id", "epoch", "kind",
                   "token_begin", "token_end", "status_code", "error_reason", "output_token"});
  } else if (type == "response_event") {
    fields.insert({"at_ns", "request_id", "event_kind", "output_position", "token_id",
                   "finish_reason", "status_code", "error_reason", "prompt_tokens",
                   "output_tokens"});
  } else if (type == "idle") {
    fields.insert({"at_ns", "idle_reason", "next_event_ns"});
  } else if (type == "footer") {
    fields.insert({"final_status_code", "error_reason", "events", "steps", "requests", "finished",
                   "cancelled", "failed", "used_sequences", "used_kv_tokens", "live_tickets",
                   "leased_plan_slots", "invariants_ok"});
  }
  return fields;
}

absl::Status ValidateLine(absl::string_view line, uint64_t expected_ordinal,
                          std::string* record_type) {
  simdjson::dom::parser parser;
  simdjson::dom::element root;
  try {
    const simdjson::error_code error = parser.parse(line.data(), line.size()).get(root);
    if (error != simdjson::SUCCESS) {
      return ReplayError(absl::StatusCode::kInvalidArgument,
                         absl::StrCat("invalid JSON: ", simdjson::error_message(error)));
    }
  } catch (const std::exception& exception) {
    return ReplayError(absl::StatusCode::kInvalidArgument,
                       absl::StrCat("parser threw: ", exception.what()));
  }
  simdjson::dom::object object;
  if (root.get(object) != simdjson::SUCCESS) {
    return ReplayError(absl::StatusCode::kInvalidArgument, "record must be an object");
  }
  uint64_t version = 0;
  uint64_t ordinal = 0;
  std::string_view type;
  if (object.at_key("schema_version").get(version) != simdjson::SUCCESS || version != 1 ||
      object.at_key("ordinal").get(ordinal) != simdjson::SUCCESS || ordinal != expected_ordinal ||
      object.at_key("record_type").get(type) != simdjson::SUCCESS) {
    return ReplayError(absl::StatusCode::kInvalidArgument,
                       "invalid schema_version, record_type, or ordinal");
  }
  *record_type = std::string(type);
  const std::set<std::string> allowed = AllowedFields(type);
  if (allowed.size() == 3) {
    return ReplayError(absl::StatusCode::kInvalidArgument, "unknown record type");
  }
  std::set<std::string> seen;
  for (auto [key, value] : object) {
    static_cast<void>(value);
    const std::string name(key);
    if (!seen.insert(name).second) {
      return ReplayError(absl::StatusCode::kInvalidArgument,
                         absl::StrCat("duplicate field ", name));
    }
    if (!allowed.contains(name)) {
      return ReplayError(absl::StatusCode::kInvalidArgument, absl::StrCat("unknown field ", name));
    }
  }
  for (const std::string& required : RequiredFields(type)) {
    if (!seen.contains(required)) {
      return ReplayError(absl::StatusCode::kInvalidArgument,
                         absl::StrCat("missing required field ", required));
    }
  }
  return absl::OkStatus();
}

std::string WorkloadLineFromReplay(absl::string_view line) {
  const std::string marker = ",\"at_ns\":";
  const size_t at_position = line.find(marker);
  const size_t event_position = line.find(",\"event_type\":\"", at_position);
  if (at_position == absl::string_view::npos || event_position == absl::string_view::npos) {
    return {};
  }
  const size_t at_value_begin = at_position + marker.size();
  const std::string at_value(line.substr(at_value_begin, event_position - at_value_begin));
  const size_t event_value_begin = event_position + std::string(",\"event_type\":\"").size();
  const size_t event_value_end = line.find('"', event_value_begin);
  if (event_value_end == absl::string_view::npos) return {};
  const std::string event_name(line.substr(event_value_begin, event_value_end - event_value_begin));
  const absl::string_view suffix =
      line.substr(event_value_end + 1, line.size() - event_value_end - 2);
  return absl::StrCat("{\"schema_version\":1,\"event_type\":\"", event_name,
                      "\",\"at_ns\":", at_value, suffix, "}");
}

}  // namespace

InMemoryReplaySink::InMemoryReplaySink(ReplayBufferLimits limits)
    : max_bytes_(limits.max_bytes), max_records_(limits.max_records) {
  lines_.reserve(max_records_);
}

absl::Status InMemoryReplaySink::WriteLine(absl::string_view line) {
  if (closed_) {
    return ReplayError(absl::StatusCode::kFailedPrecondition, "sink is closed",
                       ErrorReason::kIoFailure);
  }
  if (line.size() > config::kMaxJsonlRecordBytes || lines_.size() >= max_records_ ||
      line.size() + 1 > max_bytes_ - std::min(bytes_, max_bytes_)) {
    return ReplayError(absl::StatusCode::kResourceExhausted, "memory sink limit exceeded",
                       ErrorReason::kIoFailure);
  }
  lines_.emplace_back(line);
  bytes_ += line.size() + 1;
  return absl::OkStatus();
}

absl::Status InMemoryReplaySink::Close() {
  closed_ = true;
  return absl::OkStatus();
}

std::string InMemoryReplaySink::contents() const {
  std::string output;
  output.reserve(bytes_);
  for (const std::string& line : lines_) {
    output.append(line);
    output.push_back('\n');
  }
  return output;
}

struct FileReplaySink::Impl {
  std::ofstream stream;
  bool closed = false;
};

FileReplaySink::FileReplaySink(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
FileReplaySink::~FileReplaySink() = default;

absl::StatusOr<std::unique_ptr<FileReplaySink>> FileReplaySink::Create(const std::string& path,
                                                                       bool overwrite) {
  const std::filesystem::path output(path);
  if (path.empty() || output == output.root_path()) {
    return ReplayError(absl::StatusCode::kInvalidArgument, "unsafe output path",
                       ErrorReason::kIoFailure);
  }
  std::error_code error;
  if (std::filesystem::exists(output, error) && !overwrite) {
    return ReplayError(absl::StatusCode::kAlreadyExists, "output exists; use --overwrite",
                       ErrorReason::kIoFailure);
  }
  if (output.has_parent_path()) {
    std::filesystem::create_directories(output.parent_path(), error);
    if (error) {
      return ReplayError(absl::StatusCode::kUnavailable,
                         absl::StrCat("cannot create output parent: ", error.message()),
                         ErrorReason::kIoFailure);
    }
  }
  auto impl = std::make_unique<Impl>();
  impl->stream.open(output, std::ios::binary | std::ios::trunc);
  if (!impl->stream) {
    return ReplayError(absl::StatusCode::kUnavailable, "cannot open replay output",
                       ErrorReason::kIoFailure);
  }
  return std::unique_ptr<FileReplaySink>(new FileReplaySink(std::move(impl)));
}

absl::Status FileReplaySink::WriteLine(absl::string_view line) {
  if (impl_->closed) {
    return ReplayError(absl::StatusCode::kFailedPrecondition, "file sink is closed",
                       ErrorReason::kIoFailure);
  }
  if (line.size() > config::kMaxJsonlRecordBytes) {
    return ReplayError(absl::StatusCode::kResourceExhausted, "record exceeds line limit",
                       ErrorReason::kIoFailure);
  }
  impl_->stream.write(line.data(), static_cast<std::streamsize>(line.size()));
  impl_->stream.put('\n');
  if (!impl_->stream) {
    return ReplayError(absl::StatusCode::kUnavailable, "write failed", ErrorReason::kIoFailure);
  }
  return absl::OkStatus();
}

absl::Status FileReplaySink::Close() {
  if (impl_->closed) return absl::OkStatus();
  impl_->stream.flush();
  if (!impl_->stream) {
    return ReplayError(absl::StatusCode::kUnavailable, "flush failed", ErrorReason::kIoFailure);
  }
  impl_->stream.close();
  impl_->closed = true;
  return absl::OkStatus();
}

absl::string_view ToString(IdleReason reason) {
  switch (reason) {
    case IdleReason::kNoRequests:
      return "no_requests";
    case IdleReason::kWaitingForEvent:
      return "waiting_for_event";
    case IdleReason::kWaitingForCompletion:
      return "waiting_for_completion";
    case IdleReason::kCapacityBlocked:
      return "capacity_blocked";
    case IdleReason::kNoPlanBuffer:
      return "no_plan_buffer";
  }
  return "unknown";
}

ReplayWriter::ReplayWriter(ReplaySink& sink, std::string canonical_config,
                           config::ModelCapabilities model, size_t workload_event_count) noexcept
    : sink_(sink),
      canonical_config_(std::move(canonical_config)),
      model_(model),
      workload_event_count_(workload_event_count) {}

std::string ReplayWriter::Prefix(absl::string_view record_type) const {
  return absl::StrCat("{\"schema_version\":1,\"record_type\":\"", record_type,
                      "\",\"ordinal\":", next_ordinal_);
}

absl::Status ReplayWriter::Write(const std::string& line) {
  if (footer_written_) {
    return ReplayError(absl::StatusCode::kFailedPrecondition, "record after footer");
  }
  absl::Status status = sink_.WriteLine(line);
  if (status.ok()) ++next_ordinal_;
  return status;
}

absl::Status ReplayWriter::WriteHeader() {
  if (header_written_ || next_ordinal_ != 0) {
    return ReplayError(absl::StatusCode::kFailedPrecondition, "duplicate or displaced header");
  }
  header_written_ = true;
  return Write(absl::StrCat(
      Prefix("header"), ",\"inferx_version\":\"", GetVersionString(),
      "\",\"config\":", canonical_config_,
      ",\"fake_model\":{\"model_id\":0,\"max_context_tokens\":", model_.max_context_tokens.value(),
      ",\"accepts_text\":", model_.accepts_text ? "true" : "false",
      "},\"workload_event_count\":", workload_event_count_, "}"));
}

absl::Status ReplayWriter::WriteWorkload(const WorkloadEvent& event) {
  const std::string canonical = CanonicalWorkloadEvent(event);
  const std::string canonical_prefix =
      absl::StrCat("{\"schema_version\":1,\"event_type\":\"", ToString(event.type),
                   "\",\"at_ns\":", event.at.time_since_epoch().count());
  if (!canonical.starts_with(canonical_prefix) || canonical.back() != '}') {
    return ReplayError(absl::StatusCode::kInternal, "canonical workload prefix mismatch",
                       ErrorReason::kInvariantViolation);
  }
  const std::string suffix =
      canonical.substr(canonical_prefix.size(), canonical.size() - canonical_prefix.size() - 1);
  return Write(absl::StrCat(Prefix("workload_event"),
                            ",\"at_ns\":", event.at.time_since_epoch().count(),
                            ",\"event_type\":\"", ToString(event.type), "\"", suffix, "}"));
}

absl::Status ReplayWriter::ObserveTransition(const TransitionObservation& observation) {
  return Write(absl::StrCat(
      Prefix("request_transition"), ",\"at_ns\":", observation.at.time_since_epoch().count(),
      ",\"request_id\":", observation.request.value(), ",\"sequence_id\":",
      observation.sequence.value(), ",\"epoch\":", observation.epoch.value(), ",\"from\":\"",
      ToString(observation.from), "\",\"event\":\"", ToString(observation.event), "\",\"to\":\"",
      ToString(observation.to), "\",\"status_code\":", static_cast<int>(observation.status),
      ",\"error_reason\":\"", ErrorReasonToName(observation.reason), "\"}"));
}

absl::Status ReplayWriter::ObserveResource(const ResourceObservation& observation) {
  const int64_t sign = observation.action == ResourceAction::kReserve ? 1 : -1;
  return Write(absl::StrCat(
      Prefix("resource_change"), ",\"at_ns\":", observation.at.time_since_epoch().count(),
      ",\"action\":\"", observation.action == ResourceAction::kReserve ? "reserve" : "release",
      "\",\"reservation_id\":", observation.reservation.value(),
      ",\"request_id\":", observation.request.value(),
      ",\"sequence_delta\":", sign * static_cast<int64_t>(observation.cost.sequences.value()),
      ",\"kv_token_delta\":", sign * static_cast<int64_t>(observation.cost.kv_tokens.value()),
      ",\"used_sequences\":", observation.snapshot.sequences_used.value(),
      ",\"used_kv_tokens\":", observation.snapshot.kv_tokens_used.value(), "}"));
}

absl::Status ReplayWriter::WritePlan(const scheduler::StepPlan& plan) {
  std::string line =
      absl::StrCat(Prefix("step_plan"), ",\"at_ns\":", plan.planned_at.time_since_epoch().count(),
                   ",\"step_id\":", plan.id.value(), ",\"model_id\":", plan.model.value(),
                   ",\"buffer_slot\":", plan.buffer_slot.value(),
                   ",\"buffer_generation\":", plan.buffer_generation.value(),
                   ",\"num_sequences\":", plan.resources.num_sequences.value(),
                   ",\"num_tokens\":", plan.resources.num_tokens.value(), ",\"sequences\":[");
  for (size_t index = 0; index < plan.sequences.size(); ++index) {
    if (index != 0) line.push_back(',');
    const scheduler::ScheduledSequence& item = plan.sequences[index];
    absl::StrAppend(&line, "{\"request_id\":", item.request.value(),
                    ",\"sequence_id\":", item.sequence.value(), ",\"epoch\":", item.epoch.value(),
                    ",\"kind\":\"", ToString(item.kind),
                    "\",\"input_begin\":", item.input_tokens.begin.value(),
                    ",\"input_end\":", item.input_tokens.end.value(),
                    ",\"reservation_id\":", item.kv_write.reservation.value(),
                    ",\"kv_read_begin\":", item.kv_read.logical_tokens.begin.value(),
                    ",\"kv_read_end\":", item.kv_read.logical_tokens.end.value(),
                    ",\"kv_write_begin\":", item.kv_write.logical_tokens.begin.value(),
                    ",\"kv_write_end\":", item.kv_write.logical_tokens.end.value(), "}");
  }
  line.append("]}");
  return Write(line);
}

absl::Status ReplayWriter::WriteCompletion(const ExecutionCompletion& completion,
                                           MonotonicTime now) {
  std::string token =
      completion.output_token.has_value() ? absl::StrCat(completion.output_token->value()) : "null";
  return Write(absl::StrCat(
      Prefix("execution_completion"), ",\"at_ns\":", now.time_since_epoch().count(),
      ",\"ticket_id\":", completion.ticket.value(), ",\"step_id\":", completion.step.value(),
      ",\"request_id\":", completion.request.value(), ",\"sequence_id\":",
      completion.sequence.value(), ",\"epoch\":", completion.epoch.value(), ",\"kind\":\"",
      ToString(completion.kind), "\",\"token_begin\":", completion.scheduled_tokens.begin.value(),
      ",\"token_end\":", completion.scheduled_tokens.end.value(),
      ",\"status_code\":", static_cast<int>(completion.status.code()), ",\"error_reason\":\"",
      ErrorReasonToName(completion.error_reason), "\",\"output_token\":", token, "}"));
}

absl::Status ReplayWriter::WriteResponse(const ResponseEvent& response, MonotonicTime now) {
  std::string line =
      absl::StrCat(Prefix("response_event"), ",\"at_ns\":", now.time_since_epoch().count());
  if (const auto* delta = std::get_if<TokenDelta>(&response)) {
    absl::StrAppend(&line, ",\"request_id\":", delta->request.value(),
                    ",\"event_kind\":\"token_delta\",\"output_position\":",
                    delta->output_position.value(), ",\"token_id\":", delta->token.value(),
                    ",\"finish_reason\":null,\"status_code\":0,"
                    "\"error_reason\":\"none\",\"prompt_tokens\":null,"
                    "\"output_tokens\":null}");
  } else {
    const TerminalResponse& terminal = std::get<TerminalResponse>(response);
    absl::StrAppend(&line, ",\"request_id\":", terminal.request.value(),
                    ",\"event_kind\":\"terminal\",\"output_position\":null,"
                    "\"token_id\":null,\"finish_reason\":\"",
                    ToString(terminal.reason),
                    "\",\"status_code\":", static_cast<int>(terminal.status),
                    ",\"error_reason\":\"",
                    ErrorReasonToName(terminal.error_reason.value_or(ErrorReason::kNone)),
                    "\",\"prompt_tokens\":", terminal.prompt_tokens.value(),
                    ",\"output_tokens\":", terminal.output_tokens.value(), "}");
  }
  return Write(line);
}

absl::Status ReplayWriter::WriteIdle(IdleReason reason, MonotonicTime now,
                                     std::optional<MonotonicTime> next) {
  const std::string next_value =
      next.has_value() ? absl::StrCat(next->time_since_epoch().count()) : "null";
  return Write(absl::StrCat(Prefix("idle"), ",\"at_ns\":", now.time_since_epoch().count(),
                            ",\"idle_reason\":\"", ToString(reason),
                            "\",\"next_event_ns\":", next_value, "}"));
}

absl::Status ReplayWriter::WriteFooter(const ReplayFooter& footer) {
  std::string line = absl::StrCat(
      Prefix("footer"), ",\"final_status_code\":", static_cast<int>(footer.final_status),
      ",\"error_reason\":\"", ErrorReasonToName(footer.error_reason),
      "\",\"events\":", footer.events, ",\"steps\":", footer.steps,
      ",\"requests\":", footer.requests, ",\"finished\":", footer.finished,
      ",\"cancelled\":", footer.cancelled, ",\"failed\":", footer.failed,
      ",\"used_sequences\":", footer.used_sequences, ",\"used_kv_tokens\":", footer.used_kv_tokens,
      ",\"live_tickets\":", footer.live_tickets,
      ",\"leased_plan_slots\":", footer.leased_plan_slots,
      ",\"invariants_ok\":", footer.invariants_ok ? "true" : "false", "}");
  absl::Status status = Write(line);
  if (status.ok()) {
    footer_written_ = true;
    status = sink_.Close();
  }
  return status;
}

absl::Status CheckReplayJsonLines(absl::string_view text) {
  if (text.size() > config::kMaxReplayInputBytes) {
    return ReplayError(absl::StatusCode::kResourceExhausted, "input exceeds replay limit");
  }
  size_t begin = 0;
  uint64_t ordinal = 0;
  bool saw_header = false;
  bool saw_footer = false;
  while (begin < text.size()) {
    const size_t end = text.find('\n', begin);
    const size_t length = end == absl::string_view::npos ? text.size() - begin : end - begin;
    if (length == 0) {
      if (end == absl::string_view::npos && begin == text.size()) break;
      return ReplayError(absl::StatusCode::kInvalidArgument, "blank record");
    }
    if (length > config::kMaxJsonlRecordBytes) {
      return ReplayError(absl::StatusCode::kResourceExhausted, "record exceeds line limit");
    }
    std::string type;
    absl::Status valid = ValidateLine(text.substr(begin, length), ordinal, &type);
    if (!valid.ok()) return valid;
    if (ordinal == 0 && type != "header") {
      return ReplayError(absl::StatusCode::kInvalidArgument, "first record is not header");
    }
    if (type == "header") {
      if (saw_header) return ReplayError(absl::StatusCode::kInvalidArgument, "duplicate header");
      saw_header = true;
    }
    if (type == "footer") {
      if (saw_footer) return ReplayError(absl::StatusCode::kInvalidArgument, "duplicate footer");
      saw_footer = true;
      if (end != absl::string_view::npos && end + 1 < text.size()) {
        return ReplayError(absl::StatusCode::kInvalidArgument, "record after footer");
      }
    }
    ++ordinal;
    if (end == absl::string_view::npos) break;
    begin = end + 1;
    if (begin == text.size()) break;
  }
  if (!saw_header || !saw_footer) {
    return ReplayError(absl::StatusCode::kInvalidArgument, "trace is truncated");
  }
  return absl::OkStatus();
}

absl::StatusOr<ReplayInputs> ReadReplayInputs(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return ReplayError(absl::StatusCode::kNotFound, "cannot open input", ErrorReason::kIoFailure);
  }
  const std::streampos size = file.tellg();
  if (size < 0 || static_cast<uint64_t>(size) > config::kMaxReplayInputBytes) {
    return ReplayError(absl::StatusCode::kResourceExhausted, "input exceeds replay limit");
  }
  file.seekg(0, std::ios::beg);
  std::string text(static_cast<size_t>(size), '\0');
  if (size > 0) file.read(text.data(), size);
  absl::Status checked = CheckReplayJsonLines(text);
  if (!checked.ok()) return checked;

  const size_t config_begin_marker = text.find("\"config\":");
  const size_t config_end_marker = text.find(",\"fake_model\":", config_begin_marker);
  if (config_begin_marker == std::string::npos || config_end_marker == std::string::npos) {
    return ReplayError(absl::StatusCode::kInvalidArgument, "header config is missing");
  }
  const size_t config_begin = config_begin_marker + std::string("\"config\":").size();
  const std::string config_json = text.substr(config_begin, config_end_marker - config_begin);
  absl::StatusOr<config::FieldValues> config_values = config::ParseConfigJson(config_json);
  if (!config_values.ok()) return config_values.status();

  std::string workload_text;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find('\n', begin);
    const size_t length = end == std::string::npos ? text.size() - begin : end - begin;
    const absl::string_view line(text.data() + begin, length);
    if (line.find("\"record_type\":\"workload_event\"") != absl::string_view::npos) {
      const std::string workload_line = WorkloadLineFromReplay(line);
      if (workload_line.empty()) {
        return ReplayError(absl::StatusCode::kInvalidArgument,
                           "cannot reconstruct workload record");
      }
      workload_text.append(workload_line);
      workload_text.push_back('\n');
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  absl::StatusOr<std::vector<WorkloadEvent>> workload = ParseWorkloadJsonLines(workload_text, true);
  if (!workload.ok()) return workload.status();
  return ReplayInputs{std::move(*config_values), std::move(*workload), std::move(text)};
}

absl::Status CompareReplayBytes(absl::string_view expected, absl::string_view actual) {
  if (expected == actual) return absl::OkStatus();
  size_t offset = 0;
  const size_t shared = std::min(expected.size(), actual.size());
  while (offset < shared && expected[offset] == actual[offset]) ++offset;
  return ReplayError(absl::StatusCode::kDataLoss,
                     absl::StrCat("byte mismatch at offset ", offset, " (expected bytes ",
                                  expected.size(), ", actual bytes ", actual.size(), ")"));
}

}  // namespace inferx::simulator
