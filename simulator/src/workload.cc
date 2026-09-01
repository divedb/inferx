#include "inferx/simulator/workload.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "inferx/base/status.h"
#include "inferx/config/parser_limits.h"
#include "simdjson.h"

namespace inferx::simulator {
namespace {

absl::Status Invalid(uint64_t line, absl::string_view field, absl::string_view detail) {
  return WithErrorReason(
      absl::InvalidArgumentError(absl::StrCat("workload.line.", line, ".", field, ": ", detail)),
      ErrorReason::kInvalidWorkload);
}

absl::StatusOr<simdjson::dom::element> Field(simdjson::dom::object object, absl::string_view name,
                                             uint64_t line) {
  simdjson::dom::element value;
  const simdjson::error_code error = object.at_key(name).get(value);
  if (error != simdjson::SUCCESS) {
    return Invalid(line, name, "missing required field");
  }
  return value;
}

absl::StatusOr<uint64_t> Unsigned(simdjson::dom::object object, absl::string_view name,
                                  uint64_t line) {
  absl::StatusOr<simdjson::dom::element> field = Field(object, name, line);
  if (!field.ok()) {
    return field.status();
  }
  uint64_t value = 0;
  if (field->get(value) != simdjson::SUCCESS) {
    return Invalid(line, name, "must be an unsigned integer");
  }
  return value;
}

absl::StatusOr<std::string> String(simdjson::dom::object object, absl::string_view name,
                                   uint64_t line) {
  absl::StatusOr<simdjson::dom::element> field = Field(object, name, line);
  if (!field.ok()) {
    return field.status();
  }
  std::string_view value;
  if (field->get(value) != simdjson::SUCCESS) {
    return Invalid(line, name, "must be a string");
  }
  if (value.size() > config::kMaxStringValueBytes) {
    return Invalid(line, name, "string exceeds schema limit");
  }
  return std::string(value);
}

absl::Status CheckFields(simdjson::dom::object object, const std::set<std::string>& allowed,
                         uint64_t line) {
  std::set<std::string> seen;
  for (auto [key, value] : object) {
    static_cast<void>(value);
    const std::string name(key);
    if (!seen.insert(name).second) {
      return Invalid(line, name, "duplicate field");
    }
    if (!allowed.contains(name)) {
      return Invalid(line, name, "unknown field");
    }
    if (seen.size() > static_cast<size_t>(config::kMaxObjectMembers)) {
      return Invalid(line, name, "too many object members");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<uint64_t>> OptionalUnsigned(simdjson::dom::object object,
                                                         absl::string_view name, uint64_t line) {
  simdjson::dom::element value;
  const simdjson::error_code error = object.at_key(name).get(value);
  if (error == simdjson::NO_SUCH_FIELD) {
    return std::optional<uint64_t>();
  }
  if (error != simdjson::SUCCESS) {
    return Invalid(line, name, "cannot read field");
  }
  if (value.is_null()) {
    return std::optional<uint64_t>();
  }
  uint64_t parsed = 0;
  if (value.get(parsed) != simdjson::SUCCESS) {
    return Invalid(line, name, "must be null or an unsigned integer");
  }
  return std::optional<uint64_t>(parsed);
}

absl::StatusOr<bool> OptionalBool(simdjson::dom::object object, absl::string_view name,
                                  uint64_t line, bool default_value) {
  simdjson::dom::element value;
  const simdjson::error_code error = object.at_key(name).get(value);
  if (error == simdjson::NO_SUCH_FIELD) {
    return default_value;
  }
  bool parsed = false;
  if (error != simdjson::SUCCESS || value.get(parsed) != simdjson::SUCCESS) {
    return Invalid(line, name, "must be a boolean");
  }
  return parsed;
}

struct LineIdentity {
  uint64_t number = 0;
  uint64_t ordinal = 0;
};

absl::StatusOr<WorkloadEvent> ParseLine(absl::string_view line_text, LineIdentity identity) {
  const uint64_t line_number = identity.number;
  const uint64_t ordinal = identity.ordinal;
  simdjson::dom::parser parser;
  simdjson::dom::element root;
  try {
    const simdjson::error_code error = parser.parse(line_text.data(), line_text.size()).get(root);
    if (error != simdjson::SUCCESS) {
      return Invalid(line_number, "__json__", simdjson::error_message(error));
    }
  } catch (const std::exception& exception) {
    return Invalid(line_number, "__json__", absl::StrCat("parser threw: ", exception.what()));
  }
  simdjson::dom::object object;
  if (root.get(object) != simdjson::SUCCESS) {
    return Invalid(line_number, "__json__", "record must be an object");
  }
  absl::StatusOr<uint64_t> version = Unsigned(object, "schema_version", line_number);
  absl::StatusOr<std::string> event_name = String(object, "event_type", line_number);
  absl::StatusOr<uint64_t> at_ns = Unsigned(object, "at_ns", line_number);
  if (!version.ok()) return version.status();
  if (!event_name.ok()) return event_name.status();
  if (!at_ns.ok()) return at_ns.status();
  if (*version != 1) return Invalid(line_number, "schema_version", "unsupported version");
  if (*at_ns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return Invalid(line_number, "at_ns", "exceeds monotonic time range");
  }
  const MonotonicTime at(Nanoseconds(static_cast<int64_t>(*at_ns)));

  if (*event_name == "submit") {
    const std::set<std::string> allowed{"schema_version",    "event_type",  "at_ns",
                                        "request_id",        "model_id",    "token_ids",
                                        "max_output_tokens", "deadline_ns", "finish_after_tokens",
                                        "priority",          "tenant_scope"};
    absl::Status fields = CheckFields(object, allowed, line_number);
    if (!fields.ok()) return fields;
    absl::StatusOr<uint64_t> request_id = Unsigned(object, "request_id", line_number);
    absl::StatusOr<uint64_t> model_id = Unsigned(object, "model_id", line_number);
    absl::StatusOr<uint64_t> max_output = Unsigned(object, "max_output_tokens", line_number);
    if (!request_id.ok()) return request_id.status();
    if (!model_id.ok()) return model_id.status();
    if (!max_output.ok()) return max_output.status();
    if (*model_id > std::numeric_limits<uint32_t>::max() || *max_output == 0 ||
        *max_output > std::numeric_limits<uint32_t>::max()) {
      return Invalid(line_number, "submit", "model or output count is out of range");
    }
    absl::StatusOr<simdjson::dom::element> token_element = Field(object, "token_ids", line_number);
    if (!token_element.ok()) return token_element.status();
    simdjson::dom::array token_array;
    if (token_element->get(token_array) != simdjson::SUCCESS) {
      return Invalid(line_number, "token_ids", "must be an array");
    }
    std::vector<TokenId> tokens;
    for (simdjson::dom::element token : token_array) {
      int64_t value = 0;
      if (token.get(value) != simdjson::SUCCESS || value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max()) {
        return Invalid(line_number, "token_ids", "contains an out-of-range integer");
      }
      tokens.emplace_back(static_cast<int32_t>(value));
    }
    if (tokens.empty()) return Invalid(line_number, "token_ids", "must not be empty");
    absl::StatusOr<std::optional<uint64_t>> deadline =
        OptionalUnsigned(object, "deadline_ns", line_number);
    absl::StatusOr<std::optional<uint64_t>> finish_after =
        OptionalUnsigned(object, "finish_after_tokens", line_number);
    if (!deadline.ok()) return deadline.status();
    if (!finish_after.ok()) return finish_after.status();
    const std::optional<uint64_t> deadline_value = *deadline;
    const std::optional<uint64_t> finish_after_value = *finish_after;
    int64_t priority = 0;
    simdjson::dom::element priority_element;
    if (object.at_key("priority").get(priority_element) == simdjson::SUCCESS &&
        priority_element.get(priority) != simdjson::SUCCESS) {
      return Invalid(line_number, "priority", "must be an integer");
    }
    uint64_t tenant = 0;
    simdjson::dom::element tenant_element;
    if (object.at_key("tenant_scope").get(tenant_element) == simdjson::SUCCESS &&
        tenant_element.get(tenant) != simdjson::SUCCESS) {
      return Invalid(line_number, "tenant_scope", "must be an unsigned integer");
    }
    if (priority < std::numeric_limits<int32_t>::min() ||
        priority > std::numeric_limits<int32_t>::max() ||
        tenant > std::numeric_limits<uint32_t>::max()) {
      return Invalid(line_number, "submit", "priority or tenant is out of range");
    }
    std::optional<MonotonicTime> parsed_deadline;
    if (deadline_value.has_value()) {
      const uint64_t value = deadline_value.value();
      if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return Invalid(line_number, "deadline_ns", "exceeds monotonic time range");
      }
      parsed_deadline = MonotonicTime(Nanoseconds(static_cast<int64_t>(value)));
    }
    std::optional<TokenCount> parsed_finish;
    if (finish_after_value.has_value()) {
      const uint64_t value = finish_after_value.value();
      if (value == 0 || value > *max_output || value > std::numeric_limits<uint32_t>::max()) {
        return Invalid(line_number, "finish_after_tokens", "must be in 1..max_output_tokens");
      }
      parsed_finish = TokenCount(static_cast<uint32_t>(value));
    }
    GenerateRequest request{
        .id = RequestId(*request_id),
        .model = ModelId(static_cast<uint32_t>(*model_id)),
        .input = std::move(tokens),
        .generation = GenerationLimits{TokenCount(static_cast<uint32_t>(*max_output))},
        .priority = Priority(static_cast<int32_t>(priority)),
        .deadline = parsed_deadline,
        .tenant = TenantScope(static_cast<uint32_t>(tenant)),
    };
    return WorkloadEvent{1, WorkloadEventType::kSubmit, at, ordinal,
                         SubmitWorkloadEvent{std::move(request), parsed_finish}};
  }

  if (*event_name == "cancel" || *event_name == "preempt" || *event_name == "requeue") {
    const std::set<std::string> allowed{"schema_version", "event_type", "at_ns", "request_id"};
    absl::Status fields = CheckFields(object, allowed, line_number);
    if (!fields.ok()) return fields;
    absl::StatusOr<uint64_t> request = Unsigned(object, "request_id", line_number);
    if (!request.ok()) return request.status();
    if (*event_name == "cancel") {
      return WorkloadEvent{1, WorkloadEventType::kCancel, at, ordinal,
                           CancelWorkloadEvent{RequestId(*request)}};
    }
    if (*event_name == "preempt") {
      return WorkloadEvent{1, WorkloadEventType::kPreempt, at, ordinal,
                           PreemptWorkloadEvent{RequestId(*request)}};
    }
    return WorkloadEvent{1, WorkloadEventType::kRequeue, at, ordinal,
                         RequeueWorkloadEvent{RequestId(*request)}};
  }

  if (*event_name == "inject_executor_failure") {
    const std::set<std::string> allowed{"schema_version", "event_type", "at_ns",
                                        "target_kind",    "target_id",  "status_code",
                                        "error_reason",   "persistent"};
    absl::Status fields = CheckFields(object, allowed, line_number);
    if (!fields.ok()) return fields;
    auto target_name = String(object, "target_kind", line_number);
    auto target_id = Unsigned(object, "target_id", line_number);
    auto status_code = Unsigned(object, "status_code", line_number);
    auto reason_name = String(object, "error_reason", line_number);
    auto persistent = OptionalBool(object, "persistent", line_number, false);
    if (!target_name.ok()) return target_name.status();
    if (!target_id.ok()) return target_id.status();
    if (!status_code.ok()) return status_code.status();
    if (!reason_name.ok()) return reason_name.status();
    if (!persistent.ok()) return persistent.status();
    const std::optional<FailureTargetKind> target = FailureTargetKindFromName(*target_name);
    const std::optional<ErrorReason> reason = ErrorReasonFromName(*reason_name);
    if (!target.has_value() || !reason.has_value() || *reason == ErrorReason::kNone ||
        *status_code == 0 ||
        *status_code > static_cast<uint64_t>(absl::StatusCode::kUnauthenticated)) {
      return Invalid(line_number, "inject_executor_failure", "invalid target/status/reason");
    }
    FailureRule rule{
        *target, *target_id,
        absl::Status(static_cast<absl::StatusCode>(*status_code), "injected executor failure"),
        *reason, *persistent};
    return WorkloadEvent{1, WorkloadEventType::kInjectExecutorFailure, at, ordinal,
                         InjectExecutorFailureEvent{std::move(rule)}};
  }

  if (*event_name == "shutdown") {
    const std::set<std::string> allowed{"schema_version", "event_type", "at_ns", "mode"};
    absl::Status fields = CheckFields(object, allowed, line_number);
    if (!fields.ok()) return fields;
    auto mode = String(object, "mode", line_number);
    if (!mode.ok()) return mode.status();
    if (*mode != "cancel" && *mode != "drain") {
      return Invalid(line_number, "mode", "must be cancel or drain");
    }
    return WorkloadEvent{
        1, WorkloadEventType::kShutdown, at, ordinal,
        ShutdownWorkloadEvent{*mode == "cancel" ? ShutdownMode::kCancel : ShutdownMode::kDrain}};
  }
  return Invalid(line_number, "event_type", "unknown event type");
}

int EventClass(WorkloadEventType type) {
  switch (type) {
    case WorkloadEventType::kShutdown:
    case WorkloadEventType::kCancel:
      return 1;
    case WorkloadEventType::kInjectExecutorFailure:
      return 2;
    case WorkloadEventType::kSubmit:
    case WorkloadEventType::kPreempt:
    case WorkloadEventType::kRequeue:
      return 3;
  }
  return 4;
}

}  // namespace

absl::string_view ToString(WorkloadEventType type) {
  switch (type) {
    case WorkloadEventType::kSubmit:
      return "submit";
    case WorkloadEventType::kCancel:
      return "cancel";
    case WorkloadEventType::kInjectExecutorFailure:
      return "inject_executor_failure";
    case WorkloadEventType::kPreempt:
      return "preempt";
    case WorkloadEventType::kRequeue:
      return "requeue";
    case WorkloadEventType::kShutdown:
      return "shutdown";
  }
  return "unknown";
}

absl::string_view ToString(ShutdownMode mode) {
  switch (mode) {
    case ShutdownMode::kCancel:
      return "cancel";
    case ShutdownMode::kDrain:
      return "drain";
  }
  return "unknown";
}

absl::StatusOr<std::vector<WorkloadEvent>> ParseWorkloadJsonLines(absl::string_view text,
                                                                  bool require_sorted) {
  if (text.size() > config::kMaxWorkloadFileBytes) {
    return Invalid(0, "__file__", "size exceeds workload limit");
  }
  std::vector<WorkloadEvent> events;
  std::set<RequestId> submitted;
  size_t begin = 0;
  uint64_t line_number = 0;
  MonotonicTime previous{};
  bool have_previous = false;
  bool saw_shutdown = false;
  while (begin < text.size()) {
    const size_t end = text.find('\n', begin);
    const size_t length = end == absl::string_view::npos ? text.size() - begin : end - begin;
    ++line_number;
    if (length == 0) {
      return Invalid(line_number, "__json__", "blank records are not allowed");
    }
    if (length > config::kMaxJsonlRecordBytes) {
      return Invalid(line_number, "__json__", "record exceeds line limit");
    }
    absl::StatusOr<WorkloadEvent> event =
        ParseLine(text.substr(begin, length),
                  LineIdentity{.number = line_number, .ordinal = line_number - 1});
    if (!event.ok()) return event.status();
    if (require_sorted && have_previous && event->at < previous) {
      return Invalid(line_number, "at_ns", "workload is not timestamp-sorted");
    }
    if (event->type == WorkloadEventType::kSubmit) {
      const RequestId id = std::get<SubmitWorkloadEvent>(event->payload).request.id;
      if (!submitted.insert(id).second) {
        return Invalid(line_number, "request_id", "duplicate submit id");
      }
    }
    if (event->type == WorkloadEventType::kShutdown) {
      if (saw_shutdown) return Invalid(line_number, "event_type", "second shutdown event");
      saw_shutdown = true;
    }
    previous = event->at;
    have_previous = true;
    events.push_back(std::move(*event));
    if (end == absl::string_view::npos) break;
    begin = end + 1;
  }
  std::stable_sort(events.begin(), events.end(),
                   [](const WorkloadEvent& left, const WorkloadEvent& right) {
                     if (left.at != right.at) return left.at < right.at;
                     const int left_class = EventClass(left.type);
                     const int right_class = EventClass(right.type);
                     if (left_class != right_class) return left_class < right_class;
                     return left.file_ordinal < right.file_ordinal;
                   });
  return events;
}

absl::StatusOr<std::vector<WorkloadEvent>> ReadWorkloadFile(const std::string& path,
                                                            bool require_sorted) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return WithErrorReason(
        absl::NotFoundError(absl::StrCat("workload.file: cannot open '", path, "'")),
        ErrorReason::kIoFailure);
  }
  const std::streampos size = file.tellg();
  if (size < 0 || static_cast<uint64_t>(size) > config::kMaxWorkloadFileBytes) {
    return Invalid(0, "__file__", "size exceeds workload limit");
  }
  file.seekg(0, std::ios::beg);
  std::string text(static_cast<size_t>(size), '\0');
  if (size > 0) file.read(text.data(), size);
  return ParseWorkloadJsonLines(text, require_sorted);
}

std::string CanonicalWorkloadEvent(const WorkloadEvent& event) {
  std::string output = absl::StrCat("{\"schema_version\":1,\"event_type\":\"", ToString(event.type),
                                    "\",\"at_ns\":", event.at.time_since_epoch().count());
  std::visit(
      [&output](const auto& payload) {
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<Payload, SubmitWorkloadEvent>) {
          const GenerateRequest& request = payload.request;
          absl::StrAppend(&output, ",\"request_id\":", request.id.value(),
                          ",\"model_id\":", request.model.value(), ",\"token_ids\":[");
          const auto& tokens = std::get<std::vector<TokenId>>(request.input);
          for (size_t index = 0; index < tokens.size(); ++index) {
            if (index != 0) output.push_back(',');
            absl::StrAppend(&output, tokens[index].value());
          }
          absl::StrAppend(&output,
                          "],\"max_output_tokens\":", request.generation.max_output_tokens.value(),
                          ",\"deadline_ns\":");
          if (request.deadline.has_value()) {
            absl::StrAppend(&output, request.deadline->time_since_epoch().count());
          } else {
            absl::StrAppend(&output, "null");
          }
          absl::StrAppend(&output, ",\"finish_after_tokens\":");
          if (payload.finish_after_tokens.has_value()) {
            absl::StrAppend(&output, payload.finish_after_tokens->value());
          } else {
            absl::StrAppend(&output, "null");
          }
          absl::StrAppend(&output, ",\"priority\":", request.priority.value(),
                          ",\"tenant_scope\":", request.tenant.value());
        } else if constexpr (std::is_same_v<Payload, CancelWorkloadEvent> ||
                             std::is_same_v<Payload, PreemptWorkloadEvent> ||
                             std::is_same_v<Payload, RequeueWorkloadEvent>) {
          absl::StrAppend(&output, ",\"request_id\":", payload.request.value());
        } else if constexpr (std::is_same_v<Payload, InjectExecutorFailureEvent>) {
          absl::StrAppend(&output, ",\"target_kind\":\"", ToString(payload.rule.target_kind),
                          "\",\"target_id\":", payload.rule.target_id,
                          ",\"status_code\":", static_cast<int>(payload.rule.status.code()),
                          ",\"error_reason\":\"", ErrorReasonToName(payload.rule.reason),
                          "\",\"persistent\":", payload.rule.persistent ? "true" : "false");
        } else if constexpr (std::is_same_v<Payload, ShutdownWorkloadEvent>) {
          absl::StrAppend(&output, ",\"mode\":\"", ToString(payload.mode), "\"");
        }
      },
      event.payload);
  output.push_back('}');
  return output;
}

}  // namespace inferx::simulator
