#include "tokenizer/core/json_value.h"

#include <utility>

#include "tokenizer/internal/detail/json_bridge.h"

namespace tokenizer {

struct JsonValue::Impl {
  Json json;

  // Children handed out by Find()/At() must outlive the call, and callers
  // legitimately hold the pointer while walking a structure. Wrapping each
  // child on demand and caching it here keeps those pointers stable for the
  // lifetime of the parent.
  mutable std::vector<std::unique_ptr<JsonValue>> children;
};

JsonValue::JsonValue() : impl_(std::make_unique<Impl>()) {}

JsonValue::JsonValue(const JsonValue& other) : impl_(std::make_unique<Impl>()) {
  impl_->json = other.impl_->json;
}

JsonValue::JsonValue(JsonValue&& other) noexcept = default;

JsonValue& JsonValue::operator=(const JsonValue& other) {
  if (this != &other) {
    impl_ = std::make_unique<Impl>();
    impl_->json = other.impl_->json;
  }
  return *this;
}

JsonValue& JsonValue::operator=(JsonValue&& other) noexcept = default;

JsonValue::~JsonValue() = default;

StatusOr<JsonValue> JsonValue::Parse(std::string_view text) {
  ABSL_ASSIGN_OR_RETURN(Json parsed, ParseJson(text, "JSON value"));
  return JsonBridge::ToPublic(parsed);
}

JsonValue JsonValue::Object() {
  JsonValue value;
  value.impl_->json = Json::object();
  return value;
}

JsonValue JsonValue::Array() {
  JsonValue value;
  value.impl_->json = Json::array();
  return value;
}

JsonValue JsonValue::String(std::string_view value) {
  JsonValue out;
  out.impl_->json = std::string(value);
  return out;
}

JsonValue JsonValue::Int(int64_t value) {
  JsonValue out;
  out.impl_->json = value;
  return out;
}

JsonValue JsonValue::Bool(bool value) {
  JsonValue out;
  out.impl_->json = value;
  return out;
}

bool JsonValue::IsNull() const { return impl_->json.is_null(); }
bool JsonValue::IsObject() const { return impl_->json.is_object(); }
bool JsonValue::IsArray() const { return impl_->json.is_array(); }
bool JsonValue::IsString() const { return impl_->json.is_string(); }
bool JsonValue::IsNumber() const { return impl_->json.is_number(); }
bool JsonValue::IsBool() const { return impl_->json.is_boolean(); }

StatusOr<std::string_view> JsonValue::AsString() const {
  if (!impl_->json.is_string()) {
    return InvalidArgumentError("JSON value is not a string");
  }
  return std::string_view(impl_->json.get_ref<const std::string&>());
}

StatusOr<int64_t> JsonValue::AsInt() const {
  if (!impl_->json.is_number_integer()) {
    return InvalidArgumentError("JSON value is not an integer");
  }
  return impl_->json.get<int64_t>();
}

StatusOr<double> JsonValue::AsDouble() const {
  if (!impl_->json.is_number()) {
    return InvalidArgumentError("JSON value is not a number");
  }
  return impl_->json.get<double>();
}

StatusOr<bool> JsonValue::AsBool() const {
  if (!impl_->json.is_boolean()) {
    return InvalidArgumentError("JSON value is not a boolean");
  }
  return impl_->json.get<bool>();
}

const JsonValue* JsonValue::Find(std::string_view key) const {
  if (!impl_->json.is_object()) return nullptr;
  auto it = impl_->json.find(std::string(key));
  if (it == impl_->json.end()) return nullptr;
  impl_->children.push_back(std::make_unique<JsonValue>(JsonBridge::ToPublic(*it)));
  return impl_->children.back().get();
}

size_t JsonValue::Size() const {
  if (impl_->json.is_array() || impl_->json.is_object()) {
    return impl_->json.size();
  }
  return 0;
}

const JsonValue* JsonValue::At(size_t index) const {
  if (!impl_->json.is_array() || index >= impl_->json.size()) return nullptr;
  impl_->children.push_back(std::make_unique<JsonValue>(JsonBridge::ToPublic(impl_->json[index])));
  return impl_->children.back().get();
}

void JsonValue::Set(std::string_view key, JsonValue value) {
  if (!impl_->json.is_object()) impl_->json = Json::object();
  impl_->json[std::string(key)] = value.impl_->json;
  impl_->children.clear();
}

void JsonValue::Push(JsonValue value) {
  if (!impl_->json.is_array()) impl_->json = Json::array();
  impl_->json.push_back(value.impl_->json);
  impl_->children.clear();
}

std::string JsonValue::Dump() const { return impl_->json.dump(); }

// --- JsonBridge -----------------------------------------------------------

JsonValue JsonBridge::ToPublic(const Json& value) {
  JsonValue out;
  out.impl_->json = value;
  return out;
}

const Json& JsonBridge::ToInternal(const JsonValue& value) { return value.impl_->json; }

Json& JsonBridge::MutableInternal(JsonValue& value) { return value.impl_->json; }

OrderedJson JsonBridge::ToOrdered(const JsonValue& value) {
  // Round-trip through text: nlohmann's ordered and unordered variants are
  // distinct types with no direct conversion, and these values are small.
  return OrderedJson::parse(value.impl_->json.dump());
}

}  // namespace tokenizer
