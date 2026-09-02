// Adapted from divedb/tokenizer (MIT) for InferX local-only use (ADR 0024):
// consumes LocalArtifacts bytes; no filesystem access.

#include "tokenizer/internal/chat/template_source.h"

#include <utility>

namespace tokenizer {
namespace {

constexpr std::string_view kDefault = "default";

/// The deprecated multi-template form: a list of {name, template} objects,
/// used by checkpoints that shipped a tool-calling variant alongside the
/// ordinary one.
bool CollectNamedList(const Json& list, ChatTemplateSources* out) {
  bool found = false;
  for (const Json& entry : list) {
    if (!entry.is_object()) continue;
    auto name = GetString(entry, "name");
    auto source = GetString(entry, "template");
    if (!source.has_value()) continue;
    const std::string key = name.value_or(std::string(kDefault));
    out->by_name[key] = *source;
    found = true;
  }
  if (found && !out->by_name.contains(std::string(kDefault)) && out->by_name.size() == 1) {
    // A single unnamed variant is the default, whatever it calls itself.
    out->default_name = out->by_name.begin()->first;
  }
  return found;
}

}  // namespace

const std::string* ChatTemplateSources::Find(std::string_view name) const {
  auto it = by_name.find(name);
  return it == by_name.end() ? nullptr : &it->second;
}

std::vector<std::string> ChatTemplateSources::Names() const {
  std::vector<std::string> names;
  names.reserve(by_name.size());
  for (const auto& [name, unused] : by_name) names.push_back(name);
  return names;
}

ChatTemplateSources DiscoverChatTemplates(const LocalArtifacts& artifacts,
                                          const Json& tokenizer_config) {
  ChatTemplateSources out;
  out.default_name = std::string(kDefault);

  // 1. The canonical location since Transformers 4.51. gpt-oss-20b ships its
  //    16 KB template here and has no chat_template key in its config at all.
  if (artifacts.chat_template_jinja.has_value() && !artifacts.chat_template_jinja->empty()) {
    out.by_name[std::string(kDefault)] = *artifacts.chat_template_jinja;
  }

  // 2 and 3. The config key, as either a string or the deprecated list.
  if (out.empty() && tokenizer_config.is_object()) {
    auto it = tokenizer_config.find("chat_template");
    if (it != tokenizer_config.end()) {
      if (it->is_string()) {
        out.by_name[std::string(kDefault)] = it->get<std::string>();
      } else if (it->is_array()) {
        CollectNamedList(*it, &out);
      }
    }
  }

  // 4. Processor-level, used by some multimodal repos.
  if (out.empty() && artifacts.chat_template_json.has_value()) {
    auto parsed = ParseJson(*artifacts.chat_template_json, "chat_template.json");
    if (parsed.ok()) {
      if (parsed->is_string()) {
        out.by_name[std::string(kDefault)] = parsed->get<std::string>();
      } else if (parsed->is_object()) {
        if (auto source = GetString(*parsed, "chat_template")) {
          out.by_name[std::string(kDefault)] = *source;
        }
      } else if (parsed->is_array()) {
        CollectNamedList(*parsed, &out);
      }
    }
  }

  return out;
}

}  // namespace tokenizer
