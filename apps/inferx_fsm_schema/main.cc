#include <cstdio>

#include "inferx/engine/request_event.h"
#include "inferx/engine/request_state_machine.h"
#include "inferx/lifecycle/request_state.h"

int main() {
  const inferx::TransitionRuleSpan rules = inferx::TransitionRules();
  std::printf("{\"schema_version\":1,\"transitions\":[");
  for (size_t index = 0; index < rules.size; ++index) {
    const inferx::TransitionRule& rule = rules.data[index];
    if (index != 0) std::putchar(',');
    std::printf("{\"from\":\"%.*s\",\"event\":\"%.*s\",\"to\":",
                static_cast<int>(inferx::ToString(rule.current).size()),
                inferx::ToString(rule.current).data(),
                static_cast<int>(inferx::ToString(rule.event).size()),
                inferx::ToString(rule.event).data());
    if (rule.resolve_next == nullptr) {
      std::printf("\"%.*s\"", static_cast<int>(inferx::ToString(rule.fixed_next).size()),
                  inferx::ToString(rule.fixed_next).data());
    } else {
      std::printf("null");
    }
    std::printf(",\"conditional\":%s,\"effects\":%u,\"terminal\":%s}",
                rule.resolve_next == nullptr ? "false" : "true",
                static_cast<unsigned>(rule.effects), rule.terminal ? "true" : "false");
  }
  std::printf("]}\n");
  return 0;
}
