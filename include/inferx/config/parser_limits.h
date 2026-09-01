// Schema-v1 safety constants (m1.md section 8.4). Shared by file readers and
// tests; defensive limits, not hidden CLI overrides.

#ifndef INFERX_CONFIG_PARSER_LIMITS_H_
#define INFERX_CONFIG_PARSER_LIMITS_H_

#include <cstdint>

namespace inferx::config {

inline constexpr uint64_t kMaxConfigFileBytes = 1u << 20;      // 1 MiB
inline constexpr uint64_t kMaxWorkloadFileBytes = 1u << 30;    // 1 GiB
inline constexpr uint64_t kMaxReplayInputBytes = 16ull << 30;  // 16 GiB
inline constexpr uint64_t kMaxJsonlRecordBytes = 16u << 20;    // 16 MiB
inline constexpr int kMaxJsonNestingDepth = 32;
inline constexpr int kMaxObjectMembers = 128;
inline constexpr uint64_t kMaxStringValueBytes = 64u << 10;  // 64 KiB
inline constexpr uint64_t kMaxDiagnosticBytes = 4u << 10;    // 4 KiB
// Cross-field cap on max_sequences_per_step beyond the per-field range.
inline constexpr uint64_t kMaxSequencesPerStepHardCap = 16384;

}  // namespace inferx::config

#endif  // INFERX_CONFIG_PARSER_LIMITS_H_
