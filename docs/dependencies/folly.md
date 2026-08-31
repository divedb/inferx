# Qualification report: folly (rejected/removed)

- Manifest entry: `folly` — rejected, `removed: true`, owner unowned
- Former pin: `39d23d493c5947f043e78932d9c5822963449b33` (gitlink removed in M0)
- License: Apache-2.0 (historical note; nothing is consumed)

## 1. Which InferX contract would use it?

Hypothetical bounded queues/futures/executors (plan section 17.2: "deferred; prefer
C++23 + Asio for the initial runtime; remove if no measured need").

## 2. Required now, deferred, experimental, or rejected?

**Rejected.** No milestone M0–M18 requires Folly: the engine's coordination model is a
single-writer event loop with bounded channels the project designs itself; HTTP I/O is
Asio/Beast; JSON/status/strings are Abseil or the M1 choice; no accepted ADR identifies
a C++23/Asio gap that Folly closes.

## 3. Source and transitive dependencies (why it lost)

Folly has one of the heaviest dependency/build footprints available: it requires pinned
Boost, double-conversion, gflags, glog, libevent, fmt, and often more, plus long
compiles and careful toolchain matching. Against "C++23 standard library + Asio + Abseil
utilities", it adds an entire runtime stack to save writing a bounded MPSC queue.

## 4. Toolchain/C++23 compatibility

Not qualified — no build was attempted; irrelevant once rejected.

## 5. Runtime behavior caveats

Throws, uses threads/executors, global state, and background machinery broadly;
contradicts the plan's ownership model (§6) if used for core paths.

## 6. API stability and namespaces

`folly::`; large surface, high churn between releases.

## 7. License/notice obligations and security

Apache-2.0. Removal means no obligations attach to InferX.

## 8. Binary/build/startup cost

Very high (build time and binary size) relative to any identified need — a primary
rejection reason alongside §3.

## 9. Upgrade/rollback procedure

If a future, concrete, accepted use case emerges (ADR required), re-add via the standard
qualification workflow: new gitlink + manifest entry + report + build evidence.

## 10. Disposition and approvals

**Rejected and removed** in M0 (gitlink, `.gitmodules`, manifest updated together; this
report and the manifest entry with `"removed": true` retain the audit trail). Reversal
requires an accepted ADR identifying the gap and a cost comparison.
