// Status propagation macros.
//
// Exactly two macros: INFERX_RETURN_IF_ERROR and INFERX_ASSIGN_OR_RETURN.
// Both evaluate their expression exactly once, preserve move-only values and
// status payloads, work in normal statement scopes (including nested), and
// use collision-resistant internal names. INFERX_ASSIGN_OR_RETURN assigns to
// an already-declared lvalue; declarations and structured bindings are
// deliberately unsupported.

#ifndef INFERX_BASE_STATUS_MACROS_H_
#define INFERX_BASE_STATUS_MACROS_H_

#include <utility>

#include "absl/base/macros.h"
#include "absl/status/status.h"

// Indirect concatenation so __LINE__ expands before pasting.
#define INFERX_STATUS_CONCAT_INDIRECT(a, b) a##b
#define INFERX_STATUS_CONCAT(a, b) INFERX_STATUS_CONCAT_INDIRECT(a, b)

#define INFERX_RETURN_IF_ERROR(expr)                                                           \
  do {                                                                                         \
    auto INFERX_STATUS_CONCAT(__INFERX_RETURN_IF_ERROR_, __LINE__) = (expr);                   \
    if (ABSL_PREDICT_FALSE(!INFERX_STATUS_CONCAT(__INFERX_RETURN_IF_ERROR_, __LINE__).ok())) { \
      return INFERX_STATUS_CONCAT(__INFERX_RETURN_IF_ERROR_, __LINE__);                        \
    }                                                                                          \
  } while (false)

#define INFERX_ASSIGN_OR_RETURN(lhs, expr)                                                      \
  do {                                                                                          \
    auto INFERX_STATUS_CONCAT(__INFERX_ASSIGN_OR_RETURN_, __LINE__) = (expr);                   \
    if (ABSL_PREDICT_FALSE(!INFERX_STATUS_CONCAT(__INFERX_ASSIGN_OR_RETURN_, __LINE__).ok())) { \
      return std::move(INFERX_STATUS_CONCAT(__INFERX_ASSIGN_OR_RETURN_, __LINE__)).status();    \
    }                                                                                           \
    (lhs) = std::move(INFERX_STATUS_CONCAT(__INFERX_ASSIGN_OR_RETURN_, __LINE__)).value();      \
  } while (false)

#endif  // INFERX_BASE_STATUS_MACROS_H_
