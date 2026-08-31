// Request registry (m1.md section 10.2): owns contexts; hash map for lookup
// plus a btree map owning deterministic live order. Insert is transactional;
// erase is terminal-only.

#ifndef INFERX_ENGINE_REQUEST_REGISTRY_H_
#define INFERX_ENGINE_REQUEST_REGISTRY_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"
#include "inferx/base/id.h"
#include "inferx/engine/request_context.h"

namespace inferx {

// Lookup hash: std::unordered_map keeps sanitizer lanes clean (Abseil's
// header-instantiated raw_hash_map trips UBSan's null-load check; ADR 0009
// keeps deterministic order in the btree map below instead).
struct RequestIdHash {
  size_t operator()(RequestId id) const { return std::hash<uint64_t>{}(id.value()); }
};

class RequestRegistry {
 public:
  // Assigns sequence/arrival only after duplicate and capacity checks;
  // allocation failure rolls back the first insertion (transactional).
  absl::Status Insert(std::unique_ptr<RequestContext> context);

  [[nodiscard]] RequestContext* Find(RequestId id);
  [[nodiscard]] const RequestContext* Find(RequestId id) const;
  [[nodiscard]] bool Contains(RequestId id) const;
  [[nodiscard]] size_t size() const { return by_id_.size(); }

  // Legal only after terminal response emission, no in-flight ticket, and
  // reservation release (caller-checked; debug asserts here).
  absl::Status EraseTerminal(RequestId id);

  // Deterministic arrival-order view of live requests (scheduling consumes
  // this order; hash iteration is never observable).
  void AppendArrivalOrder(std::vector<const RequestContext*>& output) const;

 private:
  std::unordered_map<RequestId, std::unique_ptr<RequestContext>, RequestIdHash> by_id_;
  absl::btree_map<ArrivalKey, RequestId> arrival_order_;
  uint64_t next_arrival_ordinal_ = 0;  // simulator generators start at 1
};

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_REGISTRY_H_
