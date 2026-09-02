#pragma once

#include <string_view>

#include "inferx/tensor/tensor_view.h"

namespace inferx::model {

// Optional diagnostic sink for the stable, schema-versioned intermediate
// names (m5.md section 8.4). Production execution passes nullptr; the null
// path performs no allocation or copy.
class IntermediateTraceSink {
 public:
  virtual ~IntermediateTraceSink() = default;

  // Called once per named intermediate in forward order. The view is valid
  // for the duration of the call only.
  virtual void OnIntermediate(std::string_view name, const TensorView& value) = 0;
};

}  // namespace inferx::model
