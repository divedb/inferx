// Consumes the installed pinned Abseil package (no source checkout on any
// include path) using the Status/StatusOr facilities ADR 0002 selects.
#include <iostream>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

int main() {
  const absl::StatusOr<int> value = []() -> absl::StatusOr<int> {
    return absl::StatusOr<int>(7);
  }();
  if (!value.ok() || *value != 7) {
    std::cerr << "installed absl StatusOr did not behave\n";
    return 1;
  }
  const absl::Status status = absl::CancelledError();
  if (status.code() != absl::StatusCode::kCancelled) {
    std::cerr << "installed absl Status codes did not behave\n";
    return 1;
  }
  if (absl::StrCat("consumer", "-", "ok") != "consumer-ok") {
    std::cerr << "installed absl strings did not behave\n";
    return 1;
  }
  std::cout << "consumer ok\n";
  return 0;
}
