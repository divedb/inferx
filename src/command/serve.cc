#include "common.h"

namespace inferx::command {

void Serve(const ServeOptions&) {
  internal::Unavailable("serve", "HTTP serving is scheduled for the server milestone");
}

}  // namespace inferx::command
