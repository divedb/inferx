# Known limitations

M1 is deliberately a control-plane correctness milestone:

- execution is synthetic and CPU-only; no tensor storage, device allocator,
  CUDA stream, kernel, or physical completion fence exists until M2;
- model ID 0 with token input and a 32,768-token capability is the only fake
  model; tokenization/artifact loading begins in M3;
- capacity reserves prompt plus maximum output for the full request lifetime,
  favoring simple accounting over utilization; M8 owns incremental policies;
- prefill is not chunked, priority and non-default tenant scopes are rejected,
  and FCFS does not bypass a budget-blocked head request;
- simulator throughput and fake latency do not predict GPU throughput;
- replay schema v1 is platform-stable for supported GCC/Clang Linux builds,
  but it is not a production request/audit format;
- process signals, network serving, authentication, and graceful threaded
  shutdown are outside this simulator milestone.

These limits must not be worked around by bypassing the lifecycle controller,
resource transactions, plan leases, or completion identity checks.
