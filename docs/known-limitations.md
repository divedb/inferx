# Known limitations

- execution is CPU-only and float32; the local `run` path executes one request
  at a time — batched and GPU execution arrive with later milestones;
- model ID 0 with token input and a 32,768-token capability is the only
  in-process model record; broader model support grows with the input and
  artifact milestones;
- capacity reserves prompt plus maximum output for the full request lifetime,
  favoring simple accounting over utilization; incremental policies are a
  later milestone;
- prefill is not chunked, priority and non-default tenant scopes are rejected,
  and FCFS does not bypass a budget-blocked head request;
- process signals, network serving, authentication, and graceful threaded
  shutdown are outside the current milestone.

These limits must not be worked around by bypassing the lifecycle controller,
resource transactions, plan leases, or completion identity checks.
