# M1 scheduler and resources

The M1 scheduler is a deterministic CPU control-plane component. It receives
an arrival-ordered `SchedulingSnapshot` of value-only
`RequestSchedulingView` records; it never stores request pointers or spans
beyond the call.

The scheduling path is:

1. `AdmissionController` classifies queued work as reservable, temporarily
   blocked, or impossible under the conservative full-context cost.
2. `ResourceAccountant::BeginReservation` creates one tentative transaction.
   `Commit` publishes counters without failure; destruction rolls back.
3. `FcfsPolicy` writes eligible indices into preallocated caller scratch.
4. `StepPlanPool::Acquire` leases a fixed slot/generation.
5. `BatchPlanner` emits a whole-prompt prefill or one-token decode per item,
   stopping at the first token/sequence budget boundary.
6. `StepPlanValidator` proves the plan still matches snapshot identity, state,
   epoch, reservation, ranges, model, and resource totals before submission.

The fake executor owns the move-only lease until all completion items are
delivered and the ticket is acknowledged. Slot generations make stale reuse
observable. `plan_buffer_slots` bounds in-flight plans; M1 normally needs one
because it submits at most one plan per timestamp.

`inferx_scheduler_benchmark` covers prefill, decode, mixed, and saturated
snapshots at 1/32/256/1024 requests. Its measured region includes select,
acquire, and build and reports `allocations`, p50, p90, and p99. The M1 gate is
zero allocations and p99 ≤ 1 ms at 1024 requests on the recorded reference
machine.
