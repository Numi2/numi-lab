# Base CGLS regional selection metadata overlay 156

This external overlay corrects the meaning and validation of CGLS-primary
selection metadata. It is based exactly on the immutable source-151 snapshot
and does not modify that snapshot, the shared checkout, solver equations,
runtime selection policy, or acceptance criteria.

The regional capture is taken before `prepare_base_cgls_layout`. At that point
the policy kernel's `armijo_metrics.w == 3.0` marker means that the same-Newton
eligibility gates passed: start iteration, healthy production status,
original Newton certificate, and base residual above the unchanged residual
bound. It does not mean that the later per-environment `femBaseCGLSSelected`
layout word has been written. Schema v2 records the policy request, whether
the policy kernel is dispatched for this Newton, its start iteration, and the
eligibility semantics; it deliberately does not serialize the pre-layout
selection scratch as final selection.

The reader derives eligibility from the bound source-151 state payload and
rejects marker/gate inconsistencies. It can separately accept the later
`fem_base_cgls_direction` trace only when the trace is tagged
`after_prepare_base_cgls_layout`, matches the regional root/Newton/budget, and
its final selected words agree with the same-generation eligibility markers.
The trace is optional; without it, final CGLS selection remains unavailable.
Legacy v1 is read only with its old selection array treated as stale
pre-layout scratch, never as final selection. The one-object static FEM scope
is enforced because the producer policy kernel is a no-op outside that scope.

Validation is limited to CPU reader contracts and Objective-C++ syntax
against source-151 headers. No native Metal execution, GPU solver result,
regional action parity, or physical qualification is claimed.

## Files

- `candidate/matter/src/runtime.mm`: schema-v2 metadata and post-layout trace
  labeling.
- `candidate/matter/tools/read_regional_capture.py`: strict v1/v2 reader and
  separate eligibility/final-selection reporting.
- `candidate/matter/tools/test_read_regional_capture.py`: synthetic CPU
  contracts, including stale legacy scratch, requested-but-not-invoked,
  eligibility mismatch, final-selection mismatch, and producer scope checks.
- `patch/source151-relative.patch`: patch from the exact source-151 snapshot.
- `manifest.json`: source and output hashes, validation commands, and limits.
