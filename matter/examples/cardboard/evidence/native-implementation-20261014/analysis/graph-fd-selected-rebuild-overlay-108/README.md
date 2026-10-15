# Selected FGMRES capture rebuild overlay for source 107

This independent overlay is based on the frozen source snapshot
`/Users/home/cardboard-implementation-20261014/snapshots/source-ee-ccd-continuation-107`. It modifies only `matter/src/runtime.mm`,
`matter/src/regional_graph_schwarz.hpp`, and the already-registered CPU
contract test `matter/tools/regional_graph_completed_tail_check.cpp`. The
frozen snapshot and shared checkout were not edited.

The old runtime passed `femOperatorFDTraceEnabled` as the graph completed-tail
reuse exclusion for every Newton linearization in the transaction. This patch
computes whether a validated capture is pending for this exact control step
and solver iteration, and excludes reuse only there. Iteration zero, explicit
cache rebuilds, and the selected capture iteration still rebuild exactly as
before. Other iterations retain existing latch/factor-valid checks. Invalid or
disabled selectors never mark a selected iteration pending; selector parsing
and scope admission remain unchanged.

The runtime change has no kernel, physical operator, acceptance, candidate, or
persistent-buffer ABI change. Existing source contract test target remains the
registration point; no CMake change is needed. The strict contract test passed
19 checks, and Objective-C++ syntax checking passed against frozen-107 include
roots. No Metal compilation or native/GPU execution was performed.

`source-overlay.json` binds the exact base and overlaid file hashes.
`source-overlay.patch` is the complete patch; `patch-dry-run.log` records
application against temporary copies of the bound base.
