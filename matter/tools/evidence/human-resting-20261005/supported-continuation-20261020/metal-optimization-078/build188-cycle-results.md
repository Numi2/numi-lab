# Completed-cycle work integration: build188

All builds and native execution: SSH Numi, Apple M6. No visual verification.
Default flags remain unchanged. Minimum 0.1x end-to-end is not achieved.

The GPU now prepares explicit ranges for distributed restart initialization,
initial-basis scaling and correction accumulation. Suppression requires healthy
finite completed state; active, nonfinite and unhealthy states retain work.
Scalar initialization, output clearing, current-operator action and true-residual
certification remain mandatory. Opt-in: NM_FGMRES_SKIP_COMPLETED_CYCLE_WORK=1.
It can combine with NM_FGMRES_SKIP_COMPLETED_ZERO_OPERATOR_ACTION=1.

Metal API validation: 177 focused cycle-range checks and 69,751 generic native
checks pass. The source certification contract passes 22 checks; the CPU regional
layout checker passes its fixtures.

Three native roots (125 microseconds each), one matched triplet with API
validation enabled, retain all 17 endpoint groups exactly:
- control body seconds: 7.021883166, 4.805349916, 3.478055791.
- cycle suppression: 7.157012333, 4.804062792, 3.482565458.
- cycle plus zero action: 6.897834542, 4.750463917, 3.372779125.

Post-first body seconds: 8.283405707 control, 8.286628250 cycle-only,
8.123243042 combined. Cycle suppression alone shows no improvement. Combined
throughput is 1.0197166 times control in this single triplet; it does not isolate
an additional gain beyond the separately tested zero-action optimization.
Work counts and physical endpoints are identical. These are startup mechanics
intervals, not settled physiology or end-to-end presentation performance.
Uninstrumented validation-off timing from build186 remains about 0.0000354x.

Compile failures and repairs are retained with source closures and logs.
No unsupported accuracy gate or numerical tolerance was weakened.
