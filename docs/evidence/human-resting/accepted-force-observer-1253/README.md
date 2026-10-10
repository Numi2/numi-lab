# Accepted-step force observation

The native resting viewer can record MyoSim generalized-force contributions
alongside a bounded accepted-Q audit. This exposes the loads used during a
transition without adding a force owner, a CPU physics path, or a GPU readback.
The switch defaults off.

Use the existing COM/Q observer settings with NUMI_HUMAN_ACCEPTED_FORCE_AUDIT=1.
The force observer requires both Q window bounds. For the short retained test,
the inclusive window was accepted steps 464–512 at the native F32 2 ms timestep.
Both comparison arms used that same Q submission schedule; only the force switch
and output paths differed.

resting-com-q-force-contributions.csv contains three separate arrays in local-v
order:

- CPU double sums of the already collected per-muscle generalized-force rows.
- CPU double sums of the separate tendon-transfer correction rows. These replace
  source-point application with distributed attachment projection; they are not
  complete tendon loads.
- The unchanged F32 post-consumer MyoSim force-workspace slice. Optional consumers
  may add to this slice after the initial muscle reduction.

These columns are not independent forces to add together, and none is a total
or net body force. CPU double sums describe native rows and need not equal the
GPU F32 reduction bit for bit.

A row's accepted step/time labels the endpoint of the transition. Its
source_pre_step_q_fingerprint_fnv64 binds the load evaluation state to the
corresponding Q audit's pre-step state at (accepted_step - 1) * dt. The bounded
window submits one physical step at a time. The existing Q index map gives
local-v indices; this binary's optional names are "unnamed", so anatomical names
and joint types must be resolved from the pinned rigid-model manifest.
Translational generalized effort is N; angular generalized effort is N·m.

## Verification scope

The focused CPU test covers row order, separate correction ownership, distinct
post-consumer values, invalid shapes/counts, count overflow, nonfinite inputs,
and zero tendon bindings. The native application was compiled with warnings as
errors and linked to the retained ABI-compatible Metal runtime.

Two 512-step runs used the current FHL-corrected anatomical payload, original flat
bed, 72 kg reference mass, 64 contact iterations, released initialization, and
rigid hands on the SSH Mac mini. Both exited successfully, with immutable input
pins unchanged and native arguments matching the owner preview. This is a
1.024-second observer comparison, not a long-duration anatomy, physiology,
performance, or drift qualification. The separate numerical verification report
records exact trace and MRV comparisons.

Retained external evidence root:
/Users/n/numi-human-retained-delivery-20261009/accepted-force-observer-1253.
The failed first compile is retained under build-001; build-002 corrected an
out-of-scope CSV string helper call and compiled successfully. All simulation
and verification ran on the SSH Mac mini.

The complete-human deliverable remains open: late skin/target intersections,
passive source-mesh defects, the cause of long-duration postural motion, and the
final matched baseline/intervention anatomy qualification are unresolved.

## Measured parity

The [final verification report](retained/paired-smoke-001/review-004/verification.json)
passed: all seven common CSVs and both complete MRVPACK captures (steps 0 and 512)
were byte-identical between arms. All 49 force rows matched their Q row's step,
accepted endpoint time, and pre-step Q fingerprint. Each row contains three
finite 128-value vectors for 416 muscles and 832 tendon bindings. The verifier
checked 79 unique inputs before/after. The repository keeps one copy of the
identical shared traces plus the force trace; full captures and recordings remain
at the external evidence root, identified by the reports.

The first verifier attempt failed on an import/name typo. Revisions 002 and 003
passed their comparisons but had incomplete input pin coverage. Revision 004
additionally pins the force CSV and both source manifests. Earlier attempts are
retained, not relabeled as complete verification.

Native binary: b53e781562a9b6671bfb0f0142461a230d5f118fcffb499cbe36cf58ac03d0f6.
Runtime library: 8edaa58f5d53645e9e5e608772a134a5db58fc3bbdae58604a3cc847edfa8b61.
The exact source hashes, original commands, arm receipts, build commands, and
retained file identities accompany this record. Wrapper wall times were
17.855885959 s (off) and 17.093004416 s (on), including initialization and capture;
these short timings are not a performance comparison.
