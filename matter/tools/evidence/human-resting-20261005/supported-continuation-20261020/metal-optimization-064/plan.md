# Matched contact-incidence optimization screen 064

Status: preregistered before the new comparison runs. Exploratory performance screen on the real Human contact workload; not a performance or physical qualification.

## Question and model

Does the already implemented NM_CONTACT_NODE_INCIDENCE_SORT=1 deformable-node incidence path reduce the measured cost of contact incidence on the full resting-human mechanical scene, after its padded bitonic-sort cost is included? The alternative is that sorting the six corner records for each contact costs more than the current per-node contact scans.

Owner is Numi Lab Matter (matter/src/runtime.mm, matter/src/metal/contact.metalinc) on numi-human-aortic-contact-perf, source HEAD aadcd68e0747540978edd88a23af0cd114932942. The native executable is build-integrated-validation-20261019/profile-build-039/bin/numi-human-native, Release, SHA-256 c9dab7f23a3f8e14fa56aef2387b0df66036f154d573612ab683ac467e101828. Host is SSH numi (Apple M6, 12 GPU cores, 16 GiB unified memory, macOS 27.0). The exact model command and explicit configuration are copied from immutable audit input metal-performance-audit-063/inputs/062/invocation.json; only fresh output paths and the single treatment environment variable differ.

## Prediction

Primary observable is the native human_integrated_body real-time factor, dimensionless (simulated_seconds / wall_seconds), for four accepted roots at 0.000125 s/root. Secondary observables are reported integrated-throughput RTF and each accepted root's native GPU command time. Two paired repeats are collected in order control, sorted, sorted, control to balance time/order drift.

Prediction: the sorted path gives at least 15% greater median paired human_integrated_body RTF, with both pairs faster in the same direction. It is refuted for promotion if either pair fails to improve, the median gain is below 15%, or any validity check fails. This small screen is descriptive and cannot establish a 0.1x result.

## Intervention and held-fixed variables

- Control: NM_CONTACT_NODE_INCIDENCE_SORT unset.
- Treatment: NM_CONTACT_NODE_INCIDENCE_SORT=1.
- Same four-root initial-state construction, anatomical input files, binary, command arguments, physiology/mechanics flags, accepted-state logic, GPU, and driver in every arm. No checkpoint or viewer capture is substituted.
- Each arm uses an isolated output directory. There are no concurrent native solvers or heavy geometry jobs.

## Validity checks and failure policy

Each trial must exit zero, report exactly four accepted roots and zero rejected roots, finite terminal state, no root assistance, and passing terminal FEM and fascia audits. Record contact count, penetration, FEM minimum/maximum Jacobian and maximum displacement, terminal ligaments, all root command times, and failure receipts when emitted. The sorted arm must retain the same physical acceptance semantics; numerical state summaries are compared pairwise with 1% relative tolerance for nonzero values and 1e-7 absolute tolerance near zero. A failed or missing arm stays in the record as invalid; do not retry or exclude it to manufacture a passing comparison.

## Analysis and stopping rule

Report all four raw RTFs, the two within-pair ratios, their median, per-root command times, process wall time, and validity outcome. Do not treat roots within a trajectory as independent samples. Stop after the declared two valid pairs or upon a failed arm. If the screen passes, keep the path opt-in until it also passes the owning native check and a longer accepted/rejected integrated regression. If it fails or is inconclusive, preserve the data and move to a different measured bottleneck; do not widen the sample after seeing results.

## Exact instrument, inputs, and evidence

The source audit finding, Metal kernel identities, baseline command, explicit environment, binary/source/input hashes, and baseline log are retained under metal-performance-audit-063/. run.py reconstructs the immutable baseline invocation, removes ambient Numi/Matter override variables, applies only the declared arm variable, records the effective environment and command, and retains stdout/stderr/status/timing for each arm. The native check is numi-matter-cardboard-contact-node-incidence-sort-native-check from the same Release build tree. No visual verification is part of this screen.
