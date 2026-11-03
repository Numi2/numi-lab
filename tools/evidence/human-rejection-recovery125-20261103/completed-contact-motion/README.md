# Completed solver work and motion contact

The optimized full Human initialization replay passes four accepted steps with a byte-identical mechanical endpoint and physiology CSV. In a matched comparison using the same binary and detailed timing disabled, mean GPU time falls from **13.560 s to 8.298 s per step**, a **38.80% reduction / 1.634x throughput**.

This is a short equivalence and performance screen. It does not establish supported rest, a complete motion cycle, sustained throughput, five-minute physiology, or 0.1x real time.

## Changes

1. **Exact-input GPU contact cache.** Retain derived body-contact rows only when the node position/mass and body pose are finite and bit-identical to their prior inputs. First use, any motion, changed restore/retry inputs, or nonfinite values invoke the existing exact query. Invalid retained rows stay invalid. Source tables must remain immutable under the existing row-cache admission.
2. **Compact current exact queries.** Apply the existing closed/oriented-bank AABB certificate, then stably compact rows needing exact BVH traversal. Cooperative 256-lane tiles, hierarchical scans and GPU-generated indirect dispatch arguments eliminate expensive inactive query invocations. Uncertain, open or invalid cases retain the exact query. Signed-zero force arithmetic is preserved. Extra index scratch is approximately 61.8 MB for this scene.
3. **Reuse only completed Newton geometry.** GPU-latched tails retain frozen deformable contact geometry while refreshing current friction history and Hessian through the same helper as full narrowphase. Active Newton iterations rebuild normally. Final certification always uses the original full narrowphase.
4. **Suppress completed tangent finalization.** Use the existing Newton indirect work domain; forced rebuilds remain full.
5. **Measure adapter kernels.** Optional timing uses the adapter's existing encoder boundaries.

Collision floors, force ownership, material equations, precision, mass mappings, accepted-state processing and final physical acceptance checks remain unchanged.

Enable in the retained launch configuration:

    NM_CONTACT_COMPLETED_GEOMETRY_CACHE=1
    NM_MATTER_CONTACT_ZERO_STEP_GEOMETRY_REUSE=1
    NUMI_HUMAN_CONTACT_ROW_CACHE=1
    NUMI_HUMAN_CONTACT_AABB_EXCLUSION=1
    NUMI_HUMAN_CONTACT_EXACT_MOTION_CACHE=1
    NUMI_HUMAN_CONTACT_COMPACT_QUERIES=1

Admission fails outside supported immutable configurations. No unsupported path is enabled by bypassing restrictions.

## Native verification

- Focused contact check: **99,409 checks pass**. Includes moving nodes, body translation/rotation, two environments, restore/retry, nonfinite input, deterministic compaction across two scan blocks and partial tiles, all-active/mixed/empty queries, invalid metadata, active rebuilding, and four current friction histories. Direct force/validity and cached-contact comparisons are bitwise.
- Full native FGMRES true-residual/kernel contract: **916,443 checks pass**, including those focused checks.
- Four existing Newton-tail runtime fixtures pass bitwise accepted-state/certificate parity, late rejection rollback, replay and static completed tails.
- Native trials 042, 043 and 044 each accept four 15.625-microsecond steps. The complete **833,232,924-byte** node/tetrahedron endpoint and physiology CSV match control 035 exactly. No failure receipt is emitted.

Endpoint SHA-256:

    66c476e14b867eb9a19039ec112750a3d517a18ce039b24bdb5181fec7ac24c7

Physiology CSV SHA-256:

    77edb932a1a06eacdba8efe1f97bc5492ff8185c37a2e31a5c8d09045eb41aee

Failed test attempts 040/041 are retained. Their manufactured active feature had missing primitive-count metadata and initially lay outside the activation band. The fixture was corrected; physical/admission gates were unchanged.

Test entrypoints, using the tested execution checkout:

    build-native/matter/numi-matter-fgmres-true-residual-native-check --completed-contact-only
    build-native/matter/numi-matter-fgmres-true-residual-native-check
    NM_FEM_MATERIAL_TAIL_ICB_CHECK_ONLY=1 build-native/matter/numi-matter-fem-material-frame-check matter/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter

## Matched measurements

SSH Numi: Apple M6, Mac18,5, 16 GiB, macOS 27.0 build 26A425.

| Measurement | Control 043 | Optimized 044 |
| --- | ---: | ---: |
| Accepted steps | 4 | 4 |
| Mean physical GPU time per step | 13.560 s | 8.298 s |
| Simulation-loop wall time | 61.878 s | 41.246 s |
| Simulated time | 62.500 microseconds | 62.500 microseconds |
| Simulation-loop real-time factor | 0.000001010x | 0.000001515x |
| Inclusive launch/preparation/execution | 237.406 s | 217.311 s |

Both use the same executable and physics libraries, with detailed GPU timing disabled. Existing work counters, progress markers, native command timestamps, full endpoint capture and readbacks remain enabled. This is one matched pair, not a three-pair or presentation-inclusive qualification. The JSON comparison records every changed setting and output path.

Initial screen 035/042 measured a similar 39.13% GPU-time reduction. Trial 042 used detailed contact timing and cold pipelines; its inclusive elapsed time increased to 357.732 s versus 237.358 s for 035. Do not promote that initial GPU gain to startup improvement.

Root-0 timing from 042 attributes approximately 3.315 s to linear work, 1.643 s to the final candidate/certification stage, and 1.038 s to exact body-contact queries. These remain substantial costs. Completed-tail narrowphase calls take approximately 5.5 ms versus approximately 84 ms for active calls. Nested kernel and whole-stage spans must not be added together.

Profile 039 confirmed that the prior zero-step geometry cache already hit 13 times. A prior interpretation of zero hits was wrong; narrowphase still ran 17 times. The new cache removes that measured duplicate geometry calculation.

## Evidence and limits

The five published source files match the tested execution files exactly. Plans retain runtime, asset, material and source hashes, configuration, commands, complete dirty execution identity and source snapshots. Full stdout remains on Numi at paths and hashes recorded in each retained-full-stdout receipt; compact copies omit only node/tetrahedron rows. All unique failed evidence and unrelated dirty sources remain preserved.

Inherited plan fields describing older experiments or PREPARED status remain historical; the current predeclared checks, effective environment, execution receipt and native accepted-step report establish this experiment's actual scope. The generic runner intentionally leaves physical_admission false; four accepted steps are not full physical qualification.

No changes here qualify the pre-existing long-run convergence/conditioning problem, supported-rest checkpoint or a larger timestep. **0.1x remains unmet.** The existing full-resolution skin/fat and fixed-boundary cavity candidate are unchanged.

Generated outputs are hard-linked only after complete hash and fresh inode verification. Sources, unique evidence and all retained paths remain present. No native job remains active after these checks.
