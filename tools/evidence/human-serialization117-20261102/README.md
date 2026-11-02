# Human serialization audit and parallel contact kernels

Two production contact paths now distribute work over 256-thread tiles and
hierarchical reductions. The audit visits every original contact and attachment.
Body wrenches use stable per-body indices instead of each body rescanning every
sample. Source-order membership, force law, current geometry queries, opposite
forces, torque, mass, capacity rejection and accepted-state ownership remain.

The current source scene has 318,974 FEM nodes, 1,495,294 skin/fat tetrahedra,
15,453,677 body-contact rows and 128 generalized DOFs. No physical rows were
removed. The retained serial paths serve small workloads and numerical controls.

## Confirmed additional serialization

| Path | Source finding | Execution scope and next change |
| --- | --- | --- |
| Human provisional status | One thread validates all nodes and four load slots per node. | Normal adapter graph. Tile finite/identity checks with deterministic invalid witness. |
| Tendon-force and anchor audits | Each environment sums all nodes on one thread. | Normal adapter graph. Use finite-aware hierarchical reductions. |
| Anchor generalized reactions | Each DOF rescans all nodes, twice across pending/apply paths. | Normal reaction graph; about 40.8 million anchor tests per pass. Stable active-anchor incidence removes inactive scans without changing arithmetic order. |
| Vascular/support working-set reset | One SIMD group clears whole coupled node arrays. | Conditional on a changed working set. Separate bounded decisions from tiled predicated clearing. |
| Regional restart energy | Lane zero scans all node basis/operator products. | Only admitted restart augmentation; not requested in trial120. Preserve compensated energy and sign gates when distributing. |
| Topology incidence rebuild | Each node scans every tetrahedron in both count/mass and scatter. | Topology rebuild/growth only; not established active here. Replace quadratic traversal with deterministic corner incidence. |
| Blood MPM handoff | A single thread repeatedly searches all particle pairs for conservative merges. | Disabled in this trial. Worst-case cubic work at saturation; use existing spatial bins and deterministic compatible-pair selection. |

The tangent-finalization full-tet scan is a diagnostic branch. Ordinary
finalization is already per tetrahedron. Armijo measurements, determinant
validation and global FGMRES measurements already select distributed paths.
The loop inventories are source candidates, not a count of measured problems.

An additional two-second native CPU sample at roughly 135 seconds found the
main thread in skin/fat exact geometric predicates during initialization. That
is separate from per-step GPU cost. Small body, muscle and compartment loops
have not been promoted to bottlenecks solely because they contain loops.

## Verification and measured limits

Production and publication-source Metal compilation passed without fast math.
Serial/tiled comparisons cover zero rows, 255/256/257 boundaries, 8,193 and
262,145 rows, two environments, active/inactive contacts, malformed sample and
attachment rejection, all audit fields and body forces/torques. Maximum audit
relative difference was 0.0420067%, within the existing 1% comparison gate.
Native adapter controls passed accepted deformation, invalid-step rollback and
bitwise replay with both parallel paths forced on.

All timing trials are retained, including variation between builds. Isolated
large-row audit and wrench kernels improved by hundreds of times. These use a
controlled elastic-contact fixture; they are not full triangle-BVH Human
throughput or a measured contribution to integrated speed.

Trial118 stopped before GPU execution because cleanup114 had misclassified
two transitive mass-proof inputs as superseded. Both were independently
downloaded from the retained GitHub archive, archive/member hashes verified,
restored byte-for-byte, and explicitly added to trial120 input pins.
Trial120 reached the parallel layout with all 15,453,677 solver rows, then
stopped at 271.0 seconds under the unchanged +3 GiB swap-growth guard.
No physical step completed and no end-to-end real-time factor is established.
No source/cell/contact/mass or physical acceptance tolerance was relaxed.

Optional shared-event progress markers report CPU notification arrival,
not GPU kernel duration. They perform no physical-state readback and are off
by default. Their instrumented controls pass; full-scene GPU progress remains
unmeasured because trial120 stopped during initial adapter preparation.

Exact source pins, build commands, all control outputs, failed builds, trial
plans, resource traces and restoration receipts accompany this report.
The next work is resource-admitted full-scene stage measurement and the active
node reductions/anchor incidence before dormant topology or blood paths.

Apple recommends enough independent threadgroups and measurement of actual
limiting stages: https://developer.apple.com/videos/play/wwdc2022/10159/ .
