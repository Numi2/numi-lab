# Native crease-tooling analysis

This package binds four distinct native tooling runs to fail-closed analyzer
reports and SHA-256 snapshots of the analyzer source. The reports separate
certificate-verified accepted-step gaps and punch-force receipts from rejected
or unverified attempt diagnostics. After rejection these gap diagnostics can
pair the attempted tool endpoint with rolled-back accepted FEM geometry; they
are not an accepted endpoint or a recovered rejected FEM candidate. The signed-distance checks sample FEM
nodes at each commanded end pose; they are not a continuous surface-contact
proof. The FP32 allowance is `8 * float_epsilon * max(length, width, height,
punch radius)`. Contact slop is a barrier range, not an allowed penetration.

| Run | Result | End-pose finding |
|---|---|---|
| `crease-contact-001` | Failed after 17 accepted steps; analyzer verdict is inconclusive. | Accepted step 16 had a −0.432 µm punch-node gap. Rejected step 17 diagnostics reached −1.057 µm. |
| `crease-endpose-guard-002` | Failed after 17 accepted steps; analyzer verdict is inconclusive. | The independent caller-side audit reported the same −0.432 µm accepted endpoint after the state had already been accepted; this did not roll that state back. Guard run 001 has matching hashes for the five report inputs listed in the index; run 002 additionally captures the failure-end material states. |
| `crease-endpose-native-001` | Native post-commit endpoint check rejected the next step after 16 accepted steps; analyzer verdict is inconclusive. | Accepted endpoints stayed positive, with a 0.091 µm minimum. The attempted endpoint against the rolled-back FEM state measured −0.534 µm. The rejected-step indented OBJ is byte-identical to the preceding accepted OBJ (SHA-256 `389a880a2be45d1974c5a638a5bb5789e737e954008f795aa268dc1606c95794`), confirming retained FEM state rollback for this case. |
| `shallow-contact-cycle-001` | Completed 96/96 steps; analyzer instrument checks passed. | At 10 µm maximum commanded travel, the minimum sampled accepted gap was +0.187 µm and the moving punch had 56 native contact samples. This is a shallow contact cycle, not a crease or fold qualification. |

The common contact slop was 1.91 µm and the probe-matched roundoff allowance
was 0.0301 µm. A nonzero signed force is an instrumentation receipt derived
from native punch impulse divided by the timestep; it does not establish
material, fold, or physical validity. None of these runs qualifies cardboard
forming, a crease, a 90-degree fold, or a box.

`package-index.json` maps every report to its raw run, invocation receipt,
binary/input bindings, source patch hash, and report checksum. The receipts
record unchanged inputs. `source-snapshots/SHA256SUMS.json` binds both analyzer
versions and the shared 9-test stdlib suite. The preserved
`reports/analysis-attempts/` JSON files record an initial analyzer timing-schema
rejection; they are not run measurements or final reports.
