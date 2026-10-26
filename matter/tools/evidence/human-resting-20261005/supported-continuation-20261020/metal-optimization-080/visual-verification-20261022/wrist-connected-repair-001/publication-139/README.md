# Focused connected-skin topology repair from candidate 538

Continue the existing connected surface, its coordinates and back repair. Apply local edge-flip batches jointly across the six dominant typed bone banks, then remove only operations implicated by exact self-intersection witnesses.

| Checked candidate | Surface crossing pairs | Contained node-bank pairs | Self pairs | Retained flips |
| --- | ---: | ---: | ---: | ---: |
| 538 baseline | 2981 | 267 | 0 | — |
| 541 topology run003 | 2927 | 267 | 0 | 71 |
| 541 topology run005 | 2900 | 267 | 0 | 53 |
| 541 topology run007, selected | 2892 | 267 | 0 | 34 |

The selected NPZ is work/topology/run-007-filtered/candidate.npz, SHA-256 a1fce94c3dc57b79f16f94b117e7d49e83a0ff7219de73ce6e7abcec17f22dae. All point coordinates remain bitwise identical to 538. All 237 typed anchors, outside-patch connectivity and ancestry, boundary edge set and manifold edge incidence remain preserved. Fresh Float32 area, current-topology edge, trust and source-distance construction gates pass. Exact audits cover self-intersection, contained points and all 185 bone banks. The final bound receipt is work/topology/check-007-arithmetic/bound-exact-construction-receipt.json.

89 fewer surface pairs is real construction progress; 2892 crossings and 267 contained pairs remain. Two banks increase by one pair in the last batch despite the net decrease. Stop topology iteration after measured reductions of 54, 27 and 8; coordinate correction remains necessary.

Failed unfiltered batches contain respectively 72, 102 and 109 self pairs. Their candidates, exact witnesses and operation manifests remain here. The partial interrupted final 159 audit is preserved but unused; the complete rerun is bound by the final receipt.

The reusable check entrypoint invokes existing 408 arithmetic, 130 exact self/inside and 159 typed-bank owners. It regenerates edge rows from current connectivity. The coordinate-proposal preparation helper reuses 423 local backtracking, masks correction to witness regions and preserves all other state. Historical absolute paths and source pins are retained for exact replay in the existing SSH Numi workspace; referenced prior owners and the 538 baseline were retained in earlier publications.

Root tests: 33 passing checks for typed-bank completeness, stale reports, topology/anchor invariants, Float32 budgets, witness scope and filtered-operation replay. See the retained command and source hashes in root-test-receipt.json.

This is offline anatomical construction evidence. Native admission is false and accepted physical steps are zero. Skin/fat volume transfer, whole-body clearance, dynamic validation and integrated throughput remain unqualified. The existing 45 head boundary edges have not been silently treated as closed skin. No simulation force, material law, mass or controller was changed by these topology operations.
