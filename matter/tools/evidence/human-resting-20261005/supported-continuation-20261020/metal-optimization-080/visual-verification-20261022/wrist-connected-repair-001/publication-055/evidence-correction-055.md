# Publication 055 evidence correction

This is an evidence-only bundle for the hand/forearm connected-skin experiments. It changes no runtime source, asset, or accepted geometry. All proposals in this bundle remain unadmitted.

## Results and interpretation

- The 013 source-refinement baseline completed the full exact self-surface audit with 0 exact self-pairs across 38,104 faces.
- 014 run-005 is a finite-witness construction only. It retained 519 moved nodes, up to 43.3013 mm displacement, and a worst emitted constraint slack of -32.8395 mm. Its 39 nonpositive old/new normal cosines are a comparison diagnostic, not by themselves proof of a fold. The candidate was rejected.
- 016 run-001 moved 2,340 nodes by up to 24.9928 mm. The reported 76 nonpositive normal cosines are likewise not a fold proof. The exact predicate found 302 added self-intersection pairs, so the proposal is rejected.
- 017 run-001 retained positive tetrahedron determinants in its bounded proposal but stopped because no safe connected step remained; it was not an exact-contact or native acceptance.
- 017 run-002 has 0 added exact self-pairs against 013, but 78 nonpositive tetrahedra (minimum normalized determinant -0.1182); reject.
- 017 run-003 reduced target residual but produced 198 added exact self-pairs and 2,652 nonpositive tetrahedra (minimum normalized determinant -0.9337); reject. Its 47 reference normal deviations over 90 degrees are recorded only as diagnostics.
- The 018 exact self audit compares each candidate to the same complete 013 surface. 016 adds 302 exact pairs; 017-002 adds 0; 017-003 adds 198. Exact pairs are collision evidence; cosine-only normal differences are not.
- Detailed-skin audit run-001 is retained as a superseded hybrid: it combined identity-004 CSV positions with source-294 topology, so it is not an exact owner-replayed source-294 result. Run-002 is the corrected owner-194 Float32 replay of the pinned source-294 payload at identity-004 body poses. It finds 3,070 transverse skin/bank triangle intersections over the complete 108,987-face detailed skin against 9,964 selected facets in 67 banks. The selected tissue-bank side is incomplete, this is a preparation replay rather than a native capture, and native/physical admission remain false.

The 014 run-005 report references a later-overwritten source filename. Its exact executed bytes are preserved as finite_witness_fit_014-run005-executed.py and are mapped by the manifest. Similarly, 017 run-001 as-run source is preserved as safe_connected_fit_017-run001-executed.py. Original reports and logs remain byte-for-byte unchanged.

## Validation records

The retained unit logs report 3 passing tests for 014, 3 for 016, 3 for 017 incremental fitting, 3 for the current detailed-skin parser, 5 for the lexicographic LP helper, 28 for the registered-pose helper, and 55 for the exact intersection/seam suite when its one unavailable external-source test is excluded. The older exact-suite log also remains: 49 tests pass and one test errors because external anatomy source partof_element_parts.txt is unavailable. No test result here qualifies the geometry as native or physical.

## Remaining gates

No candidate is admitted. The 018 bank subset does not cover all registered tissue banks; no complete hand/fat transfer, all-pose attachment/topology/volume gate, native rebuild/replay, or physical acceptance is claimed. The results are bounded evidence for repair planning only.
