# Publication 093 — barycentric connected-clearance continuation

This is a bounded evidence freeze for completed barycentric connected-clearance work. It is based on origin/main `e0be15ee7f147693ff98dd89c34dfdb9c74adef7`. No geometry or native admission, accepted physical steps, commit, or push are claimed here.

The target lineage is retained explicitly. The 171 virtual-point output used the wrong vector/frame transform and is archived as failed scope. The 172 output applies the corrected raw-vector transform: for a primal sample-to-contact vector it uses `solve(M, local_vector)`, not the covector inverse-transpose. Both full target row sets and summaries are included, alongside their source candidate and input pins.

Completed 170 outcomes:

- 001 stopped after the upstream wrong-frame issue.
- 002 reached no accepted SCP step under infeasible hard-plane constraints.
- 003 reached no accepted step after its exact self-intersection gate rejected the trial.
- 004 produced a construction-only candidate that passed its full exact self audit (zero PLC surface pairs); reported bank/surface pair count decreased 3,784 to 3,687, and inside node/bank pairs decreased 19 to 14. Its selected candidate and complete 185-bank point/surface receipts are included.
- 005's immutable as-run fit report remains rejected with no accepted SCP step. A separate parent selection receipt records that the saved alpha 0.125 backtracking candidate was selected after its complete joint audit. It reports 3,658 surface pairs and 18 inside node/bank pairs, versus 3,687 and 14 for 004; total pairs decrease 3,701 to 3,676, while maximum and squared penetration depth are nonincreasing. This is construction progress only, not a successful fit or a physical step. The candidate hash and selection receipt are pinned in the bundle.

The canonical `human_skin_fat_bone_audit.py` owner and focused tests are included with their fixtures and exact-segment predicate dependency. The staged focused suite passes 54 tests. The owner validates candidate/query identity and complete bank reports, and provides construction-progress comparisons; it does not emit construction planes or certify closed-solid containment.

The exact audits certify only their stated point and surface checks against the queried banks. They do not certify all-edge/tet clearance, tissue mechanics, mass transfer, or runtime behavior. Candidate alternatives are omitted from the payload but their original paths, sizes and SHA-256 values are indexed in `evidence/170/excluded-candidate-hashes.json`. No active 006 files are included or inspected.
