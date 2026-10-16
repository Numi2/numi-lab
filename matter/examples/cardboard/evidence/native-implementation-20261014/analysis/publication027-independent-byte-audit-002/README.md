# Publication 027 independent audit, revision 002

This is a read-only audit of publication027 after the postseal compaction addendum. It validates the exact binding and summary bytes, the 1,424-record historical prefix, the pre-addendum 1,457-record state with only three added `inputs.compaction.json` pointer rows, the 4,811-entry prospective pathspec prefix, all 503 payload rows, the three recovery pointers, the two compaction packages, and the 511 staged Git paths and index blobs.

The prior revision-001 audit remains unchanged. It correctly refused the stale prospective inventory because case 464 had one new postarchive pointer. Revision 002 includes the pointer rows and the two new compaction evidence cases. The added pointer rows extend three old case records; they do not replace prior file rows or rewrite source/native payload bytes. The original 1,424 summary records remain exactly identical.

The result is limited to byte integrity, history/prefix integrity, payload-scope exhaustiveness, and exact Git staging. It does not upgrade any physical, convergence, calibration, analogue, assembly, or impact claim. The six-axis verdicts are reproduced from the binding.

Run with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/publication027-independent-byte-audit-002/audit.py
```

The audit does not mutate the repository index, worktree, external case payloads, or historical references. It writes only `report.json` and `audit.stdout.json` in this package.
