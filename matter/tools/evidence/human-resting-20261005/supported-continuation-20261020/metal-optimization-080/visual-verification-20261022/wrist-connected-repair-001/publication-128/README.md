# Connected skin clearance: tested construction increments 410–416

This publication contains offline construction tools and exact geometry evidence. It does not alter the native simulation owner. Native admission is false and accepted physical steps are zero throughout.

## Measured outcomes

The prior published checkpoint 408 had 3,302 skin–bone surface pairs, 425 contained node–bank pairs, and zero skin self-intersections.

- Checkpoint 410: 3,268 surface / 406 contained / self 0. All 169 preexisting non-coordinate arrays and 237 typed anchors remain byte-identical. No introduced contained pairs; 103 introduced surface identities accompany the net surface-count reduction.
- Checkpoint 413: 3,261 / 400 / self 0. A 0.25 mm construction proposal required uniform reduction to 0.0312554 mm. No introduced contained pairs; 38 introduced surface identities.
- Topology alternative 414: 3,254 / 391 / self 0. Two local flips remove four exact self-pairs from the larger 413 pass-002 proposal in 17.09 s and four trials, with no introduced self-pairs and all vertex positions unchanged. Three final face rows change (17449, 17450, 17454); explicit immediate-parent union ancestry is regenerated. Full 185-bank audits and independent root checks pass. The 237 anchors, 45 open head boundary edges, manifold orientation and matched-037 area floor remain intact. There are 110 introduced and 124 resolved surface identities among unchanged faces versus 410; changed-face identities are not treated as the previous triangles.
- Same-topology continuation 415: 3,255 / 391 / self 0, preserving 169 non-coordinate arrays and 237 anchors. The retained step is 0.125013 mm, four times the 413 retained step; this is a construction-step comparison, not a runtime speedup. No introduced contained pairs; 110 introduced surface identities. This is the continuation baseline because it avoids a topology transition for one additional surface pair. Alternative 414 is retained independently.

Zero crossings have not been reached.

## What changed

Resolved crossings now become fixed-boundary guards instead of being forgotten. The 408→410 refresh adds 137 resolved pairs to 942 historical guards, excludes all current crossings, and rebuilds 6,474 zero-feasible rows for 1,079 guards. The reusable CLI reproduced the previous tested NPZ byte for byte.

A witness-focused prospective scan queries 271 faces against the entire surface index, obtaining 38,156 new pairs rather than the whole-body 1.4 million candidate set. All 56,225 union pairs have current-coordinate rows; none are silently dropped or incompatible. However, solving all 323,528 pair rows timed out at the 90-second first-stage budget without a primal candidate (checkpoint 412). This failed trial is retained.

The active selector preserves all known failure/history guards and includes prospective pairs threatened by the actual previous proposal. The clipped 4× prediction selects 22,144 pairs and 125,214 rows, deferring 34,113 explicitly. The 415 solve succeeds in 114.27 s including preparation, versus the full-union 412 no-proposal result. Exact self, area, point containment and all 185 surface banks remain mandatory after selection.

The full 415 proposal has eight self-pairs, compared with 32 for 413. A separate bounded local flip test (416) makes no improvement: eight remain after one eligible trial, with no accepted flips. Its next change is adding those exact witnesses to the next connected solve, rather than repeating a larger flip search.

Reusable same-topology input preparation validates exact current candidate/audit/source hashes, Float32 points, bytewise ancestry including intentional NaNs, and seed coverage. The point-depth CLI correctly validates sparse pair tables using per-node complete 185-bank accounting, including AABB exclusions. Actual replay reproduces the frozen 410 point-row NPZ byte for byte. The erroneous dense-table check and an initial owner-report-schema failure are recorded; the reconstructed old source is explicitly not claimed byte-identical.

The new flip path checks the matched-037 Float32 magnitude floor on the baseline and every changed face. It retains full-surface cached-index exact checking; increasing local search scope does not weaken global self-intersection acceptance.

## Verification and recovery

53 focused tests pass: 7 original preparation, 14 extended preparation, 10 area/prospective guards, 7 fixed-guard refresh, 9 sparse point-depth refresh, and 6 active-set selection tests. Actual NPZ parity receipts and independent geometry checks are separate evidence.

Sources, commands, exact source/asset/binary hashes, complete reports, failed attempts and arrays are retained in the two manifest-bound archives. Each archive is locally restored and checked for content hashes, sizes, modes, owners and nanosecond mtimes before publication. Independent GitHub recovery receipts are retained after publication. The shared dirty checkout and normal Git index are not used for staging.

## Remaining physical scope

The candidate is not accepted physical skin. Inner-fat cell validity, thickness and regional mass, conservative mechanical transfer, the eight retained moving poses, and complete native 16/60-second qualification remain pending. The 45 known head boundary edges remain open. Mixed-source anatomy remains a reference assembly, not measurements of one individual. No visual inspection or native simulation was performed in this increment.

The first retained archive is 53,716,468 bytes. Its initial 50 MiB packaging bound failed after writing; the explicit bound was raised to 64 MiB and the existing archive was independently streamed and verified before reuse. This was an archive-size diagnostic, not a geometry gate change.
