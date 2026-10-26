# Publication 148: offline topology-chain checkpoint

This checkpoint reduces the surface-crossing count from P147's 2462 to 2409 with a four-round topology-flip chain. The selected candidate is SHA-256 eadcd3838ec5bf4f2d72d45ba5d19c775e9fef4361f92ebea6f49e18d4bfe3a5. Its completed exact report is final-check/report.json (SHA-256 27c01ad4033b9b1181ada53cff9f9aca27799e637741c1fced969effab78e979).

## Chain results

The chain starts from the P147 candidate 211e0e6c422761633ea5e81c5eaf8ee91797a4cf9c5444bf37e4c6bd1f126b38 and keeps each round's launch, execution log, candidate, state, progress, incremental self certificate, and composed exact-159 inventory.

- Round 1 accepted 17 operations and reduced crossings 2462 → 2423.
- Round 2 accepted 5 operations and reduced 2423 → 2411.
- Round 3 accepted 1 operation and reduced 2411 → 2409.
- Round 4 accepted 0 operations; it preserves the 2409 candidate and records the no-change endpoint.

The chain summary says it was pending one final full exact audit, because that summary predates the separate full audit stage. The retained final-check report and audit files are later evidence: the complete exact-construction report counts 2409 surface-to-bone crossings, zero contained-node/bank pairs, zero self intersections, and all 185 typed banks. The final full audit took 35.96 seconds. The completed 159 audit is separately pinned and complete.

The method states that coordinates and typed anchors are bitwise unchanged across each batch. The topology changed through accepted face operations; the final arithmetic report therefore has a zero coordinate step. This is a topology-construction result, not point displacement or physical motion. The maximum source distance remains 11.9944701235 mm under the existing 12 mm source cap; all 237 typed anchors remain represented. The full-F32 area check passes all 41,733 faces, with current oriented ratio 1.0 within floating precision and fixed-reference minimum ratio 0.1000935695.

## Provenance and limits

The exact input candidate and source audits come from P147 commit c55a6ffe6cb4c940948389f89c0dd9deed260667: candidate SHA 211e0e6c422761633ea5e81c5eaf8ee91797a4cf9c5444bf37e4c6bd1f126b38, input 130 audit SHA 5d58c82f70db30bb556e1bbee7a60b90d7b4bf89854463c0f222754bb11dd2a, input 159 report SHA e0d73d26f4c7fb609a87d3d1cfca29851bee6b413c0144b79db0d1f2fdaa585b, and self-pair list SHA 55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de. The package includes hash-named snapshots of the chain driver, topology wrapper, point checker, witness proposal helper, and exact audit sources used by these stages.

This is offline construction evidence only. Native admission is false and accepted physical steps are zero. The 2409 surface crossing count is nonzero; these checks do not establish physical skin/FEM qualification, closed-skin status, anatomy completion, visual inspection, clinical or biological validation. This package excludes the active refined-point branch and mutable agent sources.
