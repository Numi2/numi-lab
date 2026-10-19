# Aortic wall outer-lateral contact selector evidence

This evidence bundle covers an opt-in per-face selector in the registered aortic-wall owner. The default remains contact-off. When enabled, it retains 216 outer-ring query nodes and the complete immutable surface map while selecting 384 outer-lateral wall faces and excluding 672 nonouter wall faces; it does not enable wall self-contact. Mixed MPM/FEM use remains rejected by the existing owner.

## Evidence

- `cpu/` reruns the CPU probe against the registered wall source. It reports 532 nodes, 1,584 tetrahedra, 0.00705926 kg, 1,060 canonical surface faces unchanged, 384 selected outer faces, 672 excluded wall faces, and four preserved nonwall map rows. The receipt pins probe source and binary.
- `build/` retains runtime and app compile/link argv, receipts, and the app dependency file. The app binary was linked to the candidate library. Its source base was `a778267aee1af8f78e7f778f62a1526eb92b1777`; it was not rebuilt against the later sparse checkout head `cbd4fac19c7a038af1e15a5deec3dec381f892a7`.
- `native/` retains the one-step body invocation/result, compressed logs, and CSV. The selector initialized and the frame was accepted. The frame reported one existing support contact; no wall-specific impulse was observed.
- `gpu/` retains a direct Metal selector fixture. It dispatches the production compact-surface primitive builder and production `surfacePairEligible` on a synthetic two-object/two-tetrahedron fixture. Default unmasked outer and cap pairs are eligible in both argument orders. Masking the cap emits the disable sentinel and rejects both cap-pair orders while the outer pair remains eligible. This establishes GPU selector semantics only; it does not measure wall force or qualify anatomical contact.

The exact tested MSL, app/library binaries, invocations, dependency list, and source hashes are pinned in the manifest and receipts. The parent must validate the six-source patch against `cbd4fac19c7a038af1e15a5deec3dec381f892a7` before publication.
