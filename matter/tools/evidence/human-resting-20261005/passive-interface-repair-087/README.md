# Passive pancreas and spleen repair in the integrated resting scene

Human revision `3d706b8` adds a source-bound compiler for two passive anatomical
surfaces in the existing NHANAT1 asset. It preserves every other surface's local
geometry and identity, the physical mass ledger, and the existing physiological
state. Five compiler admission tests passed on the SSH Mac mini. All geometry
preparation, numerical work, tests, simulation and audits in this evidence ran on
that Mac; no simulation ran on the Air.

The registered reference had small overlapping volumes between the pancreas,
spleen, stomach, left kidney and duodenum. A localized inferred interface repair
preserves the stomach, kidney and duodenum, with a declared 50 micrometre numerical
separation. The final geometric volumes are 63.912492 mL for the pancreas and
114.710524 mL for the spleen, reductions of approximately 2.20368 and 0.56475 mL.
These are derived surface volumes, not measured person-specific organ volumes
or changes to physical body mass. BodyParts3D source identities FJ1895 and FJ2561
and their provenance remain in the existing receipt.

At 25/50/100 micrometre separation, the unconditioned pancreas volume reductions
were 2.17359/2.20368/2.26464 mL; spleen reductions were
0.55407/0.56475/0.58645 mL. This is a geometry sensitivity check, not physiological
validation. The 25 and 100 micrometre candidates were not admitted to the native
scene. The retained 50 micrometre candidate was conditioned to a minimum triangle
altitude of 1.2 micrometres, with closed oriented surfaces and exact self checks.
The four intermediate diagonal flips reproduce bit-for-bit from the retained
helper and input geometry.

The exact source audit checks both surfaces against all 79 declared passive
structures (155 unique pairs). All valid pairs have zero crossings. The
pancreas–taenia-mesocolica comparison remains unresolved because the original
taenia source contains a degenerate face; it was neither omitted nor waived.

The combined cardiac/passive scene completed 3,000 accepted 2 ms steps in
95.75684 wall seconds (six simulated seconds; real-time factor 0.06266). Exact
pancreas/spleen self and passive-neighbor checks on accepted captures at steps
0, 63, 511, 639, 1951, 2207, 2783 and 2999 have the same result: zero self
crossings, zero valid neighbor crossings, and the same explicitly unresolved
taenia comparison. The physiology and geometry traces are from that one native
scene. Cardiac and lung clearance are separate checks; this evidence does not
establish complete anatomy, five-minute endurance, intervention recovery or
real-time performance. Other CPU geometry work ran concurrently.

The six-second run used the earlier receipt073 with the same anatomy payload
bytes as compiler output074, SHA-256
`82248c7d2397a19af382c21646da6c827b901880a08ae8242c80dab8b4329b0e`.
The cardiac owner attached its source-bound map correction after passive asset
preparation. The invocation records the exact binary input assets and source
file hashes; the frozen source directory's Git HEAD alone does not identify the
combined build. The native viewer recording was not retimed and remains on the
Mini with its exact path and hash. The Mini had no logged-in graphical desktop,
so interactive window operation remains unverified.

`manifest.json` binds retained compact evidence and the full inputs, movie and
accepted geometry packs kept on the Mini. Large JSON files are losslessly
gzip-compressed, with original and retained hashes. Its SHA-256 is
`3354a4912b137e57fc88f5e9f46d4b7d1128eb6ad7ef8ae7e2afb5d364cd5c3f`.
Source anatomy remains subject to the source licenses and attribution in the
receipt; see [BodyParts3D licensing](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html).
