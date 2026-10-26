# Anchor-compatible finite-triangle clearance
Offline connected mesh construction only. Native admission is false; accepted physical steps remain zero. This does not finish skin/fat mechanics or the zero-clearance goal.

The checked single candidate is c1c0b63ee448703ef3240dbcc251e650022d7f76f475bc75db9cae040d1e3a6c:
- Surface crossings: 3,090 -> 3,079 across all 185 typed banks; 33 resolved and 22 introduced.
- Contained node/bank pairs: 283 -> 278; five resolved and none introduced.
- Skin self-intersections: zero.
- Independent 494 verification: 237 anchors and 169 non-coordinate arrays unchanged; 154,796 self-seed rows and 15,102 fixed-guard rows have zero violations.
- Maximum source displacement 11.98284 mm, maximum step 0.250013 mm; all 41,733 faces pass existing Float32 area gates.

506 stopped because outward infinite bone-face halfspaces demanded motion of two fixed vertices. Exact queries placed those vertices outside all 185 banks; their shared edge has no intersection with nearby emitted bone triangles. 508 retained the outward-hemisphere restriction and failed closed. 513 instead admits either sign of finite-triangle SAT axes for anchored pairs while preserving anchor compatibility. Unanchored pairs retain the existing 466 outward policy. All three corner rows remain, and the existing full geometry checks select the candidate. This changes an offline construction heuristic, not physical forces or runtime acceptance.

Tests: 508 five tests; 513 six tests including opposite-sign support separation and unanchored row parity; 510 two tests and a real matched plan. Root full-input preflight covers all 3,090 pairs and all 237 anchors, yielding 9,270 corner rows and zero positive anchored bounds.

The 511 fit retains its successful first SCP result after SCP2 reported HiGHS status 4 in component 7. It does not claim full solver convergence. Full-mesh area backtracking still limits the SCP result to alpha 0.25; alpha 0.5 fails only fixed-reference face 40714. The checked improvement is modest. Failed preparation and preflight evidence is retained.

510 reuses the tested source-ancestry adapter for refreshed targets; 515 refreshed the new selected state successfully. Sources, input pins, timings, launch commands and compact receipts are retained. The 45 existing open head edges, physical skin/fat cell positivity and transfer, regional mass/thickness, retained motion poses and native 16/60-second qualification remain unresolved or unqualified. No images or native viewer checks were performed.
