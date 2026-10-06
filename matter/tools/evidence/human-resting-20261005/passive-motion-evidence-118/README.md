# Native bladder and taenia interface checks

The existing NHANAT1 preparation and native scene now consume a localized
bladder repair and a motion-conditioned taenia mesocolica. Human implementations
`d11c0bc` and `7ba50a5` preserve anatomical IDs, source members, attachment
owners, prior repairs, and all unrelated geometry bytes. They add no body mass,
forces, physiological state, or per-step CPU physics.

The first taenia native candidate failed: audit095 found three self-crossing
triangle pairs after motion. Diagnostic099 localizes a 9.297 µm source clearance
that becomes a 0.655 µm crossing. Candidate102 moves one vertex 30.706 µm to a
declared 40 µm source margin; exact audit103 then passes. This numerical margin
is inferred, not measured anatomy. The sampled bidirectional shape limits
remain below 0.5 mm, volume is 6.908535 mL, and minimum source triangle altitude
is 1.766 µm. The original failed native report is included.

The bladder repair removes 0.904 mL from the 78.236 mL source envelope at four
bowel interfaces. Candidate109 is closed and oriented, with volume 77.332 mL
and minimum triangle altitude 2.971 µm. All non-neck source neighbors are clear.
The 13 source bladder/prostate neck triangle pairs retain their exact source
triangles and localized anatomical interpretation; the compiler independently
checks this geometry. Source-binding, derivation, exact predicates and all
intermediate identities remain in the included reports and manifest.

Native run114 completed six seconds on the SSH Mac mini, Apple M4 Pro, using
frozen build026. Native wall time was 88.1374 seconds (0.06808 times real time);
the wrapper including setup took 92.1099 seconds. Exact audit115 covers four
accepted captures (steps 0, 639, 2783, 2999), with eight self checks and 624
passive-neighbor checks. Both organs have zero self-crossings at every capture.
No invalid neighbor remains in these checks. Every nonzero contact is either
the declared taenia/colon-wall interface, localized band convergence/rectal
entry, or the preserved bladder/prostate neck. Native witness barycentric
coordinates map each contact back to source material for localization;
changed triangle pairs are retained, not silently discarded.

Both physiological and surface-observer CSVs are byte-identical to run094.
The unretimed native movie, full accepted packs, current scene payload and
receipts remain at the exact Mini paths and SHA-256 values in `manifest.json`.
This bundle contains source/measurement evidence; original anatomical asset
rights and attribution remain in the bound scene receipts. All execution,
preparation, tests, and audits for this repair ran on the Mini.

These are checks at four retained phases, not proof for every unrecorded state
or full-human acceptance. Respiratory slivers, lung/rib clearance, cardiac
neighbor geometry, and other bowel/liver interfaces remain unresolved. The
five-minute paired demonstration and real-time target remain unmet.
