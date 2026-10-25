# Shared skin/fat interface construction increment 083

The existing layer compiler now provides explicit shared-vertex scalar-interface snapping. All incident crossing edges must fit the caller's declared geometric perturbation budget. Original coordinates and fields remain unchanged; the output contains the changed scalar field and complete edge/vertex ledger. It rejects invalid indices, nonfinite budgets, invalid emitted Float32 input cells and ambiguous all-zero parents. No physical positivity or volume-partition tolerance changes.

The new tests cover shared-edge decisions, all-incident-edge bounds, deterministic visit order, input immutability, exact-zero reuse, two-side positive-cell partition, invalid inputs and zero budgets. The complete compiler suite and continuous triangle-translation proposal tests pass: 54 tests.

A replay on the actual 129,526-point / 461,669-tetrahedron construction input reproduces the frozen trial fields and every ledger array exactly. The trial snaps 18 shared vertices with maximum edge-interface movement 2.4323 micrometres at the explicitly declared 0.1% local skin-depth budget. This is an asset-construction perturbation, not a runtime force or a Float32 tolerance waiver.

That full material clipping trial still FAILED: one required emitted Float32 child is nonpositive. The exact input, original/snapped fields, source pins and failed partition witness are retained. There is no accepted material volume or physical mass mapping.

The complete 185-bank face audit on the separately frozen zero-self/zero-inside-node candidate finds 1,868 crossing triangle pairs across 388 faces and 33 banks. Continuous translation proposals include all 185 swept AABB coverage. Of 388 crossed faces, 295 have an initial overlap exit before 6 mm; 93 are cap-limited and 21 have later re-entry components. These are proposals, not anatomical clearance.

All preparation and tests ran on SSH Numi. No candidate was installed and zero physical steps were advanced. Material/contact/mass/physiology/accepted-state owners are unchanged. Connected face clearance, positive well-conditioned material cells, conservative physical ancestry, matching detailed/mechanical geometry and native qualification remain required.
