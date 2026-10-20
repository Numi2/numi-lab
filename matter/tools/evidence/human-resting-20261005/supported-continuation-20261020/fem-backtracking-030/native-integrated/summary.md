# Native FEM-backtracking run 030

The invocation requested 32 physical steps of 125 microseconds (4.0 ms) with FGMRES iteration budget 64. It exited 1 after 23 accepted CSV rows (2.875 ms) in 501.485 seconds. The remaining nine requested steps did not complete.

This run advanced beyond the prior whole-body FEM quality-gate failure at root 22. It instead failed at root 23 after three microsteps in the vascular line search (matter code 10, passive_tissue_runtime owner, object UINT32_MAX, failing index 23), with diagnostics [0.00048828125, 1.7720612754956733e-09, 1.7721559775196738e-09, 1.8279266357421875]. This is the new frontier, not a passing horizon.

All 23 CSV rows report zero normal impulse and zero root assistance; maximum peak penetration is zero and the minimum reported contact gap is recorded in capture-facts.json. Therefore the run does not demonstrate sustained bed support. Initialization is explicitly transient and unbalanced. Mass calibration executed at 97.1319506911 kg against the declared reference; this is a model budget, not a measured person.

At admission, the run had 28 registered ligament force rows, 26 fascia regions, zero legacy fascia reduced-ligament rows, and a 20-sample elastic attachment binding with zero hard ties and zero added mass. The skin transfer diagnostic reported 54,551 active vertices and 7,399 unreferenced payload vertices, with a 42.585 mm maximum projection witness. These are construction/runtime diagnostics, not whole-body supported-motion qualification.

The exact run source revision is 67c85cb3098ee6a7261e6dde4a5112d4afad75eb; the repository HEAD at packaging is recorded separately as bcdbbc2b3c57269985061fca4c8b35b5eeb65f8a. Binary and Metal library hashes are bound to the completed run's provenance. Thirteen argv asset files were rehashed and matched the retained native-029 input manifest. The 283 MB render pack is intentionally excluded.
