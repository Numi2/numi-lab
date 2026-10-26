# Fixed bone outward-plane rows 458

This isolated experiment builds three displacement lower bounds for each
current skin/bone crossed pair using the actual fixed bank triangle's oriented
world-space vertices. The caller verifies implicated bank winding by exact
oriented edge incidence and positive signed volume. The plane point is the
first bone vertex.

For each skin corner x_i, the bound is
  n dot d_i >= epsilon - n dot (x_i - q0)
where d_i is the local-fit displacement. epsilon is computed from one full
float32 ULP for each coordinate of q0 and each skin corner, projected by
abs(n), taking the maximum corner budget. No arbitrary world-unit margin is
used. Meeting all three bounds puts the complete skin triangle strictly on the
bone triangle's outward side of its plane, so this pair is separated. The
helper returns all three inequalities, including negative bounds. It also
returns positive-deficit rows as normalized LP slack targets; nonpositive
baseline-clear corner rows are added as hard nonregression constraints so they
cannot be moved back inside the plane.

The opt-in runner validates the frozen 408 objective first and preserves its
point row block unchanged. It requires the matched 456/457 candidate, complete
159 report, 335 face targets, 390 point targets, and exact 039 bank export.
It does not replace self, area, depth, fixed guard, edge gradient, source-cap,
or trust constraints. A successful fit remains a construction proposal and
requires complete exact post-fit 130/159 audits.

Run helper tests:
`/Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v test_fixed_bone_outward_plane_458.py`

Run the 408-compatible fit wrapper with the same arguments as
`run_grouped_fit_435.py`; set optional input overrides in
`NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458` and
`NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458`.

