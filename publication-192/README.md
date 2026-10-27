# Numi Human P192: contour precision and fit orientation diagnostics

Construction diagnostics on SSH Numi continuing P191 commit 3b42c2ceeb58e3563f865504cb69fd45c53fc06b. The best complete static audit remains 653 comparable parent skin/bone intersection identities, 125 sampled inside node/bank pairs, zero skin self-intersections, and 45 skin boundary edges. No scene or physical admission is implied.

For bank 43/16, the 502-face two-ring patch retains a simple 38-edge fixed boundary and valid source/detail/subface CSR ancestry. Its 249 exact segment pairs form three degree-two cycles of 8, 108 and 133 edges. Two distinct exact endpoints are 8.2862 nm apart but round to the same world Float32 coordinate. The face-wise CDT stopped on a zero-area reference simplex; no patch candidate was emitted. Rounded points failing exact point-on-input-surface checks are reported as a construction limitation, not as proof the source is defective.

The 115/118 rejected constrained fit was independently rechecked. Its face and parent arrays match; the five inversions are not an indexing mismatch. In double precision those faces sat at the optimizer's near-zero 1e-6 signed-area floor, with residuals around 1e-13, and Float32 rounding then changed their signs. SLSQP's success tolerance is not a hard constraint-slack guarantee. This does not alter the earlier exact bone/self failures.

No full-bank, whole-body self, volume, tissue-transfer, mass, motion or native qualification was performed on these prototypes. Source and mixed-reference attribution remain in P182/P183.
