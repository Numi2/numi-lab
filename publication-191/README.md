# Numi Human P191: exact contour precision and rejected wide fit

Construction diagnostics on SSH Numi continuing P190 commit 210abbd79debf9ce1543fa5bb1859b7d692f5c5d. The best complete static audit remains 653 comparable parent skin/bone intersection identities, 125 sampled inside node/bank pairs, zero skin self-intersections, and 45 skin boundary edges. No scene or physical admission is implied.

For 65/99, 148 rational crossing segments form one closed degree-two contour. Two distinct exact endpoints are 13.7649 nm apart but round to the same emitted Float32 position. The face-wise construction stopped on a zero-area reference triangle, emitted no candidate, and retained the exact witness. Valid nonempty source/detail/subface CSR ancestry exists for all 241 parent faces; no distinct endpoints were silently merged. Exact point-on-input-surface checks on rounded endpoints are reported as a construction limitation, not as proof the source is defective.

The wider 115/118 topology includes the previously intersected neighbor face and preserves a connected 47-edge boundary. One shape-constrained 3D fit was rejected: despite optimizer success and all 110 output vertices classifying outside the bone, exact local checks found 105 bone pairs and 15 patch-internal self pairs, and five emitted Float32 faces inverted. Patch-versus-complement self pairs were zero. The private topology and failed fit are retained, not promoted.

No full-bank, whole-body self, volume, tissue-transfer, mass, motion or native qualification was performed on these prototypes. Source and mixed-reference attribution remain in P182/P183.
