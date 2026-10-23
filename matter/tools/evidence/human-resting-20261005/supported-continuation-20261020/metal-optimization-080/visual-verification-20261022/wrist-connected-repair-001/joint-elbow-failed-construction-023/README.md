# Rejected elbow construction trials, SSH Numi, 2026-10-23

No candidate in this bundle is admitted or used in the native scene. These
nonvisual tests concern the 069i detailed skin, matched shoulder coordinates
0.1 rad, neutral elbows versus 0.8 rad flexion. They do not measure FEM dynamics.
The unchanged baseline has zero tested neutral bone/self/outside intersections;
flexed skin/bone triangle pairs number 139 left and 128 right.

Trial 036 generated 3,498 linear constraints on 451 local source vertices
(1,353 variables) and failed to converge in its 30,000 ADMM iterations. This
failure alone establishes no geometric infeasibility. Coordinate-box and L1
feasibility diagnostics are different bounds, not substitutes for L2 balls.
A minimax L2 outer approximation for the saved cut bank subsequently found
rho lower bound 1.0057162084 > 1 (subject to LP tolerances). It concerns those
linear constraints and tapered 5 mm construction caps, not the nonlinear
physical feasibility of elbow motion.

Trial 037 raised only the mesh-edit construction cap from 5.0 to 5.1 mm, with
the same taper. Collision, material, mass, numerical and orientation acceptance
gates were unchanged. The exact 037c trial moved at most 5.0418843 mm and left
vertices outside its 451-vertex support bitwise unchanged. It FAILED: active
bone pairs 148 left / 101 right; skin self-pairs 6 left / 0 right; neutral
self-pairs 3 left / 0 right; and 8 active / 11 neutral face reversals. A second
linearized solve exceeded the new displacement cap and was stopped before
exact testing. No passing mesh was emitted. All recovered/new LP vectors and
exact trial witnesses in this bundle remain failed evidence.

This stops the repeated rest-contour method. The next method must address
physical FEM motion, volumetric transfer and anatomical contact or use a
justified local remesh, rather than hide crossings with locks or tolerance
changes. The actual finite-spring force owner and separate native runtime
coupon are published in 38660e86 and ce490652; those passing software tests do
not qualify this failed mesh. Joint-wide supported-motion qualification,
independent toe/IP articulation and full-body skin admission remain unfinished.

Files are gzip-compressed losslessly. manifest.json binds each to its original
path, byte count and SHA-256, plus the retained compressed hash. Trial scripts
retain their original dependencies and exact input hashes; this is evidence
retention, not a standalone replacement simulator or new artifact format.
