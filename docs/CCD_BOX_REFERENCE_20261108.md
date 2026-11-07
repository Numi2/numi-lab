# Independent four-corner box reference

The rejected frictionless box in CCD_GEOMETRY_V2 has mass 1 kg, inertia diagonal 0.1 kg m², normal initial velocity -1 m/s, zero target, zero regularization and identity pre-step pose. Its four point anchors give the FP64 normal response `W_ij = 1 + 10*(x_i*x_j + z_i*z_j)`. Applying this independent matrix to the retained candidate impulses (0.91451377, 0, 0, 0.09063549) N s yields row velocities (0.01029849, 0.00370747, 0.00659104, 0.000000014) m/s, reproducing the failing certificate. Equal 0.5 N s impulses at opposite corners produce zero normal velocity at all four rows, total impulse 1 N s and zero torque. The impulse distribution need not be unique; this is one valid response, not a claim that it is the only solution or that all box geometries are qualified.

The single-coordinate algorithm slowly shifts impulse between nearly parallel rows; the failure is not due to an infeasible contact. A bounded constrained pair solve is being tested separately with the same sweep and certificate. Singular, frictional, anisotropic, finite-state and broad articulated behavior require independent qualification.

Evidence `evidence/ccd-box-reference-v1-20261107.tar.gz`, SHA-256 `7db9f1270fa57dc976adb155b55b90b4c52fef526e2243b3289bbbbd13ba5d59`, includes the pure-Python FP64 reconstruction, its output and the source result hash.
