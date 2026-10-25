# Necessary connected-topology feasibility
This is an offline, fixed-surface diagnostic, not a runtime solver or geometry admission.

Run 009 failed an invalid construction assumption: a prism's one inner apex need not have its corresponding outer-column node on that particular outer face after retessellation. The failure is retained.

Run 010 uses the actual affine determinant planes instead. It checks all one-inner/three-fixed-outer cells for the 36 inner nodes with at least one nonpositive such cell. Thirty-five stars permit positive altitude within a 20 mm coordinate box. Node I3656/O1245 does not.

The second check removes all coordinate bounds. Its LP optimum is zero. The nonnegative dual weights sum to 1; their normal sum has norm 1.66e-16 and their weighted altitude upper bound is 1.90e-19 m. All 15 fixed outer faces contain O1245. Therefore the current fixed outer face star and shared inner apex cannot produce strictly positive adjoining cells to floating-point diagnostic precision. This is a necessary obstruction, not a proof that all other cells, material boundaries or contact are valid.

The corrective construction must replace the connected coarse surface/volume topology at this star, or revise the target shape using anatomical source constraints. Increasing an optimizer budget cannot remove this obstruction. No negative cell or fitted surface has been installed in the native scene. There were zero accepted physical steps.
