# Candidate skin registration owner

The existing whole-body skin initializer now accepts an explicit complete candidate registration bound to the actual source payload SHA-256. It rejects wrong hashes, partial arrays, nonfinite frames, mismatched donor identities and reference coordinates inconsistent with emitted Float32 frames. Empty candidate metadata retains the legacy owner.

Projected rows retain the frozen source-face-to-reference registration residual in both world and donor-local coordinates. A projected first row derives common placement from the source-face point before adding that residual, preventing a doubled translation. A rotated first-row regression checks that the residual is applied exactly once.

The C++ regression builds with warnings as errors against clean main dependencies; its registered CTest target passes. These tests cover registration ownership and transforms, not anatomy or physical qualification. The private D24 initialization compiled and then correctly rejected a candidate whose five stored normals were stale after weight changes. That candidate remains unadmitted; no default asset, simulator, material law or physical acceptance tolerance changes in this increment.
