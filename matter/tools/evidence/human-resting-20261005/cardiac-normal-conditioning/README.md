# Scale-aware cardiac surface display normals

The source compiler previously rejected a well-conditioned Float32 atrial partition vertex solely because its area-weighted normal vector had length below 1e-12 square metres. The changed guard checks relative cancellation using incident area and valence in Float64. It still rejects zero-area faces, nonfinite values, unused vertices, and cancelled incident normals.

All 16 existing/refocused tests pass on the SSH Mac mini. The retained report binds the exact owner source and RA020/LA007 candidate hashes. All four wall/lumen normal arrays retain the same area-weighted directions bit for bit; the RA wall and lumen each had one vertex rejected by the former absolute cutoff.

This is a display-normal compiler repair. Geometry, exact intersection admission, cardiac state, and runtime physics are unchanged. The atrial candidates still require material-neighbor resolution and native cycle qualification.
