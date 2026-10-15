# Same-generation center residual and J-delta capture overlay

This is an external, source-bound diagnostic overlay on `source-contact-capture-and-resolved-material-sweep-095`. It adds a bounded full-space Newton center capture of `b`, the completed Newton correction, and the same-generation production `J * delta`, together with row mapping and the contact-history buffers actually selected by the Newton center. The resulting linear residual is `b - J * delta`; the physical center residual has the opposite sign of `b` in the existing runtime convention.

The direct action dispatch reuses the production matrix-free kernels and accepted linearization. It runs after the last FGMRES restart and before Armijo, using the already selected candidate state, tangent cache generation, contact samples, and the same history selection. It is not an alternate solver or acceptance path. For a completed FGMRES state, a diagnostic scalar bypasses only the kernel's early-return guard. Every existing production call binds that scalar as zero; only the extra selected diagnostic action binds one. The direct action writes preallocated true-residual scratch/status buffers. The native on/off history comparator must still establish full physical-state and rejected-transaction byte parity before the capture is trusted.

The instrument accepts only its explicit bounded block layout. It excludes MPM, Human support/equality/limits/attachments, vascular mechanical loads, and articulated generalized rigid coordinates. FEM rows map `xyz` to mechanical velocity and `w` to pressure; field rows use `xyzw`; a free rigid body contributes six scalar rows. The capture is raw diagnostic state, never an accepted checkpoint. Material and cohesive accepted history is held fixed through Newton/trial evaluation. Contact history is a lagged algorithmic impulse/frame estimate; the history is not integrated physical tangential slip.

## Validation

- `python3 verify_overlay.py` verifies every one of the 774 frozen095 source hashes, the three base copies, that the overlay contains no symlinks, and the updated kernel argument bindings.
- Objective-C++ `runtime.mm` syntax check: pass, using the frozen095 compile database command with the source path redirected to `overlay/matter/src/runtime.mm`.
- Metal compile-only: pass with `xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math` and the frozen095 include roots. The compiler emits two existing warnings in `fgmres.metalinc` about threadgroup `sharedValid` and `refinementValid` initialization.
- GPU execution and native on/off parity: not run by this overlay owner; no physical or solver qualification is claimed.

Apply `overlay.patch` at the frozen095 source root. The incident record documents a transient write-through into one frozen file, its exact two-line diff, restoration hash, and the available timestamp window. All overlay files are now regular files, so later edits cannot write through to the frozen snapshot.
