# Source101 control-2/Newton-23 audit (wrapper revision 2)

This is an offline, read-only audit harness for native capture 360. It is bound
at run time to source snapshot `source-boundary-reset-order-101`, the captured
prediction, terminal receipt, stderr, and rich `operator-capture`. It rejects
missing captures and selectors other than control step 2 / Newton iteration
23. Before analysis it verifies that 360 used the same source101, input hashes,
compiled pack, mesh, initial object, physical argv, and the first two accepted
bent/held OBJ checkpoints and final accepted material-history snapshot as
baseline 352 (allowing only the output directory and added diagnostic
selectors to differ).

Revision 2 fixes a wrapper-only `baseline_manifest` NameError in revision 1.
The manifest remains hash-bound as one of the baseline files. Revision 1 is
preserved in `../source101-control2-newton23-audit-001/` with its original
package hashes; no native evidence is rewritten.

It runs the source101 cached-operator auditor separately for environments 0
and 1, retains a per-tet material/tangent status inventory, and writes reports
plus a hash-bound summary under a new output directory. It never edits the raw
native run.

Run only after the native runner has written `receipt.json` and completed the
capture:

```sh
python3 run_operator_audit.py \
  /Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360 \
  --output /Users/home/cardboard-implementation-20261014/analysis/source101-control2-newton23-audit-002/result-360
```

The root label emitted for `NM_FEM_RESIDUAL_ARMIJO_ROOT=all` is a control-step
selector. The early `root=0` line is expected for control step 0 and is not the
`NM_FEM_OPERATOR_FD_SELECTOR=2:23` capture. The wrapper requires the selected
operator-FD and true-residual completion records for 2:23 before analysis.

The exact first-layer triage after the capture is: (1) confirm Newton-23
candidate, correction, deformation high/low words, rate, accepted material
state, and current restart generation all belong to the same request; (2)
check per-tet production projection validity/status and cached tangent validity;
(3) use the source-bound matrix reconstruction to compare native versus
assembled A action, mass and stiffness terms; (4) compare the selected true
residual/FGMRES diagnostics and current RAP generation. Keep any local
material/projection mismatch distinct from cached derivative and assembled
operator mismatch. No FD after terminal failure is treated as a valid
production baseline, and no offline reconstruction converts the rejected
physical step into a pass.

The native capture may not contain enough information to independently
recompute the nonlinear material projection. If material replay needs additional
state (for example, exact accepted history or projection trace), report that as
missing rather than substituting captures from 304, 323, or 352.
