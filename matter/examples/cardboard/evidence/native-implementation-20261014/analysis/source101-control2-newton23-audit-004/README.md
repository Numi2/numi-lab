# Source101 control-2/Newton-23 audit (wrapper revision 4)

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


Revision 4 fixes the material-health inventory to read the capture's actual
tangent shape (`cached_algorithmic_tangent_local` is a validity dictionary
containing an 81-float matrix), rather than misreporting every tangent as
missing. Revisions 1–3 and their outputs remain preserved; this correction
changes only the external audit wrapper, not the native run or operator report.


## Native 360 result (wrapper revision 4)

The source101 run completed with exit 2 (not timed out) after about 1498 s. It
matched baseline 352's first two accepted bent/held geometry checkpoints and
accepted material-history snapshot, then rejected control step 2. At selected
control 2/Newton 23, all 1,920 environment-0 tets had valid projection status
and finite valid cached tangents. The Arnoldi recurrence estimate dropped
below the unchanged linear forcing target at restart cycle 6, after 111 total
columns, and latched the completed tail. The independently assembled native
true residual remained 17.99 times the target; true-residual replacement was
explicitly off. The outer line search then first failed at Newton 23 after 24
backtracks, with no accepted trial; later trace rows carry the latched failure.

The same-center production action versus unsymmetrized FP64 reconstruction of
`M + dt^2 K` differed by 0.807% relative (6.323e-6 Ns absolute); mass and
stiffness action norms were each about 0.01951 Ns and nearly canceled. This is
a measured discrepancy, without a native noise-floor result. The selected
RAP snapshot was skipped because its estimated 22,890,320 bytes plus the
17,818,224-byte other-readback upper bound exceeded the unchanged 32 MiB cap;
the Newton-trace recorder also declined the 64-iteration request because its
per-run cap is 32. See `findings-360.json` and `fgmres-cycle-360.csv`.
