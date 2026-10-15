# Cardboard solver repair: residual certification and effective material budgets

The full shipping-box sequence remains unqualified. This milestone repairs a
silently ignored material iteration setting, adds an experimental true-residual
completion check, repairs delayed diagnostic ownership, and corrects convergence
metrics. It preserves the kraft/NSSC/starch recipe, strict solver tolerances,
failed runs and the shipping geometry.

## Reproduced discrepancy

Native368 applies the configured cached and freshly evaluated production
operators to the actual accumulated correction captured in native360. The
fresh action exactly matches the original capture. Cached and fresh actions
differ by 0.3298%; both actual residuals are approximately 2.16e-4 Ns, above
the unchanged 1.1993e-5 Ns forcing target. The Arnoldi estimate of 9.7662e-6 Ns
therefore cannot certify this solve.

The opt-in `NM_FEM_TRUE_RESIDUAL_CERTIFY=1` checks the accumulated correction
against the configured operator before latching linear completion, within the
existing restart and iteration budget. Unsupported scopes reject explicitly.
The unchanged nonlinear production certificate still decides acceptance.
Native376 passes 104 production Metal state-machine assertions; this instrument
is not a complete FEM qualification. Native370 using this option accepts one
of 27 coupon controls, then rejects a material projection at tetrahedron 1820.

An independent FP64 material/frame/element/mass/assembly replay of one captured
resolved direction differs from the native action by 0.7420%, within the 1%
assembled-action gate for that direction. Its strict dynamic correction check
required an additional FP64 Newton update for 1,031 paper rows. Earlier audit
versions that used fixed residual weights for this correction metric are
retained and superseded. No universal near-zero noise floor or complete
smooth-branch derivative gate is claimed.

## Effective material budget repair

The probe requested 32 local Newton updates in native370, but the compiler
silently emitted 16 for the accurate bilinear15 policy. Source109 adds a
separate bounded explicit override, preserves the historical default of 16
and the generic default of 8, validates 1 through 32, and reports the requested
and compiled values. Package round-trip and fingerprint checks cover 32;
compile-only probe controls confirm default16, explicit16 and explicit32
without initializing Metal.

Native379 reconstructs the exact rejected tetrahedron using production FEM
and material-frame helpers, the immutable candidate node buffer and accepted
rollback history. Its private 16-update descriptor reproduces failure with
J=0.9977456927. The 32-update descriptor converges in 18 updates, with final
projection residual 4.7503e-14. The instrument exports raw FP32 words; it does
not export the native correction norm. The 32-update point is a controlled
local recompile, not the complete native370 trajectory. The full 27-control
source109 rerun, native380, confirms compiled budgets of 32 but still accepts
only control0. Control1/Newton0 rejects tetrahedron674. Control0 CSV rows and
both accepted geometry files exactly match native370. The newly exposed
material failure needs its own replay; no complete-coupon claim is justified.

## Diagnostics, contact and convergence reporting

Source106 owns the completion handler's snapshot buffer and copies its actual
metadata at encoding. Native371 restores the RHS-ready flag and matches all
145 captured arrays and transaction states. Strict parity still reports
`FAIL_UNEXPECTED_STATUS`: one rejected private CCD trial reports a different
failing candidate. The status writer uses first-arriving atomic compare-and-
exchange; the captured primitive/candidate records and exact branch determine
the failure attribution. This is a supported nondeterministic selection
explanation, not proof that both candidates failed in both executions.
Erroneous initial branch attributions are retained with their correction.

Source107 extends the edge-edge conservative search bound from 17 to 33 while
keeping its speed bound, separation band and failure policy. Native377 passes
94 direct/helper checks, including the captured near-band ray. That ray
certifies on evaluation 18 inside the existing roundoff-expanded band; it
does not prove crossing the nominal 10 micrometre band. Native378 passes the
prior Newton7 contact rejection on a changed trajectory, then rejects residual
descent at Newton9. It is not a parity run or full contact qualification.

Source108 limits diagnostic factor-rebuild suppression to the selected pending
capture iteration. The earlier broad suppression caused excessive diagnostic
cost; native374 timed out and its attempted SIGKILL reported a permission
error. Its original receipt remains unchanged. A separate hash audit confirms
all 791 bound inputs unchanged and the accepted prefix equal, without claiming
complete history parity or successful process reaping.

Evaluator110 integrates absolute piecewise-linear moment work, splitting at
zero crossings, and compares signed measured springback. This prevents force
cancellation and opposite-direction springback from passing. Schema v2 records
the corrected definitions. Thirteen resolution and nineteen convergence-reader
tests pass; these are synthetic evaluator checks. Source109 and predecessors
pass 69 build targets and 74 CPU checks. Neither count is physical evidence.

## Remaining qualification gates

| Verdict | Current evidence and remaining work |
|---|---|
| Solver correctness | Actual residual discrepancy and ignored budget reproduced; local budget repair verified. Full nonlinear and contact histories remain open. |
| Numerical convergence | No completed coupon spatial, timestep or slower-loading series. Accepted prefixes do not qualify. |
| Component calibration | Matched progressive corrugated damage and local starch/hot-melt fracture parameters remain unidentified. Robertsson absolute-energy convention remains unresolved; its grade coefficients are separate. |
| Analogue validation | Held-out component, scored-bend and closure validation remain open. Published nearby grades retain transfer uncertainty. |
| Physical assembly in simulation | No coupon-derived frozen score allowance, completed twelve-score process or fixture-free shipping checkpoint. |
| Crush/impact prediction | No numerically qualified first compression peak, frozen scenario masses, gravity crushing or drop-impact result. |

The explicit-flute shipping preflight counts 528,444 nodes, 1,477,728 tetrahedra
and 260,835 cohesive faces. A compact modeled allocation including tangent
cache is 14,281,570,410 bytes; this excludes arena costs and is not a measured
peak. Current Newton, graph and rigid-contact limits reject this geometry.
No limits were raised to imply shipping admission.

Commands, source/build fingerprints, full histories, failed audits, timeout
receipts and lossless restoration packages are indexed under
`matter/examples/cardboard/evidence/native-implementation-20261014/summary.json`.
The implementation cannot yet produce a qualified assembled-and-crushed box.
