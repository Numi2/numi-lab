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

Uniform v3 FP64 replay recomputes all 1,728 directional paper rows, including
center and both endpoint projections. At deformation-norm steps 1e-6, 3e-7 and
1e-7, respectively 33, 2 and 1 nominal-direction comparisons exceed the 1e-6
derivative gate. Every projection passes the residual and dynamic-correction
checks, and the recorded branch predicates remain stable. No common declared
step passes every row. A separate realized-endpoint-direction comparison passes
all 1,728 rows at 1e-7; MP90 adjudication of the remaining nominal outlier
isolates direction representation and FP64 stress cancellation. These explanatory
results preserve the original gate and establish no universal noise floor.

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
material failure is reproduced by native382 using production deformation
helpers: all 27 F-high/F-low/Fdot words match the independent point input.
Both private 16- and 32-update descriptors terminate after update14 with local
`ArmijoNoAcceptedTrial`, so the local iteration ceiling is not this failure's
cause. Native388 records all iterations and matches trace-off stress, tangent
and projected-state bits. Its raw residual is 5.2042e-17, while its scaled
correction remains 7.2182e-8, above the unchanged 2e-8 gate. Raw mixed-unit
line-search merit is dominated by a stress row although the frozen-row-scaled
merit is dominated by complementarity. A candidate dimensionless Armijo repair
therefore needs native validation; no complete-coupon claim is justified.

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

Source115 records private deformable narrowphase failures per candidate and
reduces them in ascending slot order, preserving any earlier environment
failure. Ordinary contact keeps its established first-writer behavior. Native381
passes 169 production CCD/routing checks, including an actual uncertified query
through the failure producer, both selectors, and ordered reduction. The query
uses finite synthetic coordinates whose squared relative distances overflow;
it tests status routing and is not a physical contact specimen. Native384
repeats all 169 checks on a freshly built current-main integration with identical
output. Native383 matches all 145 center-capture arrays and all 32 Armijo rows
from native378. Transaction payloads match apart from their explicitly reported
eight-byte device-program fingerprint. Both runs reject residual descent at
Newton9; output parity does not qualify full nonlinear contact.

Native385 captures Newton9, and native386 disables capture. All 32 Armijo rows
and transaction histories match source115 native383. The full 78-row coupled
residual, recomputed from captured FP32 operands, is 8.1766e-8, agreeing with
the Arnoldi estimate within 4.13e-12. It exceeds the relative forcing target
but satisfies the separate 1e-7 outer-target linear exit; nonlinear convergence
remains false. Native389 captures the first trial through residual stages.
That trial fails private vertex-triangle feasibility, so its residual cannot
be used as a complete Armijo measurement. Neither the final tiny trial's
roundoff-limited norm nor the first trial's infeasibility explains all 24
rejections; per-trial feasibility and residual evidence remains necessary.

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

Source115 also rejects ambiguous duplicate crack-band ownership when adjacent
tetrahedra select the same complete shared face for the same damage mode.
Although each element's V/A identity holds, summing both full face budgets
would dissipate twice Gc times the physical area. The mesh builders reject this
case because no one-sided owner is supplied. This exact-coordinate guard and
its interior-cut control do not calibrate a damage law or resolve the source
paper's absolute compression-energy convention. Source115 passes the same
69 build targets and 74 CPU checks; fresh current-main probe, contact-checker
and damage-builder targets compile, and the damage-builder CPU check passes.

An independent 70-digit Robertsson equation benchmark confirms the printed
compression branch gives a negative energy limit at the virgin compression
switch, conflicting with the paper's stated nonnegative-energy premise.
Recentring preserves within-branch mechanical stress/tangent but changes the
damage-driving energy. Six CPU equation checks pass; no author implementation
or erratum establishes which executable convention was used. This remains a
source-model inconsistency, and its paper grade is separate from kraft/NSSC.

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
