# Native 026 vascular row 23 audit

Read-only post-run analysis of native-vascular-row23-026. No build, native run, GPU work, physics changes, or tolerance changes were performed.

The exact failure is at control root 20, microtick 3, Newton iteration 1, line_search: NM_STATUS_NONLINEAR_SOLVER_FAILURE, row 23, attempted alpha 0.00048828125, current merit 3.7328690138949128e-10, trial merit 3.7340436298549662e-10, worst certificate ratio 1.4490128755569458. The bounded trace has 49 records for this root/tick and identifies flow_02 at row 23: stable connection 24 from stable compartment 2 (compartment_01_brachiocephalic_arteries) to stable compartment 3 (compartment_02_upper_body_arteries). Source JSON from/to are stable IDs, which the native compiler maps to compact compartment indices.

The trace contains 46 state rows although the source graph has 45 unknowns. The extra row is expected: NumiHumanRestingAorticWall.hpp attaches a deforming cavity to stable compartment 7, the abdominal aorta, replacing its linear pressure law and adding one pressure unknown. Its configured pressure scale is 13,300 Pa. This audit uses the extra row's physical pressure for that compartment; treating its source-only linear pressure law as active creates false large residuals in flows 07, 08, 11, and 14. The cavity geometry equation at extra row 45 is not reconstructible from this trace because its FEM geometry state/direction is not included. Its squared-residual contribution is reported as the exact difference between native merit and the sum over the 45 reconstructible volume/flow equations, without assigning a sign to its residual.

At the failing iterate, the reconstructed physical row equation is R Q - (Pfrom - Pto) = -0.1439870608 Pa, or F23 = -1.439870608e-5 after the authored 10,000 Pa row scale. The continuous analytic correction derivative is Jp23 = +6.279161366e-7 per alpha; thus row 23 alone has a descending continuous merit slope 2 F23 Jp23 = -1.808235979e-11 per alpha.

The full correction is not a descent direction. Across the 45 reconstructible source equations, the base merit is 3.574036040e-10 and its analytic directional derivative is +2.005397620e-10 per alpha. The omitted cavity-geometry row raises the native total from the known-row sum to 3.732869014e-10; its inferred squared-residual contribution increases by 1.607086062e-14 at the tested alpha. The native total merit consequently rises by 1.174615136e-13.

Float32 trial construction adds a second effect at this tiny alpha: volume normalized states 1 and 2 remain bitwise unchanged, while the row-23 flow state moves by -5.960464478e-8 (two representable steps). The pressure compensation in the continuous derivative is therefore lost in the actual trial; F23 becomes -1.439981861e-5 and row-23 merit rises by 3.203918815e-14. The residual arena at line_search contains the Krylov recurrence residual, not the physical equation residual: its row-23 value is 2.517211578e-5, so it must not be substituted for F23.

FGMRES diagnostics at the failing iterate are current residual 6.592377904e-4, initial residual 9.362745332e-4, convergence flag 0, and 16 Arnoldi columns in the restart; the runtime failure record reports 32 Krylov iterations. The residual is still 70.4% of its initial value, and the correction has positive full-system merit derivative. The next bounded numerical experiment should increase the supported iteration cap to 64 while preserving the same residual tolerances, geometry, and physical state, then repeat the same trace and check that the correction is a descent direction. This report does not claim that a larger cap will pass; it is the narrow experiment justified by the trace.

## Audit implementation and provenance

analyze.py decodes the native bounded trace and reconstructs the 21 volume and 24 flow equations, their directional derivatives, and the float32 line-search trial. analysis.json contains every selected record and the machine-readable receipt.

- Native stderr SHA-256: 652b639c8b81bd130680b558a2a30e878afae001e3210e5b2187f01c94f7b2f2
- Invocation SHA-256: 7fb72fd64105afeac3c760c4289fc28fac4c0a39059b82e30f34e6120e4337af
- Native provenance SHA-256: 326b9901a8ca828ce9a7178aa3ca637417ffb2b04cde19da821e2f637cd482d4
- Executable SHA-256: 83bf98bee369a57a9dd19e74a6aea343cc767486e189d0ff6a2aa6c8982e5b2e
- Metallib SHA-256: 582a030a67312ee00f3358e4a48e8c7741e7d2b67584fe5d496f58f578532535
- vascular.metalinc SHA-256: aad796d22579c7c3a5cff56055960ddad759ed3dc70e29a393d4b06c5d66632e
- NumiHumanRestingAorticWall.hpp SHA-256: bbc45f7d33ff5ad9d088cc3c44e913029c83a025e92ea90478b73ad19e37def0
- Vascular graph SHA-256: 43ff6b49daf1d42cf9e88d85d8577f70e112dfcc3756ec43954e544a3e4bc0dc

The run's reproduction entrypoint was matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/vascular-row23-026/template.json; its invocation selected NM_VASCULAR_TRACE_ROOT=20. The run completed with 20 accepted rows and the failure described above.
