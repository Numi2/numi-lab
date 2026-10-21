# Solver integration: builds 181–187

All execution is on SSH Numi, Apple M6. No visual verification. The minimum
0.1x end-to-end requirement remains unmet. None of the timing results below
establish sustained supported rest, physiological qualification, or presentation
throughput.

## Changes and outcomes

| Change | Verification and Human result |
| --- | --- |
| Source-bound larger optic local factors | Keeps full mechanical coordinates, existing 32-node factors and skin-plus-optic coarse basis. Adds 67 connected 128-node optic patches; 141,792,536 device bytes in the explicit bounded 192 MiB arena. CPU layout/ownership/coverage checks pass. Native trial fails vascular Armijo at row 16 before accepting a root, including with persistent vascular low words. Not promoted. |
| Completed exact-zero operator action | GPU tiles establish exact zero; healthy completed Newton state may suppress FEM/cavity operator grids. Current-RHS residual measurement and output clearing remain mandatory. 251 focused GPU checks pass with Metal validation. Three three-root matched pairs preserve all 17 endpoint groups exactly; median post-first mechanics throughput ratio 1.01309. Only 250 microseconds are measured after the first root. |
| Full-coordinate multiplicative global coarse correction | Uses cached current A*P, residual deflation, fine solves of the deflated residual and the existing coarse addition. 781 full-vector coarse and 1,537 hybrid-RAP GPU checks pass; 69,749 generic FGMRES checks pass. Human body cost increases 8.1815 to 12.5611 seconds; acceleration comparison misses 1% (2.89084%). Remains experimental. |
| Source-derived cavity load coarse direction | Registered closed cavity boundary, oriented closure, ownership, finite geometry, free-node support and source IDs are validated. CPU fixtures cover translation/dilation, mass/constraint masks, closure roles and invalid geometry. The first 185 attempt rejected valid port-closure faces; 186 repairs that admission. Raw-gradient mode takes 12.5808 versus 12.5513 seconds and differs 10.5032% in anchor reaction versus the already unqualified multiplicative control. |
| Mass-weighted cavity response direction | The velocity operator contains nodal mass, motivating M^-1*dV/dx rather than the force covector alone. Nonuniform-mass CPU invariants pass. Against a 6.99398-second default control, additive/multiplicative variants take 12.1170/12.4218 seconds and fail 1% (maximum acceleration 2.7927%/4.4186%). Neither is promoted. |

The existing certified-empty-restart reuse and the exact-zero option are
mutually exclusive admissions. A fresh API-validation-off three-root comparison
keeps all 17 endpoint groups exact for both. Post-first body costs are
7.205796293 seconds (control), 7.061562791 (zero action), and 7.073869667
(certified-empty reuse). The zero-action mechanics factor is approximately
0.0000354029x. This is a single short timing pair without GPU timing
instrumentation; ordinary endpoint/work-counter logging remains enabled.

## What the source and execution now establish

All 54,133 active Human tetrahedra are nonmixed and stateless. The graph path
already consumes cached material tangents, with zero generic reconstruction
fallback rows in the audited current run. Repeating a fifteen-state
factorization optimization would not accelerate this Human workload. The latest
detailed stage profile is build172, not build187: linear solves account for
about 5.34 of 9.80 seconds, versus roughly 0.50 assembly and 0.35 globalization.

The ordinary global-coarse census is not the Schur experiment. In the actual
active-contact/node-promotion Schur run, 581 cores retain 5,358 interior nodes.
The reported 10,083 potential contact nodes are an inventory; potential-only
anchors are removed before that partition and current contact nodes are
promoted each Newton. Removing that inventory count would not provide a valid
new reduction.

The supported prelude already includes full gravity, recruitment, real support,
tissue/contact and vascular loading. It starts balanced with approximately
811 N support and three active support rows, then accepts 29 near-stationary
steps. The low-word replay fails at root 29 in passive-tissue nonlinear
convergence: nonvascular norm about 4.4568e-4 against 1.25e-5; vascular norm
6.61e-6 passes. The maximum configured 64 linear iterations are already in use.
No settled checkpoint or qualified larger timestep follows from this replay.

## Reproducibility and resources

Each frozen cohort retains source closure, patch, asset identities, binary and
shader hashes. The invocation-identity reconciliation explicitly preserves the
original stale descriptive fields in old copied invocations and maps actual
argv binaries to independently verified frozen provenance. Those original
records are retained; timings are unchanged.

The reusable run_human_frozen_invocation.py entrypoint now verifies every frozen
artifact/source hash, fingerprints file arguments, records fresh invocation and
process fields, and clears ambient simulation selectors before applying the
declared configuration. This replaces trial-specific copied receipt fields.

Exact SHA256 comparison and independently rehashed temporary hardlinks
deduplicated 20 of 21 completed 180–185 replay packs while preserving every
path. Available space increased from about 14 to 19 GiB. No unique source,
asset or evidence was deleted.

The next physical diagnosis is the actual root29 tissue residual, captured
before rollback. Full conditional solver-graph coverage, accepted-step
recycling, qualified reduced coordinates/timesteps, sustained 10/60/330-second
coupled runs and 15-fps end-to-end 0.1x remain open. Failed candidates must not be
enabled in the accepted scene.
