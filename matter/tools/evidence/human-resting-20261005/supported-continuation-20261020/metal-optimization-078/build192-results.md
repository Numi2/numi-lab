# Build 192: conditional accumulation, contact clocks, grouped refinement

SSH numi only; Metal API validation enabled; no visual verification.
All three candidates remain opt-in. This cohort does not establish sustained or integrated 0.1× performance.

## Validation

- FEM compiler fixture: 45 checks, including deterministic default scheduling, common contact clocks, fingerprints, and unchanged FEM geometry/mass.
- Registered skin/fat label compiler fixture: PASS, including grouped source-face ownership/capacity checks.
- Focused native completed-cycle suite: 282 checks PASS. All six correction domains have direct/indirect parity; completed zero-work tails preserve sentinels. Root review added an empty-domain omission before binding potentially absent arenas.
- Existing 22 true-residual source contracts: PASS.

## Actual full-scene measurements

Three-root validation-on startup control body times: [7.299405709, 4.792278208, 3.467813958] s.
Completed-cycle all-domain candidate: [7.158271541, 4.803508833, 3.476613416] s.
All 17 endpoint comparison groups are EXACT. Post-first cost is 8.26009 vs 8.28012 s: no demonstrated gain.

Common contact-clock candidate: [5.529257292, 5.857236250, 5.295373541] s.
Startup Krylov columns fall 1587 -> 196; budget-exhausted solves 24 -> 0.
The later steps cost more because more tissue advances on each tick. No net throughput improvement.
Old-versus-new endpoints fail the 1% comparison (maximum generalized acceleration 4.32%, peak anchor reaction 2.28%, max J 2.11%).
The schedule change needs a finer common-clock reference before any physical promotion.

Grouped reduced-coordinate candidate uses 659 provisional source-face witnesses, 60 donor-local refinement patches, 64-face base regions and 10 optic bins per side. Its selector records the differing original build171 solver controls and does not treat them as strict A/B evidence.

The first candidate preparation correctly rejected inherited dense45; both comparison arms were then staged with dense45 removed. A second preparation correctly rejected a missing optional witness root; the retained final run selects root 0. No admission restrictions were bypassed.

Matched one-root staged full model vs hybrid body cost: 8.492486583 -> 10.773611958 s (slower).
Both accepted the native step. The hybrid's own full-operator certificate remains mandatory; its incompatible legacy diagnostic certification variable is absent.
Peak anchor difference is 2.68%; accepted anchor-history relative L1 error is 5.3869%, exceeding 1%.
Global volume-weighted Green-strain difference is 0.0930%, but registered regional differences exceed 5%, so the global number cannot establish regional fidelity.
The strict existing fidelity analyzer fails. The candidate was stopped before a longer supported prelude.

## Scope and next work

These are bounded startup diagnostics, not supported resting equilibrium or timestep/long-horizon qualification.
No force was suppressed, precision reduced, threshold relaxed or failed state promoted.
The remaining source-bound targets are complete completed-solver graph suppression, exact Schur partition improvements, and the supported-rest contact/convergence failure retained in build191.
