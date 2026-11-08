# Sustained mixed rod contact: qualification and failure diagnosis

Native Apple GPU research probe; source revision 5556cf956 (integration), candidate shader is a research-only derivative of the mixed-contact prototype retained at cb79e5621. The production status-reduction correction is f6015eec1. Both experimental CCD and augmented-Lagrangian contact options remain disabled. None of the candidate shaders in this directory is a production change.

Twenty uninterrupted 1 ms steps use a three-node rod, a sphere, a static plane, gravity, and friction. Fixed 64- and 128-sweep split solvers are discrete controls, not independent physical or material references. The capture is a read-only post-commit GPU blit; each run compares callback-free and captured exposed final q/v, scene bodies, rod nodes/edges and statuses, and repeats the captured run. Per-step q/v, rod-node records, statuses, and active contact constraints are traced. Hidden owner/contact caches are not established by these checks. "Published" in the raw owner result is not equivalent to an accepted step: the per-step status determines physical time.

| Arm | Accepted | Rejected | Result |
|---|---:|---:|---|
| fixed 64 | 20 | 0 | all steps have both rod and ground contact |
| fixed 128 | 20 | 0 | 1 ms force vs fixed 64 differs by 0.001738% of fixed-128 peak; impulse differs 0.00007972% |
| bounded 0.002 m/s, cap 64 | 20 | 0 | 1 ms force differs by 17.5282% of fixed-64 peak, failing 0.25%; total impulse differs 0.005732% |
| bounded cap 2 rejection control | 0 | 20 | rejected exposed q/v and full rod-node records equal initial state; final scene and rod edge snapshots also equal initial |
| bounded 0.0002 m/s, cap 64 | 8 | 12 | first rejected normal/tangent residual 0.0002249945/0.0002960506 m/s |
| bounded 0.00002 m/s, cap 64 | 7 | 13 | first rejected normal/tangent residual 0.00008767145/0.0001267294 m/s |
| bounded 0.0002 m/s, cap 128 | 20 | 0 | 1 ms force differs by 1.57796% of fixed-128 peak, still failing 0.25%; impulse differs 0.002957% |
| bounded 0.00002 m/s, cap 128 | 7 | 13 | first rejected normal/tangent residual 0.00008645095/0.0001255644 m/s, a plateau versus cap 64 |

The original plan inadvertently specified 5 ms force bins; ONE-MS-ANALYSIS.md records the correction to the required 1 ms gate. The original 5 ms result and the underlying traces are retained. Reporting-fix variants reproduce all original exposed values after accounting for the separately repaired failed-iteration counter. Rejected steps in those variants restore q/v and all rod-node bytes to the immediately preceding accepted state; per-step scene bodies, rod edges and hidden caches were not captured for those failures. The 128-cap accepted prefixes match their 64-cap counterparts. Capture transparency and exact captured replay pass for these arms.

The loose bounded arms maximum independently summed linear momentum imbalance on accepted steps is 3.73e-9 N s. Total estimated ground normal and tangential work are +0.000795 J and -0.001248 J. The midpoint impulse-work estimate uses pre/post rigid velocity at the recorded contact point, with the prior center for the lever arm. The fixture starts with overlap; stabilization, stored energy, rod deformation and the discrete integration rule are not a closed energy balance. Positive normal work is a diagnostic, not proof of energy creation. See analysis.json for every accepted step and the original recorded impulses.

## First-failure row trace

ROW-TRACE-PLAN.md describes a research-only shader that writes the final per-row normal equation, tangent natural residual and relative normal velocity into a scratch contact field. After removing those fields, rowtrace-control.json matches the earlier uninstrumented cap-128 trace, and rowtrace.json matches the control on all remaining captured fields, including accepted and rejected states and statuses. Both pass exposed capture and replay checks. At zero-based step 7, row maxima reproduce the normal and tangent failure certificates:

* Sphere-plane: normal equation -1.12e-8 m/s, tangent residual 0 m/s.
* First rod-sphere row: normal equation -8.64509e-5 m/s, tangent 5.72e-10 m/s, normal impulse 3.27708e-6 N s.
* Second rod-sphere row: normal equation +7.14017e-5 m/s, tangent 1.25564e-4 m/s, normal impulse 3.49153e-6 N s.

These are solver-row diagnostics, not independently evaluated physical forces or proof of root cause. Contact values and statuses are captured after the attempted step; exposed restored q/v and rod nodes are the previous accepted state. The cap-128 test cannot establish what more iterations or another algorithm would do. Next examine how near-colocated rod contacts, friction projection and coupling produce the plateau, with independently checked operator and row residuals before solver changes. Do not promote the bounded solver or infer performance improvement.

## Reproduction and provenance

Plans and amendments are retained in execution order; earlier results are not overwritten. verify.py checks retained arithmetic, rejection/restore observations and row-trace consistency. shader-sources.tar.gz contains isolated shader variants; build JSON retains native commands and host paths. Binary hashes identify executed artifacts; binaries and build intermediates are not committed. SHA256SUMS covers the evidence bundle. Historical input manifests reference the same filename sustained.mm at different versions: map the first manifest to sustained-v1.mm and the restoration manifest to sustained-v2.mm. Later failure and row records use v3 and v4 respectively.
