# Distributed contact response containment

A native MetalWorld fixture with CCD disabled places 257 sphere-pair contacts
in one moving rigid island. On the unmodified 304fd4b shader, all requested
1/2/3/16-sweep steps reported success although normal velocity oscillated
between +250.038 m/s and the original -1 m/s. The 33-contact non-distributed
control converged near zero. This is independent of the directional CCD and
augmented-Lagrangian experiments, which remain disabled.

The distributed final sweep now rejects an uncertified local normal
complementarity residual. Its interim tolerance is 1e-4 m/s absolute plus
1% of the larger pre-solve or target normal speed. The residual uses the
appropriate sign at an explicit upper impulse limit, requires finite
quantities, and reports the first failing constraint with status
`MR_STEP_DID_NOT_CONVERGE`. This threshold is a containment criterion, not
a validated physical-force accuracy requirement.

In the retained 24-case native comparison, all four incoming 257-contact
sweep counts reject with byte-identical q/v/scene restoration and empty
accepted contact cache; zero-velocity and separating 257-contact cases, and
all 33-contact controls, complete. Replays match. The existing owner probe's
257-contact tile case also rejects: its evaluated target is 0.265333 m/s,
incoming normal velocity zero, and recorded residual 92.2678 m/s. That probe
previously required success and thus exposes an unresolved useful-motion
regression, not a reason to accept its large residual.

This change is an interim fail-closed measure. It does not make the
unrelaxed distributed Jacobi iteration converge; the supported 257-contact
impact must subsequently complete with correct forces and replay. It does
not qualify friction, coupled articulations, moving shapes, rods, meshes,
hybrid CCD, or every transaction field. Raw comparison evidence is retained
in the research evidence archive.

## Uniform damping candidate: failed useful-motion gate

A subsequent candidate used the complete island contact count to damp the
simultaneous distributed updates. In the aligned 257-contact fixture it
resolved the impact at two sweeps: normal impulse 0.99982 N s, with
2.4e-7 N s linear momentum mismatch. At friction coefficient 0.5 the
tangential impulse was 0.49991 N s and the kinetic-energy change was
-0.87495 J. The fixture has zero gravity, zero applied effort and disabled
body damping, so its kinetic-energy change is attributable to contact for
this step. One-sweep trials rejected and restored.

The same update rejects a distinct 257-contact island whose normals form
three approximately orthogonal groups at 1, 2 and 3 sweeps. At 16 sweeps it
accepts, with 0.029953 N s summed normal impulse, 7.6e-10 N s linear
momentum mismatch and -0.000150 J kinetic-energy change. The default-budget
useful-motion gate is therefore not met. All trial outcomes replayed, and
rejected trials restored q/v/scene and the accepted contact cache.

The old owner probe uses initially coincident spheres and fails after the
certificate. An opt-in noncoincident variant with two sweeps also fails
with a 0.342 m/s normal residual, despite aligned witnesses; this must not
be silently reclassified as a successful owner test. Uniform damping is
retained as a failed research candidate, not a production repair.
