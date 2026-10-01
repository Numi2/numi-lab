# Open Knee source operators, October 1

Matter now executes the source cylindrical-connector equations on CPU and Metal,
and its full continuum material stress/tangent have an independent compiled FEBio
comparison. This is an equation-level milestone. **The complete source knee is
not yet assembled, initialized or physically qualified.** The shifted-ACL hybrid
and its eight-microsecond regression remain unchanged.

## Native cylindrical connector

`matter/include/numi/matter/source_cylindrical_joint.h` provides residual and exact
directional derivative for the source quasi-static (alpha=1) equations. It keeps
finite force/moment penalties, free/prescribed translation and rotation, reference
joint origin, both body frames, explicit multipliers and attachment lever arms.
Inputs are immutable and all history belongs to the caller. Unsupported/singular
states reject execution. This operator is ready for coupled residual/Jacobian
assembly; it does not yet execute the full source graph or load continuation.

The source connector has penalty compliance. Replacing it with an ideal two-DOF
cylinder or releasing three MyoSim coordinates would not execute these equations.
The quaternion logarithm retains the source branch but uses atan2 near zero;
its analytic derivative includes changes in axes and moment arms.

Evidence: [receipt](media/open-knee-source-joints-20261001/receipt.json),
[source graph](media/open-knee-source-joints-20261001/source-rigid-graph.json),
[CPU/Metal output](media/open-knee-source-joints-20261001/matter.stdout).

- Six connectors at 140 archived configurations: 840 paired equation evaluations
  against compiled public FEBio 2.9.0, not the archived 2.9.1 binary.
- Maximum force error 1.452e-10 N; maximum connector moment error 1.772e-4 N mm,
  within fixed 1e-8 N / 1e-3 N mm gates. The original acos implementation loses
  precision near zero rotation; the difference is retained.
- 2,016 finite-difference derivative components: maximum relative error 3.309e-7.
- Free axial motion and axial rotation, invalid-input rejection, and 96 mm/metre
  conversions preserve forces and rescale moments consistently.
- 1,035 Metal rows: maximum scaled float CPU/Metal error 2.580e-7.

Archived log poses are rounded. Both operators receive identical normalized
poses; these recomputed reactions are not presented as reproductions of the
archived reaction magnitudes. Equal-and-opposite connector force alone is not
whole-system angular-momentum qualification.

## Distributed continuum equations

The earlier exact fibre law and isochoric prestrain implementation now passes
486 cases against actual public FEBio 3.0 material classes, including the
prestrain wrapper and stretch generator. The probe uses no Matter equations.
Cases cover all six tendon/ligament parameter sets, two menisci and cartilage,
with axial/oblique fibres and sheared, compressed and extended deformation.

Evidence: [receipt](media/open-knee-source-material-febio-20261001/receipt.json),
[compiled reference output](media/open-knee-source-material-febio-20261001/febio.stdout),
[Metal output](media/open-knee-source-material-febio-20261001/matter.stdout).
Compressed inputs and both full result arrays are in the same directory.

Maximum Frobenius errors, normalized by max(1 MPa, reference norm):
first Piola stress 2.954e-15; directional tangent 9.254e-15. The reference's own
analytic tangent agrees with stress finite differences within 7.856e-9. The
existing real Metal material checks also pass.

Legacy energy reporting is a separate unresolved comparison. Without GSL the
public source returns no fibre strain energy; its GSL high-stretch expression
also lacks the constant joining it continuously to the toe energy, and the 3.0
prestrain wrapper does not override deviatoric energy reporting. Matter's
continuous stored-energy primitive agrees with the source force law and its
integral. These probes do not claim legacy energy-output equivalence or complete
assembled energy accounting.

## Reproduction and remaining work

Build the existing `matter` CMake project and run:

```sh
cmake --build build-open-knee-source --target numi-matter-fiber-check numi-matter-source-joint-check -j4
ctest --test-dir build-open-knee-source -R 'matter\.(compiler|metal)\.source_(fiber|joint)' --output-on-failure
```

Human's `tools/open_knee_reference/README.md` contains pinned public comparator
build and comparison commands. These remove the need for a FEBio login. The
full public 3.0 comparison loaded the original mesh and assembled the same
495,960 equations as the archive, but its first SuperLU factorization exceeded
an 18 GiB RSS cap: zero increments accepted. That failed reference remains
separate from the successful local equation comparisons.

Required before native source admission: full rigid graph/attachments and
material-frame lowering; coupled quasi-static solve including cartilage and
meniscus volumes; source prestrain/contact initialization without resetting
energy; complete swept contact; and refinement/energy/momentum/rollback gates.
The passive-flexion source contains no active quadriceps loading experiment.
A separately specified proximal-QAT experiment must predict PTL tension through
tissue deformation and patellar response before whole-body integration.

No additional global solver or repository was created. The owning Human/Matter
paths must complete those gates; none are inferred from these operators passing.
