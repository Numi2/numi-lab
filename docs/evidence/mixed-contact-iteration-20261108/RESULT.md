# Mixed rigid–rod contact: iteration and residual diagnosis

**Outcome: no candidate promoted.** The existing mixed-island solver accepts the tested two-sweep step despite exceeding the 0.002 m/s normal-residual gate in two of three frictionless cases. The isolated frictionless candidate from the [preceding qualification](../frictionless-active-mixed-20261108/RESULT.md) does not repair that failure. At higher iteration counts, the unmodified finite-friction path exhibits a different failure: a rod row retains positive normal impulse while its normal equation indicates separation. These are diagnostic fixture results, not full-scene or surgical qualification.

The native experiment ran on `ssh numi` against integrated source `1bcf2272058ab585ab81ca584840d95ddd23bc61`, using the same compiled library for each arm. The fixture has one 1 kg sphere, a static plane, three-node rod, two rod contacts and one rigid contact in one island; it uses zero gravity, a 1 ms step, initial finite overlap with a positive stabilization target, disabled CCD and incoming normal speeds −1, 0 and +1 m/s. The zero-speed cases therefore are not free-motion or force-free tests. Friction is 0 or 0.5. The tested candidate is the separate, unpromoted frictionless response patch described in the preceding qualification; finite-friction results use its unchanged legacy path.

## Findings

* A trace of the existing Wave32 relaxation gives the rigid row a diagonal response of 1, absolute response row sum 3, and damping factor 1/3. For the frictionless −1 m/s case, both rod rows have negative normal right-hand sides and zero impulses in each of the first two sweeps. The captured right-hand sides and the regularized preconditioner reproduce the first and second rigid impulses within the script's numerical tolerance. This does **not** justify pruning inactive rows: contact active sets can change.
* Across 1, 2, 4, 8, 16, 32 and 64 velocity sweeps, frictionless normal residual falls to at most `5.97e-8 m/s` at 64, but at the originally configured two sweeps two cases fail the `0.002 m/s` gate. Every run was accepted with success status and repeated deterministically; total linear momentum error passed `2e-5 N s`. This is evidence of underconvergence being accepted, not evidence that a larger fixed budget is an adequate fix.
* With friction 0.5 at 64 sweeps, the maximum reported residuals are `0.0819663`, `0.165547` and `0.119665 m/s` for the three incoming speeds. A second trace attributes each maximum to a rod row whose normal equation is positive while its normal impulse is positive. In the −1 and 0 m/s cases the independently checked rigid-plane normal deficits are about `4.10e-8` and `9.31e-10 m/s`. Thus a rigid-only normal correction cannot explain or resolve the residual plateau.
* The predeclared **isolated** split-step research shader uses a scalar normal update and a tangent-gradient update for uncapped isotropic cones; the existing cone projection and island damping remain. At 64 sweeps its maximum reported normal residual is `5.69e-8 m/s` for both friction values; cone membership, momentum accounting and identical repeated runs passed the narrow scripted checks. At two sweeps, **two of three cases still fail** for each friction value, with worst residual `0.453349 m/s`. It is not production code and has not established correct sliding direction, friction work, active-set transitions, rollback on nonconvergence, or multistep behavior. The experiment must not be enabled or promoted.

Both diagnostic trace shaders reproduce all original case outputs exactly after their additional trace fields are removed. `analyze_traces.py` and `analyze_residual_split.py` verify this, the row attribution, repeated-run checks, momentum, and the stated normal gates. The normal residual is the runtime's complementarity measure; a small residual alone is not a force-accuracy or energy certificate. No timings or general convergence guarantees are claimed.

## Reproduction and retained evidence

The preregistered plans, instrumentation generators, isolated split patch, native probe source, compiler commands, all 42-case JSON results and analyses are retained here. The generators reference the remote paths and prior frictionless candidate used in this run; the preceding qualification contains its patch and fixture details. No compiled binaries are committed. SHA-256 values identify the actual native artifacts:

| Artifact | SHA-256 |
| --- | --- |
| Common native probe | `12a75d1ec99094b094d0e0820cae9a68a9f20f177aba1b27b68ba90c69bd6b3b` |
| Common `libmetalrobo.dylib` | `2c3c88d26e7d141d965b61d5c76826674d42055e4ab8036669879bc3106e585e` |
| Prior frictionless candidate metallib | `de4b28c6bee9d50c54dea783bd03d7cd2053ada158834b5b8a9fce2920df2c02` |
| Residual-trace metallib | `e96dcf23e1411a5ee27ee68401df22322aed75c883963e8bfb741a9b2b0eb95f` |
| Split-step research metallib | `eb50d360aa00d89390efb52ec8f59637e8b21398b968aab689b2443db1511ccf` |

Next, design a bounded, residual-certified mixed solver with explicit failure and complete restoration when it cannot meet the required tolerance. Independently evaluate normal complementarity and tangential sliding/work at changing active sets, including coupled articulated and rod cases, before considering production integration. Separately continue the already scoped FEM-plane contact-onset diagnostic and force-qualified control; neither disabled CCD nor augmented-Lagrangian experiment has been promoted.
