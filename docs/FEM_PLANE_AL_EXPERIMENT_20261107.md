# Frictionless FEM–plane contact experiment, 7 November 2026

Implemented and run on **ssh numi, Apple M6 Mac mini, 16 GB**, from
`9c5e8fb5316dc5c24c7c39ce40055be7cae09b3d`. **Keep the AL option disabled.**
The force-qualified instrument could not complete either arm's requested
physical duration. The reference also failed its refinement gate. There is no
qualified force-accuracy result, contact-law speedup, or 10% improvement claim.
This is a retained failed/inconclusive experiment, not production qualification.

The [prepared specification](BARRIER_FREE_CONTACT_EXPERIMENT_20261107.md)
remains the original scope. Raw plans, failed calibrations, source snapshots,
traces, binary hashes and analyses are in
[evidence/contact-al-20261107](evidence/contact-al-20261107/README.md).

## Owning implementation and multiplier consistency

`RuntimeConfiguration::experimentalFEMPlaneAugmentedLagrangian` defaults false.
The host admits only immutable, explicit nonmixed FEM with frictionless fixed
planes, one fixed timestep/substep, no adaptive transfer, topology changes,
MPM, Human attachments, articulated/moving proxies or coupled callbacks. It
rejects incompatible models and requests before encoding the experiment.
This is a normal-contact prototype within Matter's existing solve, not a
reproduction of a complete barrier-free cloth algorithm.

For signed gap g, old force multiplier lambda, fixed penalty rho and positive
clearance target c, the law used by the normal sample is

```
F(g,lambda) = max(0, lambda - rho*(g-c))
E(g,lambda) = F(g,lambda)^2 / (2*rho)
dE/dg       = -F
active d2E/dg2 = rho
```

The omitted constant `-lambda^2/(2*rho)` has no primal derivative. In the owning
velocity solve the normal impulse is h*F and the rank-one normal tangent is
h²*rho*n*nᵀ when active (zero when inactive). These feed the existing residual,
force transfer and operator. Both certification paths use the same augmented
potential. The instrument independently recomputes the force, tangent rows and
certificate energy in FP64 from retained Metal samples. This verifies local
law consistency; it does not establish convergence or physical force accuracy.

One corrected primal Newton iterate is followed by an inexact AL dual update
from that **working FEM configuration**, using the same old multiplier. The
previous implementation published the pre-correction force; that lagged update
is retained under `candidate001`, and was replaced before the later series.
The multiplier update is not a separate converged primal/dual inner solve.

Each immutable node–proxy contact pair owns one existing impulse-history slot.
Stable pair identity, the package/topology fingerprint and the runtime program
fingerprint define ownership. Topology mutation/migration is excluded rather
than mapping a dual to a new contact. The old force is the stored impulse / h;
the fixed-cadence admission prevents reuse with incompatible impulse units.

All Newton dual writes target `contactHistoriesCandidate`, never accepted
history. Iteration zero reads accepted history; later iterations read the
unpublished candidate. Existing checkpoint, final commit, rollback, reset,
snapshot and restore own those arrays. Accepted-state restoration rejects a
different runtime identity. The AL selector, penalty and target are included
in that identity. IPC and AL observations have equal source-physics
fingerprints and different runtime-program fingerprints for every matched unit.

AL trial geometry can cross its target during nonlinear work. The unchanged
strict final geometric predicate still requires gap > 0.1 micrometre here.
An additional check requires the checkpoint gap above the same floor. For this
fixed-plane, affine nodal committed path, positive start and end gaps certify
the entire path; this does not generalize to moving or curved geometry.
Active AL contact additionally requires |g-c| <= 1 micrometre. No trial geometry
or trial multiplier becomes accepted on a failed step.

The runtime uses the caller's command buffer. It adds no queue, commit, wait,
CPU contact solve or solver-state readback. The standalone observer owns its
queue, completion waits and diagnostic snapshots.

## Fixed comparison and measurements

Both laws use the same initially stress-free 20 mm cube, density 1100 kg/m³,
neo-Hookean mu=2000 Pa and lambda=5000 Pa, gravity −9.81 m/s², fixed plane,
mesh and authored solver settings within each pair. Initial primary gap is
0.2 mm; the IPC activation band is 0.1 mm. AL rho=1000 N/m per contact node
and c=0.5 micrometre remain fixed. Per-node penalty is not a mesh-invariant
physical stiffness; the mesh study is part of the required qualification.
Both arms retain the same strict final clearance requirement. The plans seal
input definitions; per-run compiled-physics fingerprints are retained, but
independent compiled-package byte archives were not captured before trials.

The requested duration is 10 ms. Timesteps are 0.25, 0.5, 1 and 2 ms, with
1 or 2 subdivisions per cube edge (8 or 27 nodes), in IPC/AL/AL/IPC order.
The control refinement uses 2 subdivisions at 0.125 ms and 4 subdivisions
(125 nodes) at 0.0625 ms. Each series also retains both laws at initial gaps
50 and 0.5 micrometres, with h=0.5 and 2 ms: **near-contact starts were not
removed**. These are limited load/geometry units, not broad statistical evidence.

Each trace records accepted/rejected status, normal force and impulse, gap,
height, COM, momentum, contact count, Krylov work, certificate diagnostics,
GPU time and two forms of discrete contact work. For every accepted step,
using actual contact-node displacement and the final force of the backward
Euler step,

```
W_contact = sum_i J_i · (x_i,end - x_i,start) / h
W_midvelocity = sum_i J_i · (v_i,start + v_i,end) / 2
```

The impulse vector is the reaction on the continuum at its nodal witness.
Rejected attempts contribute zero accepted physical work and impulse; their
trial contact samples remain labelled rejected. Work is not inferred from the
AL potential difference, whose multiplier changes. These discrete work
measures do not by themselves establish total-system energy conservation.

The primary cost is accepted physical seconds / wall second, with rejected
work and mandatory observation snapshots in the denominator. Initialization
and separate replay/forced-rollback checks are outside that timed interval;
GPU time is also retained. A truncated run is not compared to a complete run.
Fine per-stage dispatch profiling was not pursued after the physical gates
failed; no Newton/Krylov count is used as a speedup claim.

## Results and retained failures

The initial mixed-FEM calibration started inside the IPC band and moved upward.
It is retained, including the AL admission failure for that unsupported mode.
Its exact source/binary hashes were not captured; its JSON physics/program
fingerprints must not be relabelled as the final source identity.
The subsequent explicit-FEM primary geometry starts outside that band.

The first candidate's lagged dual update failed. The corrected candidate still
failed to finish all AL rollouts. The complete `final` series used the original
1e-5 authored nonlinear residual tolerance, then added independent force-law
and momentum-balance checks:

| Observation | Result |
|---|---:|
| Completed IPC runs / total IPC runs | 22 / 22 |
| Completed AL runs / total AL runs | 0 / 20 |
| Maximum AL force/tangent/energy relative discrepancy | 1.224e-7 |
| Maximum accepted-step momentum imbalance | 3.414e-5 N·s |
| Reference binned-force refinement change | 64.88% |
| Reference integrated-impulse refinement change | 30.45% |
| Reference final COM change | 87.61 micrometres |

The reference gate was 0.25%; it failed. The impulse residual certificate is
normalized by max(initial norm,1), so 1e-5 was also too loose for this small
8.8 g object's forces. A final sealed `tight` series changes that **fixture**
tolerance to 1e-9 identically for both laws; production defaults are unchanged.

| Force-qualified instrument series | Result |
|---|---:|
| Complete 10 ms rollouts | 0 / 42 |
| Maximum AL local law discrepancy, including rejected trials | 1.337e-7 |
| Maximum accepted-step momentum imbalance | 2.932e-9 N·s |
| Minimum accepted/restored nodal gap | 0.5 micrometre |
| Exact replay, forced rollback, identity rejection | 42 / 42 each |

Every tight run stopped at its first reported failure (Matter status 10,
outer status 11). No reference completed. For example, at 0.5 ms the primary
27-node IPC arm accepted 9/20 steps and AL accepted 12/20 before rejection.
All accepted AL steps in this series precede nonzero dual acceptance, so it
**does not qualify AL's physical force response**.

The looser series additionally exercises restoration of nonzero accepted AL
history: its maximum accepted dual is 2.708e-5 N·s. The 27-node, 0.5 ms AL run
accepted two contact-bearing steps before rejecting, with total contact work
−1.557e-6 J. Replay includes the terminal failed attempt, and forced rollback
compares complete physical arrays including the persistent dual. These are
transaction checks on an otherwise unqualified physical trajectory.

Near-contact IPC diagnostics remain visible. At 0.5 ms the looser 50 micrometre
start produces initial force 0.8523 N and accumulated contact work +3.797e-5 J;
the 0.5 micrometre start produces 1.6984 N and +1.580e-4 J. Positive work here
can release initially stored barrier potential. It is evidence of materially
different near-contact behavior, not proof of unexplained energy creation.
Those loose force magnitudes are not reference-qualified. The tight traces
also retain near-contact starts and their failures.

`candidate002` late near-start diagnostics overlapped a CCD test; their timing
is excluded from any interpretation. The subsequent `final` and `tight`
series had no competing experiment GPU work. All older attempts are retained.

## Decision and next prediction

Do not promote, tune away the failures, or expand to sliding, moving instruments
or sutures. The >=10% total-time prediction is **inconclusive** because the
control/reference and matched-duration gates fail. The current AL candidate is
ineligible for promotion. Replay, rollback and algebraic law agreement passed;
force accuracy, convergence and useful accepted simulation throughput did not.

A subsequent bounded experiment should first close the existing solver's
contact-onset residual failure at the fixed physical-error requirement and
obtain a converged reference. Only then can the same frozen comparison test
whether an AL update improves total accepted time. Preserve these attempts
when revising that prediction.

## Reproduction

From the repository root on an Apple Metal host:

```sh
cmake -S tests/contact -B /tmp/numi-plane -G Ninja
cmake --build /tmp/numi-plane -j 4
python3 tests/contact/run.py /tmp/numi-plane /tmp/numi-plane-evidence
python3 tests/contact/analyze.py /tmp/numi-plane-evidence > /tmp/numi-plane-evidence/analysis.json
```

The runner retains nonzero exits. Its own successful completion means every
planned attempt was collected, not that every simulation was accepted.
