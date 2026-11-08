# Independent P2 contact reference: spatial gate failed

The full 10 ms independent CPU P2 continuum reference passed its input audit,
registered n16 coarse/fine temporal gate, and sampled geometry, quadrature,
momentum, local mass-solve, energy, and work checks. Normalized coarse/fine
force-bin, impulse, and work differences were 7.36e-9, 1.48e-9 and 9.25e-10.
The preregistered n8/n16 spatial comparison failed: force-bin difference was
4.9085% of n16 peak and impulse difference was 0.66949%, against 0.25% limits.
This pair is not a force-qualified spatial reference. Sampled checks do not
establish clearance over every point and instant, and temporal convergence
does not establish spatial convergence or native production replay.

The complete committed plan, input-bound audit, output trajectories, scripts,
resource logs, and retained failures are at immutable branch commit
[94490e19e](https://github.com/Numi2/numi-lab/tree/94490e19e/docs/evidence/contact-p2-n16-restart-diskguard-20261108).
The local analysis JSON is an exact copy from that commit. Earlier artifacts
and source code are in the same commit history. The fine run overlapped other
work; its elapsed time is not performance evidence. Both experimental
contact and CCD options remain disabled.
