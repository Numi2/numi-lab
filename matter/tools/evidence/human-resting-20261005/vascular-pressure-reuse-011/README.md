# Reuse constitutive pressure during the existing dense45 assembly

The integrated profile places about 5.5 ms of each 2 ms physical step in the
existing vascular continuation. During one Jacobian assembly, candidate
compartment volumes and elastance are fixed. The dense45 Metal kernel now
evaluates each of the 21 pressure/tangent pairs once in threadgroup scratch
and reuses it across the 45 basis columns. The pressure laws, Jacobian
arithmetic, LU order, acceptance, tolerances and persistent state are unchanged.
Ordinary vector JVPs retain their existing path. This adds no CPU stepping.

All builds and execution used SSH Mac mini, Apple M4 Pro. A matched pair of
3,000 accepted steps (6 simulated seconds) has byte-identical coupled and
surface CSVs, base MRVPACK, and complete terminal body q/v record. GPU update
time was 52.40716 s before and 52.02291 s after (about 0.73% less). Native wall
time was 61.29929 s and 60.92161 s, respectively. The candidate achieved
0.09849x real time. This small single-pair difference is not a repeated
benchmark distribution; CPU anatomy preparation was also active on the host.
The sampled vascular stage at step 100 was 5.469875 versus 5.342208 ms.

The existing vascular GPU suite passes resistive, reverse and inertial flow,
valve opening/closing, species transport, conservation, refinement, exact-clock
carry, rollback, replay and invalid restore checks. The integrated rejection
probe also passes: rejecting after an accepted predecessor preserves body,
circulation, respiratory state and Brain history; retry matches uninterrupted
execution, including the inert suffix in the same command buffer.

`control/` and `candidate/` retain exact invocation, source patch, hashes,
logs and compact traces. Full packs and unretimed native movies remain at
`/Users/n/numi-human-resting-evidence-20261005/integrated-pressure-cache-{control,candidate}-6s-011`.
The candidate's paired comparison includes the terminal-state equality check.
Source trees were frozen and no consumed inputs changed during either run.

A first build command used an incorrect underscore target name, and a first
regression invocation used an incorrect executable directory. Both failures
are retained on the Mini under `pressure-cache-*011*`; the successful retry
uses target `numi-matter-vascular-check` and executable `build/matter/numi-matter-vascular-check`.
The six-second drivers also retain preflight quoting/missing-driver failures
that occurred before physics. These engineering checks do not qualify the
remaining anatomical defects or the requested five-minute integrated scene.
