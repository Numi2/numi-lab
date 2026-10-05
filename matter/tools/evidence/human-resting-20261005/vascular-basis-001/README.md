# Existing vascular solver: implicit basis-column assembly

The integrated GPU profile found about 7.0 ms/step in Matter's vascular
continuation. Dense45 previously wrote each of 45 unit direction vectors to
device memory and synchronized twice per column. It now evaluates each matrix
entry with an implicit unit-vector accessor through the same vascularEquation
and pressure-tangent functions. The LU order, pressure/valve laws, nonlinear
acceptance, scales, tolerances and all state arenas are unchanged. Ordinary
device-vector JVPs retain their original path.

All execution used SSH Mac mini, Apple M4 Pro. The six-second native control
and candidate have byte-identical coupled CSV, surface CSV and terminal body
q/v record (3,000 accepted steps, complete breath and repeated heartbeats).
GPU update time fell from 56.515 to 52.682 seconds; whole-run native wall time
fell from 63.149 to 58.692 seconds, a 1.076x speedup. Achieved real-time factor
was 0.1022, still below real time. These are two measured runs, not a repeated
benchmark distribution. The 128-step timestamp probe measured 7.021 versus
5.649 ms at root 100.

The existing vascular GPU regression passes resistive, reverse and inertial
flow, opening/closing valves with species exchange, branched transport,
refinement, exact-clock carry, rollback, replay and invalid restore checks.
The existing native integrated transaction probe also passes: rejection after
an accepted predecessor preserves body, circulation, respiratory and Brain
history; retry matches uninterrupted replay, including an inert suffix in the
same command buffer.

A further graph-sparsity skip experiment retained identical traces but took
5.757 ms in the sampled vascular stage, with no demonstrated advantage over
the simpler basis accessor. It was removed; its comparison and full Mini
run remain retained. This change adds no CPU stepping, state copy or solver.

This is numerical/performance evidence for the existing integrated path,
not final anatomical or five-minute acceptance. The old geometry map remains
in these inputs. Original logs and continuous recordings are retained in
`/Users/n/numi-human-resting-evidence-20261005/native-vascular-basis-6s-001`
and the companion `native-vascular-basis-*` directories. The exact consumed
binary, libraries and assets are recorded in the included run metadata.
