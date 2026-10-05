# Structural-zero vascular Jacobian assembly

The exact, host-validated 21-compartment/24-edge vascular-only solve previously
evaluated all 2,025 entries of its 45-by-45 Jacobian. Continuity rows depend
only on their own volume and incident flows; each flow row depends only on
its flow and the two endpoint volumes. The kernel now initializes structural
zeros and evaluates those possible dependencies through the existing owner
JVP (141 entries for this graph). It retains the same 45-row partial-pivot
solve, pressure and valve laws, working set, scaling, tolerances, and GPU state.
No host per-step physics or additional persistent buffers were introduced.

The complete integrated run had identified vascular work as one of the
largest GPU stages. Two six-second native comparisons on the SSH Mac mini
(Apple M4 Pro), in control/candidate then candidate/control order, measured:

| Run | Total GPU seconds | Native wall seconds |
| --- | ---: | ---: |
| Control 1 | 52.838258 | 63.337970 |
| Candidate 1 | 51.389141 | 61.823077 |
| Candidate 2 | 51.104891 | 59.819200 |
| Control 2 | 52.853171 | 61.596843 |

Mean GPU time decreased from 52.845714 to 51.247016 seconds (3.025%). CPU
asset preparation was active independently, so these are local comparisons
on this device, not a hardware-general benchmark. Real time remains unmet.

All coupled CSVs, rendered-surface audit CSVs and retained MRVPACK geometry
were byte-identical across all four runs, including accepted step639 and
step2783 captures in the first pair. Inputs did not change during execution.
The existing vascular suite passed; its general transport tests exercise
the unchanged general solver. The Dense45 coupled transaction check passed
bitwise rejection (including circulation clock and Brain history) and
bitwise accepted replay. The integrated pair exercises the changed Dense45
path through repeated valve transitions and complete beats/breaths.

`comparison.json` retains output hashes, per-run timing, test hashes and
comparison order. Each run directory contains its full source/asset/binary
invocation identity, completion record and compact GPU profile. Full native
logs and source diffs remain at the exact Mac-mini paths identified by the
invocations and `retained-artifacts.json`. The source patch and analysis
driver are retained alongside them.

These runs use the same numerically admitted scene as `costal-stability-018`.
Its known lung–rib, cardiac-wall and liver-interface defects remain explicit;
the optimization does not establish anatomical acceptance.
