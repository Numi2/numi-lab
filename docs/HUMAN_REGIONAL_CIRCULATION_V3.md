# Human regional circulation v3

`HumanPack.physiology-native.v3` admits the aggregate CVSim21 circulation source
variant authored by Numi Human's `numi human-circulation` command. The physical
owner remains the existing Matter vascular transaction and Metal kernels.
Python generates inputs and compares recorded outputs; it never steps physics.

The v3 qualification token is `source_model_variant`. It adds an inverse-atan
venous compliance law, delayed cosine-pulse elastance, linear one-way resistance,
and a stateless Starling resistance. All 21 compartments hold positive absolute
blood volume. The source baseline totals 5,150 mL; this is a source-model blood
budget, not an individual human calibration or a mechanical mass partition.

ABI 28 and package 13 include the new cooked pressure parameters and rational
period multiplier. Snapshot/archive and accepted-proof manifest shapes remain 6.
Existing v1/v2 readers retain their strict field sets and qualification tokens.
Law-specific unused fields must be zero and physical coefficients must be valid.
Atan state is admitted only for positive volume and `abs(V - V0) < Vmax`, including
its actual normalized and reconstructed Float32 initial value. The line search
stays inside this domain; restored states are checked against the same law.

Rational periods use canonical decimal UInt64 strings in JSON. Their two values
must both be zero (legacy/untimed) or both positive and reduced; a positive
rational period excludes legacy `period_seconds`. The accepted 128-bit binary
clock computes rational phase using overflow-safe modular multiplication. The
CVSim21 source rate of 70/min gives exactly 6/7 seconds per cycle. Delays and
activation endpoints are cycle fractions. The waveform is admitted only when
its delayed rise/fall fits within one cycle.

The original source has a discrete sinoatrial-node timing implementation, an
inconsistent lower-leg volume output at negative transmural pressure, and
unassigned Starling branches. The new native variant explicitly uses continuous
fixed-rate phase, continuous signed-atan storage, and the complete stateless
Starling law `max(Pin - max(Pout, Pfloor), 0) / R`. It does not claim exact
reproduction of those original implementation behaviors.

Two named volume-coordinate variants are authored independently. The default
preserves the upstream equation's crossed arterial filling-volume mapping.
`heldt_table_aligned` translates upper-body arterial V/V0 by +184 mL and
descending thoracic aortic V/V0 by -184 mL to match the source tables and Heldt's
thesis. Pressure, flow and total volume are mathematically invariant to this
paired translation; individual dilution volume and transit-time claims change.

`numi-matter-cvsim-check INPUT REFERENCE.csv` runs the full 45-unknown graph on a
physical Apple Metal device. It checks package roundtrip, exact clock, paired
bitwise environments, snapshot replay, and conservation, while recording all
volume/flow errors against the independent C++ source-equation trace. Its pass
is execution/invariant/replay evidence; the separate Numi Human analyzer gates
timestep refinement. The retained 64 ms CSV fixture is an exact prefix of that
independent reference, with its source and full-trace hashes alongside it.

Permanent checks include `matter.compiler.cvsim_admission` and
`matter.metal.cvsim_source`. Original vascular, cardiac, multiphysics, stateful,
snapshot, rollback and accepted-state tests remain required. Human's report and
receipt retain the full source and native refinement/ten-cycle evidence.

Reflexes, tilt, individual organ perfusion calibration, anatomy registration,
mechanical blood-mass partition, transported species, tissue exchange, standing,
walking and biological qualification remain separate gates.
