# Independent slider interface-work history

The existing independent FP64 slider references retain only total FEM-side,
slider-side and relative contact work. Before another native coupled rollout,
record their cumulative histories on the existing 2 µs reference clock at both
0.5 and 0.25 µs integration steps, for both 200 µm and 95 µm initial gaps.
Do not change the model, contact law, numerical update, material, or timestep.
Keep the historic files unchanged.

The instrumented reference must reproduce every field of the matching historic
trace exactly after removing only the new work-history field. Each work-history
row must be finite, use the same time as the existing reference trace, and
satisfy FEM work plus slider work equals relative contact work to a documented
floating-point bound of `4096 * DBL_EPSILON * (|FEM| + |slider| +
|relative|) + 1e-20 J`. Compare coarse and fine cumulative work histories and
final values in absolute units. For each history require the largest absolute
coarse-fine difference to be at most 1% of the larger of the fine-history
maximum absolute magnitude and 1e-12 J; separately report the final error.
This uses the existing cumulative-history 1% scale and a fixed absolute floor,
not a tolerance fitted to the observed result. These are independent CPU reference checks,
not a native coupled step or a physical material calibration.

Record source, included model, binary, command and output hashes. Add the
additional histories without changing or weakening the previously declared
force, impulse, work-reconstruction, torque, clearance or replay gates. Retain
any parity or temporal failure rather than replacing the historical reference.

The first instrumented primary run numerically reproduced all old fields, but
its newer source also emitted `initial_gap_m`, absent from the historical
primary schema. The strict exact-parity test correctly stopped; its output and
receipt remain evidence. For a separately named repeat, derive the primary
instrument from the exact hash-matched historical primary source and add only
work-history collection and output. Use the separately hash-matched current
near-contact source for the near fixture. Continue to require exact equality
after removing only `work_trace`.
