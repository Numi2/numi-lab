# Conditional larger spatial reference

Only proceed if the hash-bound n=64 repeat and n=128 references both pass their
separate temporal gates, and their spatial comparison fails. If it already
passes, no larger mesh is needed for this comparison. If either time test or
provenance fails, stop: a spatial conclusion would be invalid.

First run the previously frozen compact FP64 reference at n=128 and h=0.5 µs
and compare with the retained ordered evaluator at the same mesh and time grid.
Require the maximum sampled force-trace difference, binned-force, sampled peak,
total impulse, and cumulative-impulse differences each to be no more than 1e-8
when normalized by the ordered reference. Check every area weight, total area,
mass, trace, physical clearance, executable, source and output hashes. The
prior n=32 parity and n=256 noncontact one-step capacity checks are necessary
but are not n=128 parity or an n=256 contact result.

Only on successful n=128 parity, run n=256 at h=0.5 and h=0.25 µs for the
same full 10 ms physical fixture. The exact material, activation width and
reference-area coefficient remain unchanged. First compare those two time
steps using the existing gates: 0.25% binned force, 0.25% total impulse, 1%
sampled peak and 1% cumulative impulse. Only if that passes compare n=128
h=0.25 µs with n=256 h=0.25 µs using the same spatial gates. This remains an
independent FP64 reference comparison, not a native n=256 rollout, physical
material calibration, or proof of a continuum limit.

The reference is a potentially multi-day CPU job. Wait for the current n=64
repeat and all preceding studies to finish, use low-priority scheduling, and
stop on a recognized competing Numi process, 6 GiB resident memory, less
than 3.1 GiB free storage, or seven days per run. Recognized-process checks do
not detect all host activity. Stream stdout into gzip so the mostly-zero
n=256 per-node area array does not exhaust storage. Validation must decompress
and check every weight; compression cannot replace the physical checks. Retain
partial output, exact source and binary identities, execution records, errors
and fail-closed decisions. Do not call a stopped or partial run successful.
