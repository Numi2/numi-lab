# One smaller backward Euler step: accuracy gate failed

The preregistered n=2, 1.953125 us backward-Euler run completed all 5120
requested steps, passed its existing physical and replay checks, and was
compared with the unchanged, temporally qualified 0.25 us reference. It did
not pass the shared accuracy limits: binned force error 9.9839%, sampled peak
force error 1.3741%, total impulse error 2.2138%, and maximum cumulative
impulse error 2.2138%. The area sum was 0.0003999999607912 m². The prediction
that at least one gate would fail was supported. Binned force and peak force
are separate measurements. This is a synthetic two-division fixture and does
not establish spatial convergence.

The run used a newly rebuilt probe and shader, whose hashes are in the execution
record. The exact binary and shader were copied to the build directory under
`evidence-be-h001953125`; the earlier comparison used different binary hashes.
The 483-second observational elapsed time includes diagnostic snapshots,
replay and a concurrent CPU reference computation; it is not a cost benchmark.
There is still no accuracy-qualified backward-Euler partner or demonstrated
midpoint speedup. Untested smaller steps may behave differently.
