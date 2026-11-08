# P2 n=16 setup-only feasibility check

The n=8 P2 force and impulse comparisons fail the unchanged 0.25% spatial
gates. Before considering a larger P2 trajectory, measure only n=16 model
construction, consistent-mass assembly and sparse factorization on the
current host. No dynamics, force reference or GPU run is performed.

## Resource limits registered before execution

Run the unchanged P2 Model(16,18) in a child process with one BLAS thread.
Sample its resident set size every second. Kill its process group if
sampled RSS exceeds 4 GiB or wall time exceeds 180 seconds; record the
first limit or failure and retain all partial output. Capture host load and
memory before and after. On completion record DOFs, tetrahedra, mass and
factor nonzeros, total mass, elapsed time and sampled peak RSS. Require
positive mass, 0.0088 kg total within 1e-10 relative, and 35,937 P2 DOFs.

Sampling RSS is not a hard operating-system memory limit; instantaneous
spikes between samples may be missed. A successful setup says nothing about
full-trajectory runtime, temporal or force accuracy, continuum clearance,
spatial convergence or production qualification. A separate preregistered
run would still be required. The unchanged force and impulse gates remain
open.

## Result

Setup completed in 10.1 s with sampled peak RSS 1,611,872 KiB (1.54 GiB),
under the 4 GiB sampled limit. The model has 35,937 P2 DOFs, 24,576 tets,
977,793 mass-matrix nonzeros, 69,689,492 sparse-factor nonzeros and total
mass 0.0088 kg. Before/after host snapshots each reported 85% system memory
free; load averages were 5.37/4.65/4.31 and 5.00/4.59/4.29.

The [evidence](evidence/contact-p2-n16-setup-20261108/) includes the
one-second RSS samples, identities and host snapshots. Sampling can miss
short memory spikes. Neither full-step runtime nor full-trajectory peak RSS
was measured, and this does not qualify spatial force or other physical
behavior. No trajectory or GPU was run.
