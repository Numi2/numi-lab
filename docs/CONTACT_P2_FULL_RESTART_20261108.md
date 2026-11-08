# Full 10 ms P2 fixed-law reference plan

Register before running. Preserve the physical geometry, material,
continuum contact energy, consistent mass, force law and order-18 surface
quadrature used in the prior n8 and n16 studies. Before 4.5 ms contact is
inactive: use the exact freefall state at 4.5 ms and integrate from there to
10 ms. Force impulse is zero through 4.5 ms, so bins 0–4 ms are zero, the
4–5 ms bin is the 4.5–5 ms impulse divided by 1 ms, and later bins come from
integrated impulses at 1 ms boundaries. The trajectory samples are 0.1 ms
apart from 4.5–10 ms. Initial state/contact law, analytical acceleration and
nodal equations are unchanged. Store the sampled scalar diagnostics, not all
nodal states.

## Prerequisite: n8 exact-freefall reconstruction

The earlier n8 restart reproduced only the first bin of the retained full
n8 fine trajectory. Run the full 4.5–10 ms n8 reconstruction with fine
8 µs maximum step, rtol 1e-10 and displacement atol 1e-13 m; velocity atol
is 1000 times greater. Compare all 10 force bins, integrated impulse and
contact work to the previously retained full n8 fine trajectory. Each
difference must be <=0.25%, normalized respectively by retained n8 peak
bin, total impulse and the maximum sampled absolute contact work or stored
contact energy (floor 1e-12 J). The original n8 full trajectory passed
its separate temporal and sampled physics checks.

Require positive sampled 48/64/order-18 gaps and J; reject nonpositive
order-18 stage gaps or volume Jacobians. Maximum sampled Duffy-48/64 and
order18/64 force differences must be <=0.05% and <=0.25% of retained n8 peak.
Require global momentum balance, total-energy drift and contact-work balance
<=0.25% on the prior experiment's scales. Also report the largest per-node
infinity-norm mass-solve residual, normalized by the larger of the local
force and mass-times-acceleration infinity norms; require <=0.25%. This is a
local algebraic-solve check, not an independent temporal nodal-error or
continuum-force certificate. If a prerequisite or resource limit fails,
retain failure and do not start n16.

Run once, single-thread CPU with nice +10, a 2 GiB sampled-RSS stop and a
2,400 s process-group timeout. Sample RSS every second and memory pressure
plus total swap usage every 10 s. Stop on three consecutive samples of
system memory free below 15% or total swap usage more than 256 MiB above
launch baseline; stop immediately on monitor failure. Log the actual total
swap output and all readings. The preceding n8 full fine run took 893 s and
about 779 MB peak RSS; time and memory are uncertain. Coordinate other
owners: correctness GPU work may proceed, but do not use overlapping work
for performance measurements.

## n16 full comparison after prerequisite

Only if the n8 reconstruction and its bound input hashes pass, run n16
coarse/fine sequentially with respective 16/8 µs maximum steps, rtol
1e-8/1e-10 and displacement atol 1e-12/1e-13 m; velocity atol is 1000 times
greater. Use 5,400/10,800 s process-group timeouts and 4 GiB sampled-RSS
stops, single-thread CPU and no GPU. Stop after any failure. The 0.5 ms
onset runs took 261/448 s and peaked about 3.05/3.06 GiB; naive duration
scaling suggests 48/82 minutes, but later contact and quadrature could take
substantially longer. These are rough resource estimates, not performance.

Before spatial comparison require each run to pass the same sampled geometry,
quadrature, momentum, local mass-solve, energy and work gates, now normalized
by the n16 fine peak bin and impulse. Require maximum force-bin difference,
total impulse difference and contact-work difference between n16 coarse/fine
<=0.25% on the same n16 fine scales. Then compare the n8 fine and n16 fine
force bins and total impulses, normalized by n16 fine peak bin and impulse,
against the unchanged 0.25% spatial thresholds. Keep temporal and spatial
results distinct. One passing same-family pair is not proof of spatial
convergence; surface quadrature, clearance and J checks cover sampled points
and stages only. This independent CPU P2 continuum model is not native P1
production replay or physical material validation.

The machine has 16 GiB unified memory; prior n16 RSS alone cannot establish
safe concurrency with native GPU tests. Require an explicitly coordinated
host window, and do not claim swap-free concurrency without a representative
pressure and swap measurement.
