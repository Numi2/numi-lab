# Solver, contact and memory measurements on SSH Numi

Source increments: dc0acdf00 and 0e49d084e on Numi2/numi-lab main, authored by numi2.

The shipped change bounds autoreleased temporaries to one horizon iteration,
extends the existing opt-in memory observer to accepted boundaries, and supplies
an optional matched benchmark for the two existing cached-node builders. It
changes no physical equation, material, timestep, mesh, mass, force, controller,
contact law, certificate or rollback rule. The earlier working-set spike is
not attributed or declared fixed.

## Measured experiments

| Experiment | Result | Decision |
| --- | --- | --- |
| 217, cooperative row/RHS nodal builder | Bitwise parity, but about 3.9x slower than existing incidence-SIMD on the declared synthetic mesh | Removed; rejected implementation and measurements retained |
| 218/220, final-contact metric/owner cache in dead radix scratch | All 2,151 targeted checks pass; full candidate contract 2,665,706 checks pass | Not promoted after native measurement |
| 221, same-binary cache-off control versus 220 | All 11 CSV traces and three MRV captures identical. 226.881 versus 226.800 GPU seconds; 229.249 versus 229.512 simulation-loop wall seconds | No useful speedup established; production cache removed |
| 222, accepted Z0 with current scaling | Rejected during initialization: row equilibration does not admit the recycler | Restriction retained; no physical step, no speed claim |
| 223, originally supported unscaled recycler | Four accepted steps, six injections, same 128 Krylov columns, 19.693 versus 18.483 GPU seconds | About 6.5% more GPU time; disabled |

The cache reduced root-0 final-certification span from 234.54 to 228.30 ms in
one pair. Other whole-stage costs varied by more than this difference. Neither
a repeatable overall gain nor a useful optimization follows from that sample.
No isolated speedups are multiplied.

The cached-node benchmark retained in source compares only the two existing
production algorithms: 19,683 nodes, 105,456 tetrahedra, one warmup and three
rotating measured pairs. It checks bitwise outputs and reports GPU duration and
static threadgroup bytes. It is synthetic kernel evidence, not Human mechanics
or physiology qualification.

## Full-scene scope and bottlenecks

Each completed prefix advances 32 accepted steps at 15.625 microseconds:
0.00050000002374872565 simulated seconds. All physical and coupled accepted
clocks remain matched, without root assistance. These are release-transient
prefixes; no supported resting equilibrium, repeated full breaths/heartbeats,
long motion, presentation throughput, or five-minute qualification is claimed.

The fresh cache-off control reaches 0.00000218104x for the simulation loop.
Initialization adds about 207 seconds. **0.1x remains unmet by about 45,850x**.
This does not establish sustained performance.

Both configurations record 2,304 used Krylov columns, 144 active solves,
512 encoded cycles and 62 budget-exhausted solves. The root-0 control stage
spans sum to about 1,808 ms linear, 1,016 ms assembly, 412 ms globalization and
235 ms final certification. Stage boundaries perturb encoding; use whole-run
and unselected-kernel roots as controls. Do not infer live iterations from a
maximum-iterations field or encoded dispatch count.

GPU allocation has a 10.033 GB plateau in both runs, with a transient
10.198 GB maximum at accepted step 27 (root 26, selected for deep diagnostics).
Footprint peaks at 14.351 GB in 220 and 14.392 GB in 221, against plateaus near
13.5 GB. This corrects an earlier draft that missed the one-step excursion.
accepted-memory-bounds-220-221.json retains every accepted-boundary measurement.
OS samples remain within the retained 12 GiB disk floor and 3 GiB swap-growth
watchdog. The root-26 association is not yet causal attribution or a long-run
memory bound; routine control 226 removes optional deep trace roots and
encoder-splitting GPU stage timing while retaining physical checks.

The 223 screen changes both row scaling and recycling, so it does not isolate
the recycler. Nine coupled-trace columns differ; the largest atrial-volume
difference is 0.000003638 mL. No force/energy fidelity or broader motion
qualification was attempted after the performance screen failed. The supported
unscaled implementation remains unchanged and disabled by default.

## Exact recovery and limitations

Plans retain launch arguments, asset identities, timestep, environment, source
revisions, dirty-source pins, native binary hashes, runtime dependency hashes,
driver archive hash and build logs. Candidate execution-source archives retain
the experimental source, including the removed cache. Rejected compile attempts
remain labeled alongside successful builds.

compact-native-traces.tar.gz contains all 22 CSV files from 220/221.
trace-manifest.json pins every uncompressed member. The three MRVs per run
remain local as independent verified APFS COW files, with hashes in the exact
comparison records; this package does not claim their remote archival. Sources
and unique physical evidence are preserved. The original diagnostic
physical_admission=false field is a conservative runner umbrella, not an
accepted-step counter; inspect native status records and traces separately.

The current skin/fat full-coordinate conditioning, active solve convergence
and repeated assembly remain unresolved. A new contact cache with negligible
measured benefit is not retained as production complexity. The next mathematical
experiment must report current-operator actions and numerical/force/energy
fidelity, and must not bypass source admission or relax physical tolerances.

## Final regression and optional binding fix

After removing the rejected production cache, the existing full solver suite
passes 2,665,138 checks and the signed-gravity/analytic-momentum/bitwise-replay
checks pass. Metal API validation then exposed a separate preexisting minimum-
extent bug: the disabled contact-history argument borrowed a 132-byte vector
arena but its pointee requires 1,088 bytes. Source 0e49d084e bounds that existing
fallback allocation to one complete slot only for nonempty Stand fixtures.
The optional history arena remains lazy, and full-Human vector arenas are
already larger. The same multi-horizon production-path probe now passes with
MTL_DEBUG_LAYER=1, including split/monolithic and replay comparisons. Both
failure and passing logs are retained; this is correctness evidence, not
performance or full-Human motion qualification.

The rejected 223 MRVs are archived as a GitHub release asset. The archive and
each uncompressed member were independently downloaded and hash-verified before
retiring the local originals. See rejected-recycle223-mrv-manifest.json and
rejected-recycle223-remote-restoration.json. This is exact archive recovery;
no image or video was opened.

## Routine observation control 226

The complete scene again accepts 32 steps at the unchanged 15.625 us, with all
11 CSV traces and three MRV captures bitwise identical to221. Deep selected
root 26 trace flags and GPU encoder-splitting stage timing are removed; compact
work counters, accepted-memory observation and physical certificates remain.

Measured loop: 229.71753575 s; GPU command spans: 227.293288125 s;
simulated time: 0.000500000023749 s; loop real-time factor: 0.00000217659x.
Initialization-inclusive wall time: 416.134 s. Neither initialization variance
nor the small whole-run timing difference establishes a useful speedup.
The active solve counts remain 2,304 columns / 144 solves / 62 budget exhaustions.

Accepted-boundary process footprint is 13.469-13.524 GB, versus 13.567-14.392 GB
in 221. The selected-root footprint excursion is absent in this control, but
GPU allocation still spans 10.033-10.195 GB. Do not claim the GPU working set
or the older historical spike fixed. The scene remains too large and costly.
No sustained, presentation-inclusive or 0.1x qualification is claimed.

The control 226 source snapshot, exact launch/dependency hashes, execution,
resource samples and comparison are retained. Its CSV bytes are identical to
the archived 220/221 members and can be restored through those hash mappings;
three local MRVs remain independently writable verified COW files. The rejected
223 MRVs have separate independently restored remote release storage.

Continue algorithmic work from the production source, not from the removed
cache/recycler experiments. Skin/fat slivers, active operators and repeated
assembly still require a measured solver/model improvement with current-
operator certification and the existing force/strain gates. Earlier EW caps,
coarse reduction 187 and thin-cell toggles are failed or unqualified evidence,
not default settings to repeat. Larger timesteps remain retired by the user.
