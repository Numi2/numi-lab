# Full-horizon synthetic wound traction: result

## Outcome

The corrected v2 notebook reuses the retained, successful zero-force native
control and then runs one fresh loaded trajectory. Its integrity verifies, but
the loaded arm exits at 8 ms after seven accepted samples. The paired study is
therefore **inconclusive**; there is no accepted 60-step loaded result and no
paired effect estimate.

| Evidence | Result |
| --- | --- |
| Retained zero-force control | 60/60 steps; 316.2 s wall time; 305.5 s reported GPU time (60 ms simulated / 316.2 s wall = `0.00019x` real-time for this run) |
| Control maximum volume residual | `1.11993461e-6` (below `5e-5` secondary and `1e-4` native limits) |
| Loaded accepted prefix | 7 samples, through 7 ms; native exits at 8 ms on zero-based step index 7 (44.9 s wall time) |
| Last accepted loaded sample | 0.253 mN/site; minimum mean gap `0.588743268` mm; maximum displacement `8.8251` µm |
| Last accepted loaded volume residual | `8.59546635e-5` (passes `1e-4`, misses secondary `5e-5`) |
| Native failure | Matter status code 10 (`NM_STATUS_NONLINEAR_SOLVER_FAILURE`) at step index 7; 78 completed microsteps, 10 FGMRES iterations, no contacts |
| Failure diagnostic tuple | `0.000003,0.000000,0.000100,0.000003`; native output rounds this to six decimal places, so the exact failed residual component is unresolved |
| FGMRES trace | Not emitted: the candidate runtime allocates it only when `NUMI_MATTER_CAPTURE_FGMRES_TRACE=1`, which the sealed trial environment did not set |

The seven loaded states reproduce the existing short visual prefix but do not
reach the 10 mN/site endpoint. Since the new attempt adds no later accepted
state, the current seven-step clip remains the most informative video; this
result does not justify replacing or lengthening it. No needle, thread, robot,
calibrated skin, biological response, or clinical capability is demonstrated.

## Retained records

Both notebooks and all raw files remain on the Mac mini. The first registered
notebook is stopped and analyzed as inconclusive because its collector expected
the wrong native summary prefix; it contains the original zero-force run and
the failure receipt. The second registration seals the corrected collector,
binds the raw-control provenance, records a valid control reparse, preserves
the failed loaded output, and verifies its inconclusive analysis.

- Original notebook: `/Users/n/numi-skin-suture-perf-100x/runs/goal-volume-residual-full-horizon-20261004/study`
- Corrected notebook: `/Users/n/numi-skin-suture-perf-100x/runs/goal-volume-residual-full-horizon-repaired-20261004/study-v2`
- Corrected loaded samples: `trials/full60-loaded-corrected-collector/output/native/traction-samples.csv`
- Corrected loaded stderr: `trials/full60-loaded-corrected-collector/output/native-stderr.log`
- Corrected registration SHA-256: `0be554feecb0eff6f432beeee227e271bae729b606264ac7e0f9f88fb52abc6b`
- Corrected loaded attempt receipt SHA-256: `92682b108dbc403455c7b7740c9b5671d3a4528e7774684b42a919c7f3c9ee57`
- Corrected analysis SHA-256: `63779d97235e176a0d23708a73a6a800850b7ea0facf664fbac60a086b7d4ee1`

The repaired collector was calibrated against the actual retained control and
rejected a deliberately corrupted row over the unchanged `1e-4` volume gate.
That verifies the measurement path only; it does not change the failed loaded
outcome or calibrate the synthetic material to human tissue.

Although an isolated contact-free cadence probe elsewhere exceeded 100x
simulated-time throughput, this full-horizon wound-coupon control did not. Its
measured throughput is a separate workload and cannot support a whole-task
100x claim.
