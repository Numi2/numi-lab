# Blood MPM one-frame cadence check

This is a bounded no-outlet control for the NUMI_HUMAN_BLOOD_MPM_ENABLED receiver option. It checks that an enabled dormant receiver can run the configured MPM internal cadence while the resting/Brain accepted frame and vascular clock still advance once. It does not exercise active blood particles, outlet transfer, long-horizon stability, or physiological equivalence.

## Compact-grid pass

The tested binary ran the same 250 us full frame twice, first with the receiver disabled and then enabled. The enabled world retained the production cell width (0.05 m) and maximum internal rate exponent (8); its declared grid was [-0.25,0.25] m on each axis, producing a cooked 14 x 14 x 14 grid (2,744 nodes). All 128 receiver slots remained inactive.

Both modes accepted one Respiration frame at target time 250 us, and the zero-based Matter accepted cursor remained at step 0. The enabled run reported 256 microticks and 256 candidate-ready/microstep callback invocations, while frame-begin and frame-complete each ran once. Both vascular clocks ended at (high=0, low=562949980160). Per-case Brain accepted-root fingerprints matched their own expected roots; they are not compared across different world fingerprints.

The frame metabolic O2 and CO2 increments matched the baseline exactly (ratio 1 for each). Gas budget O2/CO2 maximum errors were zero, blood continuity residual was 8.314842148e-10 m^3, and final circulation volume was 0.00515000103 m^3. The raw maximum absolute difference over the vascular state vector was 0.02850853838; because that vector contains mixed-unit fields, this is not a dimensionally uniform error metric and the one-frame result does not establish identical physiology.

The exclusive-wrapper run took 5.0587 s. /usr/bin/time -l recorded maximum resident size 2,810,511,360 bytes and peak memory footprint 3,795,962,472 bytes. These are recorded observations only; this evidence does not claim a memory improvement.

## Earlier broad-grid attempt retained

The preceding broad-grid attempt used the same 0.05 m cells and exponent 8 with default bounds [-1.5,-1.5,-1] to [1.5,1.5,3] m. It ran for 109.151 s, then exited 1 because the probe incorrectly expected a one-based Matter cursor; the cursor is zero-based. Its invocation, log, exit receipt, and build receipt are retained in broad-grid-attempt-109s/ and broad-grid-build-receipt.json. This failed assertion was corrected in the compact-grid source; it is not treated as a physics failure or a successful cadence result.

## Rebuild

rebuild.py reuses the pinned compile/link argv, stages the exact source with the repository header available at the source-relative include path, and verifies external source/material/runtime hashes before building. Pass --run to execute the same one-frame check through the exclusive wrapper. The dependencies in the receipts are machine-specific absolute paths and must exist at the pinned versions.
