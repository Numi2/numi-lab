# Thin-contact energy and tiled active compaction

SSH Numi, Apple M6, macOS 27. This increment preserves the material/contact laws, activation distances, stiffness, mass, physical validity limits and accepted-state transaction. It does not qualify motion, supported rest, or 0.1x integrated performance.

## Actual energy defect

The previous full-Human trial stopped at root 9 / Newton 4 on VT contact slot 1207017. All 12 candidate sweeps were certified clear. An independent 70-digit Decimal evaluation of the captured high/low coordinates proves that the first endpoint decreases the unchanged barrier energy and satisfies Armijo. The compensated GPU energy path multiplied two small terms before stiffness, producing a subnormal intermediate and losing that descent.

Reassociate the same compensated barrier equation to apply stiffness earlier, including its C1 continuation derivative. Do not form a new epsilon or weaken acceptance. The captured old shader fails the new Float64 regression with 7.77e-7 relative energy error. The corrected shader's maximum reference error is 7.9944e-10. Testing additionally exposed sqrt(0)'s 0/0 correction in the PP energy path; handle its exact zero explicitly under the existing finite C1 continuation. Collision acceptance still requires a strictly positive certified gap.

thin-energy-old-law-regression.log is an expected failing regression. The intermediate zero-distance failures and their builds are retained, followed by the passing final result.

## Remove an active serialization

For capacities above 4096, build the deterministic active prefix with 256-candidate tile counts, a tile-total scan, and parallel scatter. Keep the legacy small-capacity path, exact ascending candidate order, contact records/history and parallel tail clearing. Scratch is one uint per tile, reused sequentially by production and private Armijo paths.

The full GPU checker passes **436,284 checks**: serial/reference list and count parity, empty and over-capacity counts, sparse/last-slot/invalid records, tail/canary preservation, existing residual/certificate checks, plus the captured contact-energy and activation/floor tests.

microbenchmarks.json retains three matched pairs with alternating order, warmup and all tiled passes/barriers. At 3,631,873 candidates the GPU-only median is **31.439 ms serial vs 3.879 ms tiled**. At 4 million it is 34.609 vs 4.238 ms. Empty candidates regress (0.00282 vs 1.049 ms) because large-capacity tile dispatches remain encoded; this requires a GPU-active dispatch follow-up. These are stage measurements, not whole-scene factors.

## Full Human result and remaining failure

tiled-compaction-barrier-native-001 executes the complete scene, 16 requested steps of 15.625 microseconds with 16 Newton slots and a 64-column FGMRES budget. It accepts **nine steps / 140.625 microseconds**, bit-for-bit identical in every coupled CSV field to the prior nine-step run.

The previous root-9 Newton-4 contact-energy stop is removed. Root 9 reaches Newton 5, then the geometric backtracking gate rejects skin object 15, tetrahedron 1456796: last trial J=2.50003362, current J=2.49985671, upper material bound=2.5. The bound remains unchanged. Failed-step physiology and controller state do not advance. The native trial remains a failed horizon, not motion verification.

First accepted step GPU command time: 21,050.985 ms in the prior completed-work trial vs 20,562.286 ms here (about 2.3% lower, one comparison with different selected-kernel profiling). Do not claim statistical sustained gain. Initialization is separate: total trial 397.60 wall seconds. Stage arrays, accepted profiles, failure certificate, plan, source/binary pins and resource receipts are retained. The largest residual cost is still the solver graph, not active-prefix compaction.

All execution occurred on SSH Numi; no visual inspection or decoder was used. Unique large native assets and ledgers remain in the source-pinned SSH evidence directory.
