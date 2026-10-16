# Native 534 Metal O0/O3 helper command

This is the corrected package for comparing actual Metal optimization levels. Both runs use the same source-bound Release host checker binary. The O0 arm loads a separately linked metallib compiled from the full source201 matter_kernels.metal AIR built with Metal -O0 -fno-fast-math; the O3 arm loads the integrated source-bound O3 metallib. The Debug host build receipt is also bound and checked for the same 817-file map, but its executable is not used, so host optimization is not a comparison variable.

The O0 AIR was compiled from the immutable source201 snapshot and linked with xcrun metallib. The command, tool path, AIR hash, linked metallib hash, and zero-dispatch link-only scope are in metal-o0-link-record.json. Both arms invoke the checker with the corresponding metallib as argv[1], which the production-kernel checker explicitly loads.

The checker calls the actual source201 restart RAP build/factor/fallback/identity/clear/reset kernels. It checks a synthetic E2 mixed case: environment 0 is active while environment 1 has a pre-set tail marker and valid or invalid RAP/LU state; it then resets markers and confirms both environments rebuild with E*m*m coverage. The marker is seeded directly rather than produced by a real FGMRES convergence event. Reset assertions verify the tail and factor-valid words, not every reset payload. It performs zero physical transactions.

The runner validates the 817 selected source files, frozen source map, both source-bound build receipts and all artifacts, compile flags, test registration, AIR/link evidence, both libraries, and unique output path. It requires 2 GiB free before launch and at least 512 MiB after; refuses other live native workers; clears NM_ and NUMI_ environment variables; checks the worker guard before each arm; and caps both sequential runs together at 120 seconds. It saves stdout/stderr and run receipt on failures. Exact stdout/stderr and exit parity are checked across O0/O3, but a pass is only synthetic kernel evidence.

The earlier host Debug/Release package, which accidentally used the shared O3 metallib for both arms, is retained separately at native534-regional-rap-tail-preserve-command-001 and is superseded for this comparison.

The final preflight command is:
 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 /Users/n/cardboard-implementation-20261014-host-macmini/analysis/native534-metal-o0-o3-rap-helper-command-002/runner.py --preflight-only
It must show ready and no active worker before:
 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 /Users/n/cardboard-implementation-20261014-host-macmini/analysis/native534-metal-o0-o3-rap-helper-command-002/runner.py --execute

Output is reserved at native/regional-rap-tail-preserve-534 and is not shared with the full-coupon command.

## Separate fresh-from-zero parity outline (not run)

Run the complete 60-control source201 coupon from zero with the source198 numerical settings. Save control-16 as a milestone, compare controls 0–16 against source198 accepted-prefix/checkpoint 530 per environment and transaction history, and continue to the same full schedule. The corrected E*m*m launch can change environment 1 coverage, so report any divergence rather than imposing whole-world byte equality. A later resume must use a source201-produced checkpoint. This helper package contains no coupon command.
