# Trial summary manifest separator repair (overlay 003)

Overlay 003 makes one source-character correction in the common coupled-capture manifest writer. The phase-order array is already at the top level after `producer_semantics` closes; the old `]}` separator closed the top-level JSON object before `payload` was written. The corrected separator closes only the array, so both the all-trial summary and private-trial mode can emit valid JSON.

The patch is incremental on overlay 002's runtime output, SHA-256 `feaf5a062138e8dc1a84994090fde27a59a2605e863bb0d5c8ed1280e07b7fc4`, which equals the frozen source-120 runtime file. Candidate runtime SHA-256 is `ac3b2bc725a6787f342b20c1c11e8adc57ad47a80b78d9281e82cce41b9ed38c`. The only changed code line is the `manifest <<` phase-order/payload separator; no capture bounds, arrays, solver behavior, or physical decisions change.

`tests/test_serializer_roundtrip.py` binds to the candidate source, extracts the exact C++ string literal at the common serialization seam, and exercises strict JSON round trips for both summary and private-trial (`numi.matter.coupled_armijo_trial_capture.v1`) and all-trial-summary (`numi.matter.coupled_armijo_trial_summary.v1`) manifest shapes. It also checks that the former extra-brace seam remains rejected. `patch_dry_run.py` checks the sealed base hash, dry-runs and applies the patch in a temporary directory, and checks the exact candidate hash.

The triggering native run 395 is preserved. Its binary payload is 12,480 bytes with 128 arrays and valid internal hashes, but its raw manifest is invalid JSON (raw SHA-256 `ba41e4bead53b0bf3408fa4297c18dfc2ed208fc03f4d12813808c44371d6837`). The diagnostic table in the separate analyzer package uses an explicitly labeled one-byte adjudication copy; it does not make the original manifest valid. Run 393 remains a capture skip with no payload. Neither run is a successful physical simulation: both retain the expected full-contact Newton-9 failure.

This is a source/serializer repair candidate. It has portable strict JSON and patch checks only; it still needs the parent’s fresh source-121 build and native summary/private output validation.
