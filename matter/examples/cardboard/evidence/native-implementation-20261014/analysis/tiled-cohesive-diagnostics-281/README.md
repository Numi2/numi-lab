# Tiled cohesive diagnostics overlay (frozen source 085)

This external overlay adds failure-preserving diagnostics to the standalone `--cohesive-tiled-parity` mode. It does not alter the material, closure/opening commands, solver settings, tolerances, kernels, or acceptance policy.

Apply `tiled-cohesive-diagnostics-085.patch` to the exact frozen source named in `manifest.json`. The only source file changed is `matter/tools/cardboard_full_contact_armijo_check.mm`.

Run it with the same material input and a fresh output directory:

```sh
numi-matter-cardboard-full-contact-armijo-check BULK.nmatter \
  --cohesive-tiled-parity \
  --diagnostic-state-output /path/to/new-tiled-parity-run
```

The output contains `face-count-1/legacy`, `face-count-1/tiled`, `face-count-3/legacy`, and `face-count-3/tiled`. Each arm has per-transaction before/after binary snapshots, JSON metadata with completion stage and accepted status, and fixture trace logs. `arm-result.json` records `completed` or `failed_prefix`, last stage, transaction count, and error. The root `tiled-parity-result.json` records whether comparisons ran or passed; it is not written as a parity pass when any arm fails. The expected zero-face initialization rejection is recorded separately.

Syntax validation passed with the exact warning-as-error flags listed in the manifest. Patch dry-run and apply-plus-byte-compare passed. No native/GPU run was performed. Case 280 remains a retained numerical failure; the overlay only makes its next reproduction diagnosable.
