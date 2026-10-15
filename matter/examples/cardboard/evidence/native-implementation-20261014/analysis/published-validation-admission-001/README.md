# Published validation admission overlay

This external patch adds source-grade, protocol, complete-history, and unit gates to the cardboard calibration ensemble evaluator. It is prepared against the retained base copies in `base/`; it does not modify the repository checkout.

## BCT full-peak admission

A BCT peak is evaluated only for the published `without_offset` condition and only when the native certificate carries the new `numi.cardboard.native-bct-capacity-binding.v1` object identically in its history, certificate, input manifest, and receipt. The adapter verifies the exact `numi.cardboard.shipping-capacity-qualification.v2` key/value schema and SHA-256 links to:

- the accepted nominal force/stroke CSV and its sample-by-sample native history;
- the nominal and half-speed accepted compression branch ledger, including actual control-step counts and dt bounds;
- the two independently hashed three-level mesh and time reports;
- the assembled Matter checkpoint, rigid scene sidecar, compiled Matter package, native executable, and solver source.

It recomputes the 70%-inside-height endpoint, nominal peak sample index, nominal/half-speed rate, and the producer's peak/work/full-curve sensitivity gates. It inspects the mesh/time report pairs and requires complete endpoints, decreasing coarse-to-middle and middle-to-fine differences, and the existing peak/curve/work limits. A caller-supplied protocol ID or matching `expected_steps` cannot substitute for these artifacts.

The current native shipping output does not yet emit this complete native-history binding object, and no retained v2 qualification artifact was found in the searched checkout/evidence roots. Therefore existing BCT full-peak cases are deliberately rejected as unsupported until a producer emits the bound artifact graph. This patch does not claim BCT material calibration or physical box validation. The published BC-780 and EB-965 rows remain cross-grade geometry/load-path holdouts.

The synthetic BCT files used by the CPU contract tests are generated in temporary directories and are explicitly test-only. They do not represent measurements or a qualified run.

## Validation

Run from the checkout whose source-data catalog is being evaluated:

```sh
NUMI_CARDBOARD_REPO_ROOT=/Users/home/numi-cardboard-20261006 \
  python3 /Users/home/cardboard-implementation-20261014/analysis/published-validation-admission-001/patched/cardboard_calibration_ensemble_test.py
```

The sealed run in `checks/python-tests.log` passed 27 tests. Python bytecode compilation also passed for both patched files. `overlay.patch` is the exact diff from the retained `base/` copies to `patched/`.

## Files and hashes

See `source-overlay.json` for exact base/candidate SHA-256 values, test receipt hash, and overlay hash. To integrate, copy only the two patched Python files to their matching `matter/tools/` paths. No GPU, native build, or production simulation was run for this evaluator-only overlay.
