# Left ankle candidate 021 on main 294

This folder freezes a proposal-only left ankle skin candidate transplanted onto the exact main 294 NHSKIN source. It is separate from active headers and runtime assets.

## Reproduce

On `numi`, run:

```sh
/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/ankle-coupled-021-on-294-001/native-compileworld-candidate-001/run_cpu_checks.sh
```

This rebuilds the common-mode O/I/D layer candidate from hash-pinned inputs, rechecks donor transport, regenerates private `.nmatter` files and the reduction header, compiles the existing `appendRegisteredSkinFatFEMFromLayers` owner harness with `-Werror`, and runs FromLayers/compileWorld on CPU. It uses link-033 and the clean main-294 app headers. It performs no simulation and changes no admission state.

## Results

- Final NHSKIN SHA256: `d07b27e7d9d6952a3bc33d41535693b45f107f629e69520c2afbfb990ed1237f`.
- Patch preserves source topology and indices; 73 source rows and 195 faces changed; 49 boundary edges; zero local orientation errors.
- Static Float32 O/I/D boundaries have zero self or pairwise intersections; all layer prism tets are positive. Existing donor mass debits are retained with in-range resolved densities: skin 999.745832 kg/m3, fat 899.743504 kg/m3. Compiler result: 6,996 nodes, 27,714 positive tets, min J/reference 0.9999999411.
- Exact pose audit: matched reference, left dorsiflexion, left plantarflexion. Changed-patch self, remaining-skin, and selected ankle-bone intersection counts are zero. Separate signed-winding receipt has 251 outside-bone point queries for the underlying 021 source candidate; this is not a complete final-payload or swept-volume test.
- In same-pose donor-transport comparison, coarse row 1759 improved from 7.38 mm to 5.04 mm; row 1773 changed from 7.04 mm to 7.16 mm. About 7 mm of dorsiflexion/plantarflexion mismatch is also present on unmodified 294 and largely reflects source LBS versus rigid donor transport.

## Limits

This is a bounded static preparation increment, not an accepted physical repair. It has not passed dynamic FEM/contact tests, continuous collision checks, full surrounding-bone clearance for deformed layers, or active-scene admission. FromLayers/compileWorld validates construction only: GPU=false, steps=0, layers_admitted=false. Do not copy generated headers into active app source.

The manifest generator hashes all files in this folder and the external source/build artifacts used for reproduction. Run:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/ankle-coupled-021-on-294-001/freeze_bundle_manifest.py
```
