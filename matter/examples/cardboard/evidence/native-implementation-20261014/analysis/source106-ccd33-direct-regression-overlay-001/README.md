# Source-106 EE CCD bound-33 overlay

This package changes the production edge-edge conservative-advance loop bound from 17 to 33 evaluations and adds one exact captured EE ray to the existing native CCD regression checker. The vertex-triangle bound remains 16. No contact equations, geometry, thickness, tolerances, or other solver code are changed.

The added ray is `trial358_ee_edge0_edge0` (primitive pair 92/127). The source-103 native record for run 367 is uncertified after evaluations 0–16 with the 17-evaluation bound. The source-103 continuation record for run 372 is certified on evaluation 17 with bound 33. The new checker runs this exact FP32 geometry through both the direct production CCD kernel and the Newton sweep wrapper. It checks that the returned separation falls between nominal thickness and the existing FP32 roundoff-expanded upper bound and that the reported TOI is in the captured terminal-time region.

This is a direct CCD regression, not a physical transaction. The captured hit is only a roundoff-band certification: the native returned separation is slightly above nominal thickness, while a separate sampled endpoint is sub-nanometer inside nominal thickness and is not a lower-bound proof. No robust nominal-thickness crossing is claimed. The previous VT diagnostic ray and all existing direct regression cases remain in place. A same-build full-contact run is still required before broader adoption.

## Source and artifacts

- Frozen base: `/Users/home/cardboard-implementation-20261014/snapshots/source-capture-callback-lifetime-106`
- Patch: `source106-ccd33.patch`
- Changed-source copies: `overlay/matter/src/metal/contact.metalinc` and `overlay/matter/tools/cardboard_contact_ccd_native_check.mm`
- Captured case and 367/372 bindings: `evidence/captured-ee-ray.json`
- CPU source/evidence contract test: `tests/check_contract.py`

The two source-103 runs are retained separately at:

- `/Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-native-367`
- `/Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-native-continuation-372`

The evidence file binds their result JSON hashes, input-corpus hash, source contact hash, prepared manifest, and metallib hashes. Those are historical source-103 comparisons; the new build is source-106 plus this overlay.

## Validation

CPU-only source/evidence contract check:

```sh
python3 tests/check_contract.py \
  --base /Users/home/cardboard-implementation-20261014/snapshots/source-capture-callback-lifetime-106 \
  --overlay /Users/home/cardboard-implementation-20261014/analysis/source106-ccd33-direct-regression-overlay-001/compile-source \
  --evidence /Users/home/cardboard-implementation-20261014/analysis/source106-ccd33-direct-regression-overlay-001/evidence/captured-ee-ray.json \
  --source103 /Users/home/cardboard-implementation-20261014/snapshots/source-whole-mortar-proof-103/matter/src/metal/contact.metalinc
```

Patch applicability check:

```sh
cd /Users/home/cardboard-implementation-20261014/snapshots/source-capture-callback-lifetime-106
patch -p1 --batch --dry-run < /Users/home/cardboard-implementation-20261014/analysis/source106-ccd33-direct-regression-overlay-001/source106-ccd33.patch
```

Compile-only build (standalone Matter CMake project; the frozen snapshot omits unrelated top-level Swift binding files required by the full repository configure):

```sh
cmake -S /Users/home/cardboard-implementation-20261014/analysis/source106-ccd33-direct-regression-overlay-001/compile-source/matter \
  -B /Users/home/cardboard-implementation-20261014/analysis/source106-ccd33-direct-regression-overlay-001/build-matter \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build /Users/home/cardboard-implementation-20261014/analysis/source106-ccd33-direct-regression-overlay-001/build-matter \
  --target numi-matter-cardboard-contact-ccd-native-check --clean-first -j2
```

The compile command successfully built the host checker and regenerated the Matter metallib. It emitted two existing warnings in `fgmres.metalinc` about threadgroup validity flags; neither file is changed by this overlay. The native checker executable was not run here. Root should run the direct CCD suite against the exact bound-33 metallib, then run the full-contact comparison against the same build while retaining the bound-17 control.
