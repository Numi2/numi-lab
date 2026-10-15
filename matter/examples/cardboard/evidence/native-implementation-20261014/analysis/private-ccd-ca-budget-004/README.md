# Captured private-CCD near-miss replay and one-step budget candidate

This package records an external, CPU-only diagnostic against the frozen source 099 inputs. It does not edit or certify source 099, and no native GPU execution was performed for this candidate.

## Bound evidence

Frozen source: `/Users/home/cardboard-implementation-20261014/snapshots/source-center-action-twoenv-restart-099`.

The captured contact implementation is `matter/src/metal/contact.metalinc` (SHA-256 `3cef6ae3dbd384d7b26ff6b0eb4ecda6da7a32be74cdea0badc5975eef692132`). Native captures are:

- Case 333: root 1201, Newton 12, trial 20; payload SHA-256 `8fe7d2943880bb9962b1745569fa4120d88a317fee43b00fbbe9d609cc5d07eb`.
- Case 334: root 1201, Newton 12, trial 21; payload SHA-256 `bb645f01802e2d0cfcec1a15ec0769ff988e8f06b1ce4a58610bc82c8c7f6e38`.

Both captures identify active candidate slot 331, primitive identities 88 and 127, topology-active bit 1. Their triangle node IDs are `[13,14,17]` and `[28,32,31]`; the failing edge-edge feature `(edgeA=0, edgeB=0)` uses nodes `[13,14]` against `[28,32]`. The native diagnostic vector `[88,127,0,0]` is ambiguous because it is also used for vertex-triangle vertex 0. The CPU source-order replay checks all six earlier vertex-triangle directions first and finds them certified, then encounters EE(0,0) as the first uncertified feature. This is a source-equation localization, not a native feature-kind trace.

The object timestep is approximately `1e-4 s`; authored contact thickness is approximately `1e-5 m`. The frozen `ccdDistanceRoundoff` formula adds approximately `1.907 nm`, giving effective thickness `10.001907 µm`. A double-precision 10,001-point sweep over the captured FP32 endpoints finds the minimum segment distance at the end of the interval, approximately `10.003240 µm`, a positive clearance of approximately `1.333 nm`. The scan is descriptive rather than a proof; the conservative-advancement step-over-endpoint is the source algorithm's miss certificate.

## Replay result and proposed change

`replay-report.json` preserves two FP32 evaluations of the affine interpolation:

- `x + t * (y - x)` exhausts the frozen 16-iteration EE budget at `t≈0.9999702`; the next conservative step is beyond the endpoint and certifies a miss at iteration 17.
- The weighted interpolation form `x*(1-t) + y*t` certifies the same near miss within 16 iterations in this CPU emulation.

This sensitivity means the replay is not bit-exact Metal evidence. It shows the candidate is a grazing near miss and that one additional conservative-advancement check resolves it for the source-order FP32 replay without changing the collision thickness, roundoff, gap, acceptance, or hit rules.

`contact-16-to-17.patch` changes only the `edgeEdgeCCD` loop bound from 16 to 17. The vertex-triangle budget and all CCD formulas remain unchanged. The companion `contact.candidate.metalinc` is a regular-file copy of the resulting source for diff review. This is a supplemental candidate only; it needs integrated Metal compilation and a fresh source-bound native rerun before adoption.

Controls use the same equations and effective thickness. A synthetic 60 nm translation of the second edge creates an initial true hit, recognized at time zero. A 50 nm translation creates a swept hit detected by the 32-step replay after 25–29 checks; at the proposed 17-step cap it remains uncertified and fail-closed, never accepted as a miss. These translated cases are CPU controls, not native fixtures.

## Reproduce

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/private-ccd-ca-budget-004/replay.py \
  --capture /Users/home/cardboard-implementation-20261014/native/center-action-feasibility-bracket-333/private-trial.coupled.json \
  --capture /Users/home/cardboard-implementation-20261014/native/center-action-feasibility-bracket-334/private-trial.coupled.json \
  --output /Users/home/cardboard-implementation-20261014/analysis/private-ccd-ca-budget-004/replay-report.json
python3 -m unittest discover \
  -s /Users/home/cardboard-implementation-20261014/analysis/private-ccd-ca-budget-004 \
  -p 'test_*.py' -v
```

The reader dependency is the source-bound `analysis/private-trial-reader-001/trial_capture_reader.py`. Tests validate both capture payload hashes, the source identity and topology, source-order feature outcomes, the unchanged thickness, hit/miss controls, patch scope, and patch application to the exact frozen source.

## Qualification boundary

This package is a CPU source-equation replay and an uncompiled source patch. It does not establish Metal instruction-level parity, a native CCD pass, a full-contact Newton convergence result, or physical validation. Preserve the original 333/334 failures. The next required evidence is an integrated build and a native rerun with the one-step EE budget candidate, plus existing conservative CCD regression cases.
