# Source 103 EE CCD continuation adjudication (CPU-only)

This package compares native diagnostic runs 367 (source-103 baseline, EE loop bound 17) and 372 (one-line EE continuation overlay, bound 33). It does not modify either prior package or production source.

Run the verifier:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-continuation-372-adjudication-001/adjudicate.py \
  --baseline /Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-native-367 \
  --continued /Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-native-continuation-372 \
  --output /Users/home/cardboard-implementation-20261014/analysis/source103-pair-ccd-continuation-372-adjudication-001/rerun-report.json
```

The verifier validates both native bundle manifests and receipts, exact case identities/input words, bitwise production/trace clone parity, and the complete result/trace change set. The committed `report.json` is its output.

## Result

All 29 cases certified by run 367 have exactly identical production-result JSON and conservative-advancement traces in run 372. Test inputs match. The sole uncertified ray, `trial358_ee_edge0_edge0` (pair 92/127, candidate 283, root 1201/Newton 7/trial 23), continues from iteration 16 to iteration 17 and reports a certified hit. The new evaluation is at `t=0.9583277702331543`, separation `1.000184056465514e-5 m`.

Nominal thickness is `9.999999747378752e-6 m`. The existing distance-roundoff expansion is approximately `1.90735e-9 m`; the returned distance is approximately `1.84082e-9 m` above nominal thickness and therefore inside the roundoff-expanded band. This is not a strict nominal-thickness crossing. The prefix through the prior terminal evaluation is exactly preserved.

## Coverage and adoption boundary

The continuation run contains 30 direct diagnostic rays and zero physical transactions. It establishes that this captured ray obtains a terminal hit certificate with the additional sample, not that 33 iterations are sufficient for all EE queries or globally safe in full contact. The retained direct regression corpus exercises an EE endpoint-clear crossing and long-tangent certified miss, but no matching near-parallel exhaustion ray. Those native outputs were produced with source049's EE cap16. Source103's EE function matches that body apart from the loop-bound constant; this is useful inherited evidence, but it is not a same-build run of the 33-bound metallib. There is no full-contact 33-bound transaction in the 372 package.

Keep promotion pending same-build runs of both the direct CCD suite and a full-contact transaction, with the bound17 control retained. Budget exhaustion must remain uncertified/fail-closed.
