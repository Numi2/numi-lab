# Source-103 edge-edge CCD continuation diagnostic

This isolated package evaluates a bounded continuation of one conservative-advance query. It is not a production-source edit and it executes no physical transaction.

## Finding from native diagnostic 367

The source-103 native diagnostic covered 30 captured queries. Direct production and traced-clone outputs were bitwise equal for all 30. The only uncertified result was `trial358_ee_edge0_edge0`, pair 92/127, candidate slot 283, at root 1201 / Newton 7 / line-search trial 23. It exhausted the existing 17 EE evaluations. All 12 VT queries and the other 17 EE queries certified. The adjacent EE(0,1) query certified a hit.

At exhaustion, the native EE(0,0) sample was 10.00195698 µm apart against an effective band of 10.00190696 µm, leaving about 0.050 nm. An independent FP64 closest-segment recurrence over endpoints reconstructed from the captured FP32 words reaches the effective band at evaluation 28. That calculation omits the production separating-axis certificate and uses FP64 arithmetic, so it is evidence for trying additional conservative evaluations, not a native or physical result.

## Candidate overlay

`continuation.patch` changes exactly one line in source-103 `edgeEdgeCCD`: the evaluation bound changes from 17 to 33. The speed bound, effective thickness, closest-distance calculation, separating-axis checks, minimum advance, and terminal behavior are unchanged. The native direct routine therefore can still return an uncertified result if evaluation 33 does not certify a hit or miss; exhaustion is never turned into a miss. VT remains at 16 evaluations. No thickness, geometry, tolerance, or timestep changes.

`source_overlay/matter/src/matter_kernels.metal` is byte-identical to the source-103 translation unit and exists so its quoted contact include resolves to the isolated overlay. The diagnostic clone has a 33-row EE trace buffer and independently checks its summary against the direct production routine.

## Reproduce CPU checks and compile

```sh
python3 audit_native367.py
python3 test_continuation_math.py
python3 test_package_contract.py
./build.sh
python3 verify_pair_package.py --prepared prepared/prepared.json --output cpu-prepared-audit.json
```

`build.sh` compiles the isolated Metal translation unit and Objective-C++ host but does not execute the host or dispatch a GPU kernel. The native command, if separately authorized and scheduled by the GPU owner, is:

```sh
./compile/pair_ccd_native_check prepared/prepared.json prepared/cases.bin compile/diagnostic_pair.metallib native-continuation-result.json
python3 verify_pair_package.py --prepared prepared/prepared.json --native-result native-continuation-result.json --output native-continuation-audit.json
```

The continuation run must remain a diagnostic. A passing native pair query does not establish full solver convergence or physical contact qualification.

## Binding and limits

`manifest.json` binds the frozen source-103 base, exact one-line contact overlay, prepared capture inputs, host, diagnostic kernel, tests, and compiled artifacts. `native367-review.json` binds the baseline native result and receipt. The tests include synthetic parser cases for 33-step EE exhaustion, rejection of a VT count above 16, direct/clone bit comparison, and an independent CPU recurrence boundary. These CPU fixtures do not replace the required native continuation dispatch.
