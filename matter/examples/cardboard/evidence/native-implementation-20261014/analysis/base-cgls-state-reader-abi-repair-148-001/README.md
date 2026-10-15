# CGLS state-capture reader ABI repair (148)

This package repairs the **reader only** from the immutable source-144 state-capture overlay. The old reader is retained unchanged at `analysis/base-cgls-state-capture-overlay-144-001/tools/read_state_capture.py`; its run against native capture 455 is preserved in `legacy-reader-455-failure.json`. It rejected `record:NMTetrahedronGPU` because it expected 112 bytes. Frozen source-144 `shared.h` defines six 16-byte fields, so the actual record is 96 bytes. The native writer uses `sizeof(NMTetrahedronGPU)` and wrote 96-byte records correctly.

`tests/test_shared_abi.py` compiles the actual source-144 `shared.h` from `snapshots/source-cgls-state-sidecar-144` with the repository's C++ ABI definitions. It compares compiled record widths to the reader table and checks the member offsets consumed by the record parser. The source header SHA-256 is `cd173c076905f660aa8aecc900e6f6aba8d69406996f3cccf2e0656c753ff871`. This test uses the real header, not duplicated test constants as its ABI source.

Validation commands, run from `/Users/home/cardboard-implementation-20261014`:

```sh
python3 -m unittest discover -s analysis/base-cgls-state-reader-abi-repair-148-001/tests -v
python3 analysis/base-cgls-state-reader-abi-repair-148-001/tools/read_state_capture.py solver/cgls-state-sidecar-observer-455/base-cgls.bin.state.json
python3 analysis/base-cgls-state-reader-abi-repair-148-001/tools/audit_capture_against_419.py --output analysis/base-cgls-state-reader-abi-repair-148-001/capture-455-vs-419.json
python3 analysis/base-cgls-state-reader-abi-repair-148-001/tools/audit_observer_parity.py --output analysis/base-cgls-state-reader-abi-repair-148-001/observer-parity-453-455.json
python3 analysis/base-cgls-state-reader-abi-repair-148-001/tools/collect_inputs.py
python3 analysis/base-cgls-state-reader-abi-repair-148-001/tools/seal_package.py
```

The 10 CPU tests pass. The strict reader validates capture 455: 24 arrays, 4,327,464 payload bytes, SHA-256 `1a99fa72a4f45c68c67a53dcace37f0f01df6587d6db1a178606247a9fdef5b6`, root 1/Newton 48, 2 environments, 882 nodes and 1,920 tetrahedra. No source, native input, payload, or production file was changed.

The exact state join with the independent 419 operator capture finds all 1,764 accepted nodes and all 1,764 captured center nodes byte-identical after decimal-to-FP32 conversion. All 3,840 accepted per-tetrahedron histories match `center.accepted_state`; all 3,840 cached tangents and validity flags match; all 118 environment parameter words match; all 1,764 Newton correction vectors match; and all 7,056 mechanical RHS float words match the 419 physical center residual with the same sign. The field correction/RHS backing blocks are all zero. Tetrahedron records have 956/3,840 exact-byte matches per arena; the other 2,884 differ only in signed-zero bits in float fields (32 words in inverse-rest row 0, 96 in row 1, 452 in row 2, and 2,304 in material-frame rotation). After canonicalizing only signed zero, all 3,840 records match; there are no other word differences. Accepted and candidate topology arrays are themselves byte-identical.

The raw candidate material-state arena is byte-identical to accepted history. The writer marks current-Newton projection as `unavailable_not_persisted`, because cache construction projects into local temporaries and stores tangent columns, while the candidate arena is populated later. This audit does not relabel that buffer as projected state.

Observer parity compares source-143 run 453 with source-144 run 455. The 409,368-byte CGLS payload and 13,568-byte status-5 payload are exact matches; 4,096 true-residual cycles, 112 Armijo rows, and 64 CGLS direction rows match exactly. All output files other than `output/manifest.json` are byte-identical; that manifest differs only in the three absolute material paths, whose basenames remain identical. The stderr streams match after excluding the six timing rows, removing the one new state-sidecar marker, and normalizing run/build paths. Timing values and source/runtime/binary fingerprints are retained as separate metadata. Both runs exited 2 with unchanged inputs and no timeout; this is attempted-history parity for a failed coupon, not a completed solve.

## Boundaries

- The source-144 reader failure is a consumer ABI-width bug; it is not a native writer defect.
- The 419 join compares captured state and operator inputs only. It does not establish CGLS convergence or validate the material projection algorithm.
- The projected material history at the selected Newton state remains unavailable.
- The observer pair is source-143/source-144 diagnostic parity, not solver or physical qualification.
