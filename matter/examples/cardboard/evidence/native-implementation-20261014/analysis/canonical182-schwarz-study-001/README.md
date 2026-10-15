# Canonical source182 offline graph-Schwarz sensitivity

This CPU-only study reruns the prior restart-16, maximum-1024 graph-Schwarz size and local-factor precision experiments on the owner-verified source182 operator (A) and captured RHS (b). The selected generation is environment 0, root 3, Newton iteration 23. Its native run later exited 2 at Newton 63. This audit does not assert that this particular Newton23 trial was accepted as a physical step.

The source report is `analysis/selected-newton-regional-rap-audit-182/operator-audit.json` (SHA-256 `7fc18d560019f74f00e5cf0b82326494316e0e0a145db0a96db493c0d3d646db`). It binds the raw operator capture (`27345f7afc7f752388bc76802657386546195ed8b13694df9b39be43be3a8446`), prediction and receipt, unchanged production inputs, and selected Newton generation. The same-generation regional RAP sidecar was independently checked before running either study: manifest SHA `8a0c68399357d3edf513499864260cdda3e7a239d429941d95a544756eb84eda`, payload SHA `5ea011af4b6c0835cf88cf9f6005514b79a14a850ba16c2c4a66374e118cb4d4`, capture point `after_regional_rap_factor_barrier_before_fgmres_correction`, generation `17179869207`, candidate-linearization SHA for environment 0 `1c756428d7ef1f5cfa6a5e035faef068d39de9b38db6195b56009bbe3c581b41`. The RAP is verified as same-generation evidence but was not loaded into the CPU Schwarz experiments.

The assembled nonsymmetric CSR system is 2538 by 2538 with 71,046 nonzeros and captured RHS norm `4.407680504925044e-6 Ns`. The offline action parity recorded by the source audit is `1.18997e-6`. The free-node graph topology matches the earlier source175 topology exactly, and the generated 256-node/stride-128 patch membership matches that prior membership file byte-for-byte at the JSON-value level. All new membership files and hashes are retained under `size-final/`.

The size sweep used 64, 128, and 256-node maximum patches with anchor strides 32, 64, and 128, respectively. Restart 16, a 1024-iteration cap, true relative residual tolerance `1e-8`, rowwise `math.fsum` outer updates, and the same exact (A,b) were held fixed:

| Patch size | Outcome | Iterations | Final true relative residual | Minimum local U-pivot ratio |
|---:|---|---:|---:|---:|
| 64 | not converged | 1024 | `5.7672164174e-5` | `8.13655438e-5` |
| 128 | not converged | 1024 | `9.80607993e-6` | `8.66160108e-5` |
| 256 | converged to the CPU study tolerance | 395 | `9.7870927681e-9` | `2.49494753e-4` |

On the exact same 256-node patches and same (A,b), three local factor/solve modes were compared with FP64 outer GMRES. All three reached the CPU study tolerance; these are SciPy CPU emulations, not Metal measurements:

| Local factor / solve | Iterations | Final true relative residual | Maximum local solve residual against FP64 patch | Minimum pivot ratio |
|---|---:|---:|---:|---:|
| FP64 factor + FP64 solve | 395 | `9.7870927681e-9` | `1.2381e-15` | `2.49494753e-4` |
| FP32 refactor + FP32 solve | 398 | `9.8172770386e-9` | `6.6793e-7` | `2.49493110e-4` |
| FP64 factor rounded to FP32 + FP32 solve | 300 | `9.9534780485e-9` | `5.2771e-7` | `2.49494769e-4` |

The complete true-residual histories, patch-local diagnostics, exact memberships, warnings, and failure/nonconvergence statuses are retained in each `report.json` and `events.jsonl`. The completed runs reported no NumPy/SciPy runtime warnings. Both processes remained below the 120-second and 1-GiB caps; the authoritative runs below used 14.55 s / 132,235,264-byte peak RSS for the size sweep and 9.72 s / 137,379,840-byte peak RSS for precision sensitivity. The earlier first-pass outputs remain under `size-final/` and `precision-final/`; `size-verified/` and `precision-verified/` are authoritative because their reports identify the event-ledger hash as the prefix before the final terminal record, while the exact full ledger hash is recorded here.

## Reproduce

Use Python 3.10.12 with NumPy 2.2.5 and SciPy 1.15.3. The scripts refuse to overwrite their event ledgers.

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/canonical182-schwarz-study-001/size-verified/study.py

/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/canonical182-schwarz-study-001/precision-verified/study.py
```

Final artifact hashes:

```text
size-verified/study.py             3beab0f589c73ae16339afabd2b63e5acc853897b660159029b205f555dbff3a
size-verified/report.json          b9301832b3d1ce2e18a98ba407eef628f19c9bacc6304b54ba1bf7e9e6188356
size-verified/events.jsonl         e2fe8f8d5dad4127d9f8436131ba5d203202c4e4c92e567f48284d3be7f8247b
size-verified/stdout.log           c2f3887667b0f5e270beed076c0b0f681583bb15b3bcb1c1fa350ee9508efa9d
size-verified/stderr.log           e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
size-verified/patch-membership-64.json   a74c9790a6468e1b75fe1333dfe02636882e0ebba4177252d409095c89e0ca6a
size-verified/patch-membership-128.json  42a0e4409134d0ed70240adbde9e32df6fe066f3b0164c67779db920413fc52c
size-verified/patch-membership-256.json  9d96255027ffcfb230aec3c1a9f302929156ceb0432af2c4783bcd26dd3dc184
precision-verified/study.py        5d8a258b1c0b916d0c39f02a66e0b78ce6c0c6d4bab510eb7db07249dbd03b47
precision-verified/report.json     9922ee292d3246775a1649b27a8960521e3cf51d8e062450aaddb7448c551cc9
precision-verified/events.jsonl    1fc99ecff1d10de4d05d8c5293493dedca666a8de233e23165679fd8ed66be4c
precision-verified/stdout.log      b04d6dedf94f43e468970ff086cfe58b78f7202a03705de86ff0b56146038ead
precision-verified/stderr.log      e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

These offline CPU experiments do not qualify the production Matter preconditioner, native FGMRES, material calibration, or physical behavior. They do show that on this particular captured operator, the 256-node graph-Schwarz CPU diagnostic reaches the selected tolerance while 64/128-node versions do not; the tested FP32 local factor/solve emulations also reach it with the same FP64 outer iteration and true-residual test.
