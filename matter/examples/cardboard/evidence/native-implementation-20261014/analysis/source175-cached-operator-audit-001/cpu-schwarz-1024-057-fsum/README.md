# Source-175 CPU Schwarz size sweep

This is a bounded CPU-only sensitivity study of the exact source-bound matrix
and right-hand side captured at root 3 / Newton iteration 23 in source175. That
Newton state was rejected; these results do not describe an accepted physical
step. The study did not read or compare a native RAP, and it is not a native
Matter solver or physical qualification.

The runner used Python 3.10.12, NumPy 2.2.5 and SciPy 1.15.3 on ARM64 macOS.
It checked the source report, prediction, receipt, capture and three operator
sidecars by their bound hashes before constructing the same nonsymmetric A and
captured RHS. It uses restart 16, at most 1024 GMRES iterations, relative true
residual tolerance `1e-8`, and the reviewed rowwise `math.fsum` accumulation.

The exact invocation was:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu-schwarz-1024-057-fsum/study.py \
  > /Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu-schwarz-1024-057-fsum/stdout.log \
  2> /Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu-schwarz-1024-057-fsum/stderr.log
```

The runner has a 120-second alarm, a preflight estimate capped at 1 GiB, and
appends case-start/completion/failure events to `events.jsonl`, so a timeout
keeps completed cases. It writes exact deterministic patch memberships. For a
repeat, copy the runner into a fresh sibling directory because it refuses to
overwrite an existing event ledger.

| Maximum nodes per patch | Anchor stride | Patches | GMRES applications | Final true relative residual | Status |
|---:|---:|---:|---:|---:|---|
| 64 | 32 | 78 | 1024 | 3.013152144e-4 | Not converged |
| 128 | 64 | 52 | 1024 | 8.8310078199e-6 | Not converged |
| 256 | 128 | 13 | 383 | 9.5554497006e-9 | Converged at the requested CPU diagnostic tolerance |

All factorizations and pivot vectors were finite/valid, and deterministic local
solve relative residuals were between `7.68e-16` and `1.35e-15`. Minimum
relative U-pivot ratios were `8.14e-5`, `8.66e-5`, and `2.49e-4` for 64, 128,
and 256 nodes. One-norm condition estimates for the three weakest-pivot patches
per size are included in `report.json`; the largest reported estimate was
`1.09e6` (256-node case). These are estimates, not rigorous condition bounds.

The largest estimated host peak was 116,018,972 bytes; observed process peak
RSS was 127,516,672 bytes. The run took 14.53 seconds, below both caps. No
RuntimeWarnings occurred in this fsum run. The earlier matmul-diagnostic run is
retained separately at `../cpu-schwarz-1024-057/`; its warnings were not
silenced or overwritten.

Primary result and complete bindings: `report.json`. Per-case factor and true
residual trajectories: `events.jsonl`. Memberships: `patch-membership-64.json`,
`patch-membership-128.json`, and `patch-membership-256.json`.

SHA-256 commands:

```sh
shasum -a 256 study.py report.json events.jsonl patch-membership-64.json patch-membership-128.json patch-membership-256.json
```

At creation, the hashes were:

```text
study.py                 c7d01a83a23422d6a357621f85fec0bdd810ebda8916bbbcab02f2ecd79d54c5
report.json              d12035734ecef9d3b7076ba79ad06a4548d9af910188f17c36c0a622aac214f3
events.jsonl             9e6ff65886af4abc90f45729272e3841a7a40b2e27b04fb233e9f7b8bca64fb0
```

Membership hashes are recorded in `report.json` and each case event.
