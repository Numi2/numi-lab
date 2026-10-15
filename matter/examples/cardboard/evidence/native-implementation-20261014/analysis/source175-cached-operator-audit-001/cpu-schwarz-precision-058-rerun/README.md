# Source-175 local-factor precision sensitivity

This CPU-only experiment uses the exact source-bound A and b from source175,
root 3 / Newton 23. That capture is a rejected diagnostic Newton linearization,
not an accepted physical state. The same 256-node graph patch membership is
used for every case and is checked against the preceding source-bound size
sweep. Only local patch factorization and triangular solve/application
precision changes. Outer GMRES, A, b, node membership, restart, iteration cap,
and true-residual tolerance are held fixed.

The three local policies are:

1. FP64 factorization and FP64 triangular solves/application.
2. FP32-cast local matrix, FP32 refactorization, FP32 solve/application, then
   widen output into the unchanged FP64 outer GMRES.
3. FP64 factorization followed by rounding packed LU data to FP32, FP32
   triangular solve/application, then widen output into the same outer GMRES.

All use restart 16, at most 1024 iterations, true relative residual tolerance
`1e-8`, and rowwise `math.fsum` for the outer `Z@y` update. The bounded runs
used Python 3.10.12, NumPy 2.2.5, SciPy 1.15.3 on ARM64 macOS. Each factor
reports finite checks, pivot vector validity/count, relative U-pivot ratio,
exact-source local solve residual, plus condition estimates on the three
weakest-pivot patches. Condition estimates are iterative estimates, not bounds.

Results:

| Local policy | Final true relative residual | Applications | Factor/local checks |
|---|---:|---:|---|
| FP64 factor + solve | `9.5554497006e-9` | 383 | finite; min pivot ratio `2.4949e-4`; max local residual `1.35e-15` |
| FP32 refactor + solve | `8.5688145039e-9` | 411 | finite; min pivot ratio `2.4952e-4`; max local residual `6.50e-7` |
| FP64 LU rounded to FP32 + solve | `9.9090582565e-9` | 356 | finite; min pivot ratio `2.4949e-4`; max local residual `5.20e-7` |

The three weakest-pivot patch condition estimates were similar across policies:
the largest reported estimate was about `1.0933e6`. No factorization or GMRES
runtime warnings were reported. Estimated host peak was 116,018,972 bytes;
observed RSS was 131,497,984 bytes. All three runs completed in 9.52 seconds,
below the 120-second and 1-GiB caps.

These CPU results indicate that FP32 local factors/solves retained the
convergence benefit of the 256-node patches on this one matrix. They do not
establish Metal arithmetic equivalence, native Matter FGMRES behavior, RAP
parity, or physical qualification. No native RAP was read or compared.

The exact command was:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu-schwarz-precision-058-rerun/study.py \
  > /Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu-schwarz-precision-058-rerun/stdout.log \
  2> /Users/home/cardboard-implementation-20261014/analysis/source175-cached-operator-audit-001/cpu-schwarz-precision-058-rerun/stderr.log
```

For an exact rerun, copy `study.py` to a fresh sibling run directory. The runner
refuses to overwrite an existing `events.jsonl`. The first attempt is retained
separately at `../cpu-schwarz-precision-058/`; it stopped before any factor or
GMRES work because it called the shared 32-node-limited patch builder. The
rerun uses the previously checked extended traversal and verifies identical
256-node membership before testing precision.

Hash command:

```sh
shasum -a 256 study.py report.json events.jsonl
```

Hashes at creation:

```text
study.py     cf2ce825e78888ebc746d90394ebb3435fa54c1f143407e71f2b2b6f79ccb31f
report.json  15cc4e3cb5a3fa050dd42da82b3b99964ba430f265148594e68c279ab4594cd8
events.jsonl 2de1b233e62a3a22cf4f5f04abf8e2284e1325dd50e657242f6249a9bddebbf0
```
