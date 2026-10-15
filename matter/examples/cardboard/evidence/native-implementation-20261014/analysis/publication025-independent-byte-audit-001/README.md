# Publication 025 independent byte audit

This is a read-only audit of publication 025. It verified the current summary against the sealed binding, compared all 1,342 historical records against `summary-reference024.json`, and checked all 1,756 payload files in the 56 appended cases against both their external source bytes and their archived representation. Raw files were hashed directly; gzip files were checked for compressed size/hash and decompressed size/hash. The cumulative NUL pathspec was checked for an exact 2,312-entry byte prefix and an exact 1,758-entry suffix (1,756 payloads plus the two publication controls).

The audit did not modify the repository, publication archive, pathspecs, or source evidence. It makes no physical-validation or solver-qualification claim. The unrelated untracked `CompilerId` directory was left untouched.

Run:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 /Users/home/cardboard-implementation-20261014/analysis/publication025-independent-byte-audit-001/audit.py
```

The result is recorded in `report.json`. Input binding SHA-256: `908a12e88c411c8240d4ab4c0f220976f6e4cff7d0d939b3bb83bf81406c66fd`.
