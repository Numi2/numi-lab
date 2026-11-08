# Executed Git recovery: implicit motor diagnosis

`verification.json` records successful execution from actual Git-restored native
inputs. Calibration stdout and all three TSVs are byte-exact: the production
96-condition diagnosis, the original zero-step IR metadata rejection, and the
corrected 144-condition alternative. Dylib load lines identify the restored
runtime. The bounded study's failed hybrid scientific gates remain failed;
reproducibility does not change that verdict.

All 43 original files are mapped by
`../crow-rl-implicit-drive-audit-20261108/payload/manifest.json`, SHA256
`04cc8fa363c16b76edc722bf9e6cb0cc7e6d7931aa060b59f2aec22aca21ad97`.
Actual Git objects, including reused blobs in earlier archives, supply every
byte. Omitted checkout copies are not deletions from Git history.

The historical `recover.py` sealed and executed this bounded recovery once;
it commits its plan and expects a clean, fresh workspace. It is retained as
execution provenance, not a general command to rerun against populated paths.
For fresh recovery, materialize the manifest at its exact tracked path with
`git show HEAD:artifacts/crow-rl-implicit-drive-audit-20261108/payload/manifest.json`,
then use the generic owner from the repository root:

```sh
python/.venv/bin/python tools/crow_evidence_archive.py verify artifacts/crow-rl-implicit-drive-audit-20261108/payload/manifest.json
python/.venv/bin/python tools/crow_evidence_archive.py restore artifacts/crow-rl-implicit-drive-audit-20261108/payload/manifest.json --output build/crow-implicit-drive-restored-new
```

The destination must be new. Paths inside it preserve original repository
relative paths. Set DYLD_LIBRARY_PATH to its restored
`.numi/retained-runtime/implicit-drive-audit-20261108` directory and pass that
directory's `metalrobo/MetalRobo.metallib` explicitly to each native probe.
`probe --calibrate` is the independent reference calibration. `probe` and
`bounded-v2-probe` take the shader and a new TSV path. `bounded-probe` deliberately
reproduces the original IR metadata rejection before physics; do not treat
its expected exit 1 as a new unexplained failure.

Exact path/hash/mode/mtime and open-handle inventories precede cleanup.
The 54-file temporary recovery tree (11,151,073 bytes) and 43 raw study files
(10,761,036 bytes) were verified and removed. The omission receipt covers
36 duplicate archive-checkout files (253,549 bytes). Recovery controls and
compressed replay logs stay visible; all payload remains reachable in Git.
The production build and protected deployment97 are preserved.
