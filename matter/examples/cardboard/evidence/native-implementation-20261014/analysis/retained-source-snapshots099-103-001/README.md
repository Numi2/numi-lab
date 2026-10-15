# Lossless source restoration for native runs099–103

Each manifest reconstructs the exact selected sources from its original build.
For099,100,101 and103 it also retains the verified root-CMake support files.
Snapshot102 deliberately remains the thin snapshot whose configure failed.
It is not repaired during restoration.

Unchanged bytes come from immutable Git revision
`6ebaf0fabcd0f824260b03994aaa13d7cab41c24`. Every other byte is stored in a
content-addressed gzip blob. Both compressed and restored hashes are checked.
All five manifests were reconstructed byte-for-byte;103 was also restored to
disk in full. The original `frozen-source.json` files remain unmodified.

From this package, use a Python3.10+ interpreter:

```sh
python3 restore.py --repository /path/to/numi-lab --number 103 --output /new/source103
```

The output directory must not exist. This restores source, not build binaries
or an accepted simulation checkpoint. Original source-bound commands and
artifact hashes remain in the corresponding run/build receipts. A new build
must receive its own build provenance and new run output directory; do not
rewrite an old receipt to refer to a relocated build.

The restored sources include known failed revisions. Their numerical and
physical verdicts remain unchanged.
