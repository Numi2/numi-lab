# Exact diagnostic source116 and117 restoration

Both builds succeeded. The package reconstructs and has verified all 1,852 source/configuration files per snapshot against the fixed Git base and content-addressed compressed blobs. These are diagnostic builds and do not qualify a physical step.

Restore with `python3 restore.py --repository /path/to/repository --number 116 --output /new/source/path` (separate command options `--number 116`; use 117 for the production-helper deformation readback). Each output directory must be new. Manifests bind exact source-build receipts.
