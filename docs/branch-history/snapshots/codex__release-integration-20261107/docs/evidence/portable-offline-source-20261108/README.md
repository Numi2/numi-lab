# Offline sparse-source package

`source-fixture-bundle.tar.gz` contains 1,973 materialized tracked source
files, tracked qualification helpers, and a hash-pinned synthetic Franka
fixture. `verify.py` in the archive verifies SHA-256, length, mode, missing
and extra files, and absence of `.git`. The manifest identifies source commit
`d9db7f9f30f75c9d54432fee2ad89d0e8f3ef26d`. It is a sparse source export,
not the full 119,164-path repository or an independently cloned object store.
The exact source list and file hashes are in `BUNDLE-MANIFEST.json`.

The archive was extracted to a fresh directory, verified, and its source was
configured and built into an empty Ninja directory under macOS `sandbox-exec`
with `(deny network*)`. A separate check showed a localhost socket changed
from connection refused (errno 61) outside the sandbox to permission denied
(errno 1) inside. Configure, build, and install exited zero. The initial
source-only archive was used for the build; all 1,973 source manifest entries
were compared with the final archive, which adds only qualification helpers
and fixture files. `build.sh`, `check.sh`, and their outputs are retained;
paths in these scripts describe the measured host and must be adapted when
reproducing. The installed `REVISION` correctly says `unknown` because there
is no Git checkout; do not interpret it as an internally authenticated commit.

The policy status CTest, 33 selector tests, installed Python producer / Swift
consumer tests, and WorldPack `--inspect` passed under the same network
denial. The first native non-G1 smoke exited zero but its wrapper failed:
the text-only helper was supplied where JSON was required. That failure is
retained in `failed/`. After adding the tracked JSON helper to the final
bundle, the independently pinned retry ran under network denial on M6 at
2026-11-08 00:14:28.010–00:14:31.910 UTC and passed: one environment, eight
steps, one update, matched selection, unchanged input hashes, no remaining
worker or lock, and a bound WorldPack hash. The non-humanoid guided screen
was unknown and refused auto-load; changed and missing fixture checks remained
unknown. The CPU producer/consumer test separately demonstrates a passing
synthetic status changing to unknown after input replacement.

This requires the host's macOS/Metal/Swift/CMake/Ninja toolchain and an
existing Python 3.13 MLX/NumPy environment. CMake disabled NumPy/SciPy tests
for its selected configure interpreter, and the default Window visual assets
remain absent. File hashes do not capture all system libraries, Python
packages, OS state, or simultaneous mutation. No physical, trained-policy,
visual UI, full-checkout, or performance qualification is claimed.
