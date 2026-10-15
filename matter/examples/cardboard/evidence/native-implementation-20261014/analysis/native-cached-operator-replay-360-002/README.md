# Source101 cached-versus-fresh operator replay — package 002

This package corrects the immutable tetrahedron audit in package 001 and adds
a CPU-only `--preflight-only` mode. It does not perform a native replay. The
preflight checks all 3,840 environment/tetrahedron records against the exact
compiled package before any `Device` or Metal pipeline is created.

The comparison is exact in value: connectivity and integer identity words are
bitwise equal; finite inverse-rest and material-frame floats must compare equal
with no tolerance. IEEE `+0.0` and `-0.0` are treated as the same numeric zero.
The capture/package differences found in the original native attempt were only
those signed-zero encodings: env0/tet0 frame component y, and env0/tet585
inverse-rest component 8. All other immutable fields match. A regression
sentinel verifies signed-zero equivalence and rejects a changed adjacent
nonzero float or a nonfinite value.

Candidate node positions, accepted material histories, the cached 9x9 tangents,
and the FGMRES correction remain the exact captured dynamic inputs in the
payload. They are not compared with rest tetrahedron fields. The capture,
compiled package, run metallib, payload, world fingerprint, and execution
fingerprint remain separately hash-bound. This avoids comparing mutable
candidate state as if it were immutable package topology and does not weaken
source or nonzero field identity.

## CPU checks

The arm64 checker was compiled against the source101 headers and provenance
libraries using `-Wall -Wextra -Wpedantic -Werror`. The exact command and file
hashes are in `package-manifest.json`. The following invocation validates
package/payload binding and every immutable tetrahedron field without creating
a Metal device:

```sh
build/cached_operator_replay \
  /Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360/output/compiled.nmatterpack \
  prepared/replay-input.bin \
  /tmp/cardboard-operator-preflight-only \
  --preflight-only
```

The result was:

```text
compiled package and captured immutable tetrahedron fields match for 3840 records; exact-float sentinel tests pass; no Metal device or pipeline was created
```

Package 001 and the failed native attempt 365 are preserved. The v2 CPU
preflight resolves the observed identity mismatch; it does not establish that
the cached and fresh production actions agree. A native replay still requires
the parent-owned GPU slot. No solver step or physical transaction is part of
this package.
