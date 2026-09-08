# Metal 4 native owner replay regression

The exact NumiBrain `numilab-owner-replay` workload was moved from the Apple
Paravirtual hosted runner to a Mac16,11 Mac mini with Apple M4 Pro, macOS 26.6
(25G72), Apple9 and Metal4 API support. At the original owner revision
`4a369ca846fde93016f3708f3fd9386c992b52a4`, all 109 production pipelines constructed,
including all 29 that failed on the hosted runner. No shader source or pipeline
inventory change was required.

Native replay exposed two independent owner defects:

- External actions bound the unallocated policy-latent stream to task action
  application. G1's persistent raw-action tail then contained stale data.
  Bind the submitted action stream when there is no native PolicyPack; preserve
  native latent binding when a PolicyPack exists.
- The multicopter identity hashed the entire C++ object, including four ABI
  padding bytes before `windVelocity`. Separate X500 runs had identical GPU
  buffers and authored program fields but different padding at byte 204.
  Hash all seven authored members individually, including the complete GPU
  model/rotor/mixer records, and exclude only host object padding.

The complete persistent-state digest remains in the replay gate. Neither fix
changes contact, CCD, constraints, solver selection, physics or shader warnings.

Build and run the owner regression on the reference Apple machine:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target metalrobo_task_owner_replay_probe -j 4
build/bin/metalrobo_task_owner_replay_probe build/shaders/MetalRobo.metallib
```

It poisons both copies of multicopter padding differently, checks sensitivity to
all seven authored members, and advances Franka/G1/X500 with one and two
parallel environments through original/replay/changed-action sequences. It
requires accepted steps, zero reported GPU failures, finite q, exact resident
and physical replay, stable idle inspection, and changed-command divergence.
The Franka fixture retains active contacts. G1/X500's four-step fixtures do not
establish standing or flight.

NumiBrain owns the full hardware admission and exact 109-pipeline inventory
command, `tools/qualify_numilab_reference.py`, plus the unchanged bridge replay.
Qualification must use a fresh build directory and process, record exact source
and binary identities, and preserve every failure. A fresh process does not
imply the operating system's driver shader cache has been purged.
