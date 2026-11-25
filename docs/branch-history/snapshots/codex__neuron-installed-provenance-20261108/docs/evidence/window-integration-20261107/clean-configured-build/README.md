# Fresh configured integration build

The independent detached worktree was pinned to `e4d80357` and deliberately
materialized source, CMake inputs, and the named test fixtures while excluding
large research evidence trees. A fresh Ninja build directory configured with
`BUILD_TESTING=ON` and built the shared library, shader library, Window and
rollout, trainer, catalog compiler, task-program check, measured-surface owner
probe, and actuator migration check. The exact configure and build logs,
source and binary hashes, sparse patterns and host tool versions are retained.
Both before/after source status files are empty, and repeated build reported no
work with unchanged recorded hashes. This is a bounded target build, not a full
repository build or proof that a default clone contains all runtime assets.
CMake reported that NumPy/SciPy Python tests were disabled.

Four CPU tests passed: actuator-pack migration, Window metadata, bootstrap and
training cancellation stubs. Separately linked fixture executables use the
freshly built library; they are not CMake target builds. On M6, six bounded
native checks passed: physics-owner and policy-owner digest guards, initial
policy-owner guards, measured-surface ownership, 768 delay-one values in a
mixed reset, and the kinematic rollback/reset/recontact fixture. Immediately
after reset the inactive manifold header and point bytes differed, while live
count and subsequent recontact matched. Full outputs, timestamps and hashes
are retained beside this file. No snapshot restoration, general rollback,
physical accuracy, perception, performance or release qualification follows.

The prior generated-Ninja diagnostic remains separately retained and is not
relabelled by this fresh build.
