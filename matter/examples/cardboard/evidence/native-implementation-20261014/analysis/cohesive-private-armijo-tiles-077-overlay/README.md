# Tiled cohesive private-Armijo overlay

This is an isolated overlay against frozen source 076. It does not modify the
shared checkout. It adds an opt-in ordered-page implementation for cohesive
trial-force evaluation in the full-contact private Armijo path. Default and
legacy cohesive paths remain unchanged.

## Source binding

Base snapshot:

`/Users/home/cardboard-implementation-20261014/snapshots/source-stable-free-graph-and-general-arithmetic-076`

Root-bound snapshot digest: `9b9047688d4da51739f97a35c0c0c24c57a38bcf4631088d4377ab9697c1d516`.

Base file hashes:

| File | SHA-256 |
|---|---|
| `matter/src/runtime.mm` | `5c70789647a9c0ceacaeb10565aaaee30e2a901761c1d04e6e431fdafe0aeef1` |
| `matter/src/metal/fem.metalinc` | `819f6724c2690c41c07972e6596624672000d35423bba3f1233e0063c47b9564` |
| `matter/include/numi/matter/shared.h` | `e1b4a229e58227280e2a1e263fa6535f325333e9edcacfd93498f699790c30d3` |
| `matter/include/numi/matter/matter.hpp` | `c3a8ee4c09c58c0891cda023c4b82ebd0e4036909007fa815e30180af8aee05a` |
| `matter/src/matter_kernels.metal` | `37d3530284e2361b20f805409768c65446fc13186c50e94f286b73256debe168` |

The C/Metal ABI is unchanged. Compiler-measured record sizes are
`sizeof(NMCohesiveFaceGPU) == 384` and
`sizeof(NMCohesiveFaceVectorGPU) == 96`; the legacy private clone-plus-force
arena is therefore 480 bytes per face per environment. The tiled arena stores
only force records and is capped at 8 MiB including the environment stride.

## Runtime behavior

Set `NM_FEM_FULL_CONTACT_TILED_COHESIVE=1` together with the existing
`NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT` selector. The new mode rejects
conflicting residual-Armijo selectors and cohesive scratch aliasing. It
processes every cohesive face in ascending face-index pages. Each page runs
the same production force/history kernel in read-only trial mode; the page
gather visits the original per-node incidence entries in their original order
and applies the completed force once using the existing residual/reaction
formula. Accepted face histories are not written by a trial. A failure or
rejection remains under the existing transaction rollback.

`NM_FEM_FULL_CONTACT_TILED_COHESIVE_PAGE_FACES` is an optional positive
decimal page-size override, accepted only while tiled mode is enabled and
only when the resulting page fits the same 8 MiB bound. The native parity
fixture uses a value of `1` to force three pages from a three-triangle
partition of one mortar patch; this is a test control, not an alternate
physical model.

## CPU check

The layout and ordered-float-gather check was compiled with strict warnings
and passed 61 checks:

```sh
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror \
  -Imatter/tools -Imatter/include \
  matter/tools/cardboard_cohesive_armijo_tiles_check.cpp \
  -o /tmp/cardboard_cohesive_armijo_tiles_check
/tmp/cardboard_cohesive_armijo_tiles_check
```

The check covers one/multiple environments, 260,835-face page plans,
byte/dispatch/face-slot encoding limits, exact 384+96 byte accounting, no
truncation, and bitwise ordered FP32 gather parity with duplicate incidence
entries and cancellation values. It rejects an incidence list that is not in
the production canonical order. This is a CPU layout/arithmetic check, not a
native or physical qualification.

## Prepared native check

The overlay adds `--cohesive-tiled-parity` to
`cardboard-full-contact-armijo-check`. The command first confirms that a
zero-face tiled request fails closed. It then compares legacy scratch and
one-face pages for one-face and three-face synthetic partial-mortar cases.
Both cases use accepted activation, temporary grips for accepted closure,
nonuniform mixed-mode opening/shear, and a rejected post-commit transaction.
The three-face case forces three ordered pages. The runtime log must report
the expected page count and zero face-state clone bytes. Snapshots are
compared bitwise, including FEM nodes/fields, cohesive history, material
history, rigid/contact state, statuses, and certificates. The test also
requires the native private full-contact Armijo path to execute with a
nonzero free-node response. It reports actual native FEM transaction count
and labels the law synthetic and physical validation false.

The prepared native target is
`numi-matter-cardboard-full-contact-armijo-check`; invoke it with the existing
bulk-material argument and `--cohesive-tiled-parity`. The integration owner
still needs to register the helper CPU target and native option in CMake. No
native GPU execution or native parity result is claimed by this overlay.
Full shipping-scale admission remains unqualified.
