# Numi Human native runtime source package

## Immutable release tuple

`human-native-runtime-package-v2-20260915` is the current documentation-only
release tag on the public `human-native-runtime-20260915` branch. It pins the
complete bounded native-runtime tuple below without changing the runtime
mechanics. `human-native-runtime-package-20260915` remains the earlier,
immutable package receipt.

| Component | Immutable identity |
| --- | --- |
| Native runtime source | `Numi2/numi-lab`, tag `human-equality-limit-active-set-audit-v2-20260915`, commit `c8d312ded319b1416207cd4f071c06e476dda202` |
| Exact five-file runtime input package | `Numi2/numilab-human`, tag `human-native-runtime-source-inputs-20260915`, commit `64b96d0327f3fda16f85e240d4e8720d3f3ff683` |
| Input package receipt | `Docs/media/native-runtime-source-package-20260915/source-package-receipt-v1.json`, SHA-256 `09f3d26cdf9adae8bead35d15e0c4a4f0021f08fe39096d1ac14bfb6c7bd1757` |
| Fresh public-source executable | Apple M4 Pro `metalrobo_numilab_human_myosim_visual_probe`, SHA-256 `b85669fefaf414a28a9eb44531985dcd83fe549eb5b8d6c7747773bedb3e15bc` |

The runtime tag contains the accepted-static-fibre handoff, stationary-fibre
root continuity repair, its focused test, and the read-only full
equality/near-boundary-limit active-set audit. The owner input tag contains
the actual NHRIGID2, NHMYO2, NHTEND3, NHCNT1, and NHEQ1 files used by the
bounded public-tag replay, their SHA-256 values, and the retained replay
transcript. Neither tag is a mutable local-machine branch.

The v2 source was rebuilt on the Apple M4 Pro and its reference probe ran the
retained 12.5 us trace at step 281 using the pinned core, equality, and trace
hashes. It reported 51 equality rows, 54 near-boundary limit rows, numerical
rank 81 of 105, and 41 retained equality plus 40 retained limit directions.
The audit intentionally reports no dependent-row identity because tied
near-zero pivots may choose equivalent rows on different hosts. This is a
read-only diagnostic receipt, not a contact, force, integration, standing, or
physiological qualification.

## Rebuild and replay

Build the public runtime source with an explicit CMake generator, then invoke
the visual probe against the companion Human input package:

```text
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target metalrobo_numilab_human_myosim_visual_probe

metalrobo_numilab_human_myosim_visual_probe \
  numilab-human/Docs/media/native-runtime-source-package-20260915/input/myosim-fullbody-core-reference.nhrigid \
  numilab-human/Docs/media/native-runtime-source-package-20260915/input/myosim-fullbody-muscle-reference.nhmyo OUTPUT_DIRECTORY \
  --tendon-payload numilab-human/Docs/media/native-runtime-source-package-20260915/input/numi-human-tendon-attachments.nhtendon \
  --support-contact-payload numilab-human/Docs/media/native-runtime-source-package-20260915/input/myosim-fullbody-support-contact.nhcnt \
  --joint-equality-payload numilab-human/Docs/media/native-runtime-source-package-20260915/input/myosim-fullbody-joint-equalities.nheq \
  --muscle-step-seconds 0.0000125 --muscle-step-count 64 \
  --muscle-activation 0.8 --persistent-metal-stand \
  --persistent-source-passive-joint-tissue --persistent-stand-trace \
  --stand-contact-iterations 64 --stand-deterministic-replay
```

The retained fresh public-tag replay completes all 64 steps on an unassisted
root, includes all 416 recruited muscle records, has zero penetration, and
is bitwise deterministic. Its maximum acceleration is
`0.115790568292 m/s2`. The transcript SHA-256 is
`bfb1888bf7dd73a3a94f527cd6ea7fcf3dd55eba85861417b6de1dd5134feac8`.

## Boundary

The source-delivery problem is closed: an independent user can obtain the
native source and its exact runtime inputs from pinned public tags. The fresh
public-source binary is not byte-identical to the historical retained binary,
so this is not a historical-executable reproduction claim. No solver
regularization, passive-stiffness tuning, or runtime mechanics was introduced
by this release. The common-duration temporal force-convergence gate remains
failed, and this release does not qualify standing, passive-force physiology,
recovery, walking, anatomy, or safety.
