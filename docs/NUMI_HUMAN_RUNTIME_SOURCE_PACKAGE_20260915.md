# Numi Human native runtime source package

## Immutable release tuple

`human-native-runtime-package-v3-20260915` is the current release tag on the
public `human-native-runtime-20260915` branch. It pins the complete bounded
native-runtime tuple below, including the read-only constraint-impulse trace
instrumentation and its retained receipts. `human-native-runtime-package-v2-20260915`
and `human-native-runtime-package-20260915` remain earlier immutable package
receipts.

| Component | Immutable identity |
| --- | --- |
| Native runtime source | `Numi2/numi-lab`, tag `human-constraint-impulse-trace-source-20260915`, commit `826b3029a9f8372b564f0729917cbd20e680b419` |
| Source and evidence receipt | `Numi2/numi-lab`, tag `human-constraint-impulse-trace-receipt-20260915`, commit `0fd1d8166b84087cecb13920b50a4b306b49743a` |
| Exact five-file runtime input package | `Numi2/numilab-human`, tag `human-native-runtime-source-inputs-20260915`, commit `64b96d0327f3fda16f85e240d4e8720d3f3ff683` |
| Input package receipt | `Docs/media/native-runtime-source-package-20260915/source-package-receipt-v1.json`, SHA-256 `09f3d26cdf9adae8bead35d15e0c4a4f0021f08fe39096d1ac14bfb6c7bd1757` |
| Local source-bound full-body executable | Apple M4 `metalrobo_numilab_human_myosim_visual_probe`, SHA-256 `9dc70fe94a9531b9412b1bb08ebf57deb288a0e866c474186f1270359290ab9a` |
| Physical execution control | Apple M4 Pro focused bundle, documented in `NUMI_HUMAN_CONSTRAINT_IMPULSE_TRACE_20260915.md` |

The runtime source tag contains the accepted-static-fibre handoff,
stationary-fibre root continuity repair and focused test, the full
equality/near-boundary-limit active-set audit, and the new v6
constraint-impulse owner diagnostics. The receipt tag contains the raw
compressed 6.4 ms local trace and physical-M4-Pro execution-only focused
control. The owner input tag contains the actual NHRIGID2, NHMYO2, NHTEND3,
NHCNT1, and NHEQ1 files used by the bounded replay, their SHA-256 values, and
the retained input-package transcript. None of these identities is a mutable
local-machine branch.

The earlier v2 source was rebuilt on the Apple M4 Pro and its reference probe
ran the retained 12.5 us trace at step 281 using the pinned core, equality,
and trace hashes. It reported 51 equality rows, 54 near-boundary limit rows,
numerical rank 81 of 105, and 41 retained equality plus 40 retained limit
directions. The v3 source adds owner-retaining diagnostics only: its local
6.4 ms trace records equality 43 / source-limit DOF 113 at the largest
acceleration and post-projection limit residual. See
`NUMI_HUMAN_CONSTRAINT_IMPULSE_TRACE_20260915.md` for the exact raw-artifact
hashes and the physical M4 Pro execution boundary. Neither audit is a
contact, force, integration, standing, or physiological qualification.

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
  --muscle-step-seconds 0.0000125 --muscle-step-count 512 \
  --muscle-activation 0.8 --persistent-metal-stand \
  --persistent-source-passive-joint-tissue --persistent-stand-trace \
  --stand-contact-iterations 64 --stand-deterministic-replay
```

The retained v3 local trace completes all 512 steps on an unassisted root,
includes all 416 recruited muscle records, has zero penetration, and has a
bitwise-equivalent segmented endpoint. Its peak kernel acceleration is
`1.2233552932739258 m/s2 or rad/s2`; its raw transcript SHA-256 is
`fe31eb64173d3479ee2b65e6f95c66199764d2f182680a9d3aaf2d884822a239`.
This is a local Apple M4 source-bound receipt, not a replacement for the
historical M4 Pro v2 receipt or a cross-hardware bitwise claim.

## Boundary

The source-delivery problem is closed: an independent user can obtain the
native source, receipt, and exact runtime inputs from pinned public tags. The
v3 local executable is not byte-identical to the historical retained binary,
so this is not a historical-executable reproduction claim. The current Mac
mini has insufficient free capacity for a new source build and was already
owned by a separate full-Human run when the v3 full-body execution was due;
the v3 physical M4 Pro evidence is therefore execution-only and focused.
No solver regularization, passive-stiffness tuning, or runtime mechanics was
introduced by this release. The common-duration temporal force-convergence
gate remains failed, and this release does not qualify standing, passive-force
physiology, recovery, walking, anatomy, or safety.
