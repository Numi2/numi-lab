# Frozen source transient-edit incident

A write was accidentally made through the overlay path `overlay/matter/src/metal/mixed_fem.metalinc` while that path was a symlink to the frozen source. The frozen file was restored immediately, the overlay symlink was replaced with an independent regular-file copy, and only the intended two-line diagnostic change remains in the overlay.

The file metadata gives a bounded local-time incident window on 2026-10-15 CEST: the overlay directory/link was present at 14:17, and the frozen file was restored at 14:24:22 (its post-restore mtime/ctime). The exact time of the transient write was not separately logged, so no narrower time is asserted.

The transient edit was exactly:

```diff
--- frozen mixed_fem.metalinc
+++ transient mixed_fem.metalinc
@@ -277,6 +277,7 @@
     device const float4* mechanicalDirection [[buffer(8)]],
     device const NMFEMNodeStateGPU* nodes [[buffer(9)]],
     device const NMFGMRESStateGPU* fgmresStates [[buffer(10)]],
+    constant uint& allowConvergedState [[buffer(11)]],
     uint global [[thread_position_in_grid]]
@@ -293,7 +294,8 @@
     if ((object.flags & NM_OBJECT_MULTIPHYSICS) == 0u ||
         object.materialIndex >= dispatch.mixedMaterialCount ||
         ((microstep.flags & NM_MICROSTEP_FGMRES_OPERATOR) != 0u &&
-         fgmresStates[environment].diagnostics.z > 0.5f)) return;
+         fgmresStates[environment].diagnostics.z > 0.5f &&
+         allowConvergedState == 0u)) return;
```

## Hashes

- Frozen file before the transient edit, as recorded by `frozen-source.json`: `5042632a1d44d12cabbc3d09fa6dc9c9f40ac51b3f81ab733ccedd95b58b4f0a`.
- Transient file bytes: `9f3fba45290c92cc782393c420520b25c7d630fa080b9e2aff1d9f52a33d1681`. These are the same intended two-line bytes now held in the independent overlay copy.
- Frozen file after restoration: `5042632a1d44d12cabbc3d09fa6dc9c9f40ac51b3f81ab733ccedd95b58b4f0a`, matching the frozen manifest.

`verify_overlay.py` rechecks every source entry from the frozen manifest, not just the files in this overlay. No GPU run or native result was produced by this overlay.
