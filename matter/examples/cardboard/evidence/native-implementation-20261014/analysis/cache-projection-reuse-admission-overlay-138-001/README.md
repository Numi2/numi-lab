# Source136 cache-projection reuse admission correction

This isolated overlay repairs two admission mismatches observed in native437. It is based on the immutable source136 snapshot and does not modify that snapshot or the earlier mode-1 candidate.

The first rejected predicate treated `NM_MATTER_IPC` as evidence of active contact. In source136, `compiler_impl.cpp` unconditionally ORs this capability bit into every compiled dispatch. The overlay removes only that bit from the cache-reuse unsupported-flags mask. It retains the actual-contact guards: contact pairs and rigid proxies must be absent, contact counts must be zero, and the object must disable both self-contact and deformable contact. Adaptive, mixed FEM, multiphysics, mutation, learned and identification flags, MPM counts, and other scope checks remain rejected.

The second rejected predicate required every active tetrahedron to use a 15-state material. The native437 source coupon has 1,152 liner tets and 576 medium tets using compensated accurate generic 15-state programs, plus 192 finite-elastic starch-glue tets. The overlay admits that zero-state glue only when its compiled layout matches the finite-elastic path: paper-compensated arithmetic, zero state, three parameters, Neo-Hookean constitutive kind, generic projection, no Hill or generic implicit flags, and no state, dissipation, viscous, or implicit program offsets. Those glue elements stay on the existing repeated-projection tangent path. At least one active accurate 15-state paper element is still required; other active material layouts fail closed.

The sibling tangent-cache kernel now matches the source134 implementation with only the earlier 15-state-only guard removed. Its existing conditional reuse code caches the projected state for the 15-state elements and passes null reuse pointers for the finite-elastic glue. The original production tangent-cache kernel is unchanged.

The original native437 initialization failure is preserved at `solver/cache-reuse-full-history-parity-437`; its selected source136 run exited 1 before any FEM step. This overlay has passed source checks, strict Objective-C++ host syntax, Metal O0/O3 translation-unit compile, and O3 metallib link. No native run, off/on production-history comparison, physical qualification, or performance claim has been made.

## Bound source and artifacts

Base: `/Users/home/cardboard-implementation-20261014/snapshots/source-cache-projection-reuse-runtime-136`.
The retained source136 bundle is `analysis/retained-source-snapshot136-001/source-136.json.gz`; compressed and uncompressed hashes are recorded in `manifest.json`.

The source compiler sets `NM_MATTER_IPC` unconditionally at `matter/src/compiler_impl.cpp`; that source file and the source136 validation predicate are hashed as review inputs in the manifest. The native437 failure text, receipt, and output manifest are hash-bound as well.

`overlay.patch` applies only to `matter/src/runtime.mm` and `matter/src/metal/fem.metalinc`. `verify_overlay.py` checks the frozen base hashes, source134 sibling equivalence, production-kernel identity, exact material-scope and retained physical-scope guards, native437 input identity, patch applicability, and absence of symlinks.

The source136 tangent-cache reuse mode remains opt-in through `NM_FEM_CACHE_PROJECTION_REUSE=1`; default mode behavior is unchanged. A fresh native off/on history comparison is required before any runtime result or speed claim.
