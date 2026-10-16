# Source193 shipping-scale admission audit

This is a CPU/source-only review of frozen source193. It does not modify production files and contains no native GPU or physical transaction result.

The exact shipping topology anchors are 528,444 nodes, 1,477,728 tetrahedra, 1,054,876 exterior faces, 260,835 cohesive faces, 527,964 contact-eligible nodes, and 12 proxies. The current full-contact path admits at most 8,192 FEM nodes and 32,768 tetrahedra; its allocator allows 16,384 nodes. The mesh is far above those admitted-scope limits, while the exact surface/BVH and flattened sparse-pair indices still fit uint32.

For E=1 and contact capacity 131,072, checked source helpers estimate 77,661,184 bytes of private full-contact scratch when surface buffers alias production storage, below the 128 MiB limit. Without alias, the same arena is 296,269,008 bytes. Alias saves duplicate memory; it does not reduce contact work. The opt-in incidence sort is what replaces the legacy per-node scan over active contacts. At this capacity it costs 12,582,912 bytes and 210 dispatches. No timing claim is made.

The broadphase candidate limit remains a real resource cap. Source code explicitly marks overflow and prevents scatter from continuing in an unhealthy environment; the actual shipping candidate count was not measured. Sparse rigid contact at 65,536 slots is within the 36 MiB helper budget, but actual occupancy is unknown. Cohesive trials fit in three 8 MiB pages.

The source193-recompiled prebuild inventory retains all eight scenarios. Its 131,072-slot sparse candidate estimates 14,660,971,798 bytes, below 16 GiB numerically, but reports not admitted due topology limits. This is not an observed runtime peak. Precompile surface count uses 4*T as an upper bound, and the estimate omits the optional incidence-sort workspace; those effects point in opposite directions.

Root reports source193 validation was refused before output-case creation because an undeclared runtime.mm.orig made the file scan 817 against the frozen 816. No GPU dispatch occurred. The source193 build receipt is separate and passed.

A safe next step is an exact compiled-layout preflight that includes all buffers and current guards. If admitted, a shipping-specific scope can preserve the existing capacities and fail closed on overflow. If actual contact occupancy exceeds the cap, candidate processing must be tiled with complete history ownership rather than silently truncated or accepted by a blanket cap raise.

See review.json for detailed checks and source line bindings. The first failed analyzer compile is retained under attempts/attempt001.
