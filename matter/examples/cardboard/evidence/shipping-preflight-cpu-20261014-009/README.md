# Shipping-size compile-only preflight

This record compiles the 300 × 200 × 200 mm FEFCO 0201 shipping-box lower-bound geometry. It is a CPU/source compile and static memory estimate only: Matter and Metal runtime were not initialized, no GPU work or physics step was run, no score line was physically scored, and the assembly remains flat and unqualified.

The command was:

```sh
build-cardboard-world/matter/numi-matter-cardboard-probe --shipping-box --compile-only --output matter/examples/cardboard/evidence/shipping-preflight-cpu-20261014-009
```

The source record is [preflight-audit.json](preflight-audit.json). It binds the current checkout revision, relevant source/material hashes, probe binary, compiled package, and generated geometry/manifests. The result reports 453,378 nodes and 1,477,728 tetrahedra in the flat blank, four panels, and twelve score lines. The authored die geometry has twelve 1 mm punch capsules and two 10 mm support rails around each 6 mm channel; CPU geometry checks passed, but no native contact was evaluated.

The requested internal dimensions are set to 300 × 200 × 200 mm, with a 25 mm manufacturer tab and 1 mm kerf. Because no accepted scored-coupon allowance was supplied, the generated dimensions use an explicit zero-offset lower bound; they are not accepted manufacturing dimensions. The score channels use the declared 0.1–0.4-caliper indentation-depth assumptions. The closure plan reports nine overlap patches (0.085 m² total overlap, 0.010 m² strip footprint) and leaves the major-flap butt edge unbonded. No HMA law, corrugator-starch cohesive law, coupon measurement, or accepted score/fold/release gate is present.

The prebuild planning estimate is 821,124,688 bytes. The exact compiled FEM/Krylov arena lower bound is 656,899,296 bytes, below the 16 GiB limit. This lower bound excludes contact, checkpoints, other runtime arenas, host/compiler/package allocations, and other buffers. Actual native peak allocation remains unmeasured and requires runtime initialization before admission.

All six relevant CPU geometry/driver checks are recorded as passed in the audit JSON. The separate retained outputs 001–008 are not overwritten; their failure/superseded classifications are listed in the audit.
