# Synthetic wound-lip traction — invalid preflight record

The first preregistered control command in
`SYNTHETIC_SKIN_TRACTION_STUDY_20261004.md` exited before any Metal command
buffer was created. Its stdout reports:

```text
synthetic_skin_wound_traction=failed reason="compiled traction world changed the skin node layout"
```

The executable compared the compiled FEM state capacity directly with the
authored node count. The coupon reserves 64 node slots and 128 tetrahedron
slots for later topology mutation, so compiled capacity is intentionally
larger. This was an instrument mapping error, not a simulation result.

The failure is retained with its exact run binding and exit status at
`/Users/n/numi-skin-suture-perf-100x/runs/goal-medical-skin-traction-20261004/trial-control/`.
That directory contains no CSV samples; execution stopped before creating
the output directory, requesting a Metal device, or dispatching GPU work.
The attempt is invalid and supplies no control observation.

The replacement instrument maps authored nodes through the compiled
`NMFEMTopologyNodeGPU` identity, verifies each mapped rest position, counts
authored active tetrahedra separately from reserved capacity, and adds a
CPU-only `--validate-only` preflight. The material, coupon, forces, timing,
prediction, and acceptance limits are unchanged. The amended preregistration
is `SYNTHETIC_SKIN_TRACTION_STUDY_20261004_V2.md`; it was written before any
new trajectory samples.
