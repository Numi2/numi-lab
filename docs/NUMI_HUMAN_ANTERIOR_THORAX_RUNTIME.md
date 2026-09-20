# Numi Human anterior abdominal-wall runtime

The `--anterior-thorax-continuum-payload <NHTHRC1>` path makes seven exact
torso-side abdominal attachment boundaries live Matter FEM owners. It is
available only inside a persistent NHTENDON2/3 Human transaction and cannot
share the current single-continuum slot with Open Knee or NHFASC3.

For every admitted muscle, the payload endpoint is the torso anchor terminal
and the opposite endpoint is the load witness. The adapter distributes the
declared force share over the payload support maps, removes the same share from
the torso endpoint's source `J^T` contribution, and projects accepted fixed-node
reaction back through the torso body Jacobian. Human and FEM state commit,
replay, or roll back together in the same Metal command buffer.

The material
`matter/materials/human_anterior_abdominal_wall_composite_effective.nmatter`
is a reduced isotropic proxy: 10% of the 1.12 MPa median human abdominal-wall
composite modulus reported by Cooney et al. (PMCID PMC10604332), paired with the
payload's 10% owner share. It is not a directional or subject-calibrated law.

## Qualified Apple path

The current `NHTHRC1` payload SHA-256 is
`066d6ada2d0680df43351fa08f3fc1e65a80387f757a72173f01bc38edf9f065`.
On Apple M4 Pro, a four-step selected-control run at `10 us` and 2% selected
increment passed with:

- 2,556 FEM nodes, 5,424 tetrahedra, 83 loaded nodes, 126 fixed nodes;
- 78.4833 N assigned terminal-force L1;
- 3.32478 N accepted fixed-node reaction L1;
- 0.249343 mm maximum displacement and `Jmin = 0.956072`;
- two FGMRES iterations;
- bitwise Human/FEM replay and verified downstream-rejection rollback; and
- 12,366.76 ms complete coupled transaction time.

This qualifies the declared four-step transaction. It is not a long-horizon
performance result, a breathing model, an abdominal-layer contact solve, or a
clinical material validation.
