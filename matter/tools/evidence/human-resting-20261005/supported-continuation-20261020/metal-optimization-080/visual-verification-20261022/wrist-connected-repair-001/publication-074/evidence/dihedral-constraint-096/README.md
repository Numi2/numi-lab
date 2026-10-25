# Dihedral safeguard draft 096

This isolated offline construction draft adds an opt-in mapped-detail curvature safeguard. It does not change the live scene, material state, runtime, or canonical P074 files.

MappedSkinSurface(max_dihedral_change_degrees=bound, dihedral_edge_ids=ids) measures signed dihedral change from the constructor's immutable detail_origin. The signed convention matches regional-distortion-094: atan2(edge_unit dot (n0 cross n1), n0 dot n1), with the first incident face's directed edge and shortest periodic angle difference. IDs are assigned after lexicographically sorting undirected edges and keeping only exactly-two-face edges; direction follows the first face-edge incidence. Omitting both options preserves disabled behavior and the prior report shape. A bound without explicit IDs covers every such manifold edge.

When used by coupled_clearance_step, active constraints are filtered through the mapped displacement matrix W, analytic angle derivatives are composed through W, and upper/lower hard LP rows bound the linearized signed change. Every backtracked candidate is still checked after Float32 emission against the unchanged exact nonlinear dihedral, area, and normal gates. Empty active support yields zero rows. This is a configurable construction safeguard, not anatomical validation or physical admission.

Run the focused suite from this directory with:

    /Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q matter/tools/test_human_skin_fat_embedded_surface.py matter/tools/test_human_skin_fat_connected_repair.py tests/test_dihedral_constraint.py

The frozen check receipt is in manifest.json; test-run.log contains the as-run pytest output.
