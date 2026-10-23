# Human FEM barycentric surface attachments

One finite spring connects a barycentric point on three FEM nodes to an
articulated body material point. Forces scatter through B-transpose; one
opposite body wrench shares the existing load, acceptance and rollback owner.
The law retains the original slave tributary area and stiffness: kA is N/m,
and stored energy is 0.5*kA*gap-squared. This does not create three springs.

Run on the configured Apple host:

```sh
python3 tests/numi_human_fem_surface_attachment/run.py --build-dir /tmp/human-surface-check
```

To include actual adapter initialization, snapshot-authority and malformed
source checks, rebuild the Matter library and supply its matching metallib:

```sh
python3 tests/numi_human_fem_surface_attachment/run.py --build-dir /tmp/human-surface-native-check --native-library /absolute/build/lib/libmetalrobo.dylib --native-metallib /absolute/build/matter/shaders/NumiMatter.metallib
```

The runner retains commands, outputs, hashes and timings outside the checkout.
It checks the production host validator, transpose work, all node-pair tangent
blocks, actual Metal forces/wrenches, energy, generalized projection, accepted
audit publication, and invalid-state rejection. The native initialization
fixture retains the existing fixed-node anchors and endpoint requirements;
its empty/surface/mixed cases refer only to optional interaction spans.
Initialization does not advance physical time.

The optional source span and diagnostics field change C++ layouts. Rebuild
library and callers together; old binaries are incompatible. The legacy
64-byte contact record and 96-byte dispatch stay the same size, with a former
reserved dispatch word carrying the new count. No snapshot format changes.

This is a force-owner extension, not a remesher, site correspondence solver,
implicit surface-spring tangent, global bone broadphase, or whole-human motion
qualification. Callers must supply anatomically valid triangle/site mappings.
The CPU force/tangent oracle exists only in tests; simulation stays on Metal.
