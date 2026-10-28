# Exact registered skin/fat volume import

The importer carries exact emitted O/I/D boundaries and the combined TetGen node map into the existing physical skin/fat bundle. It keeps source ancestry, parent registration, geometric donor/region labels and frame ownership separate. Exact shared-I identity is reciprocal; arbitrary duplicate-node welding is rejected. Full current/rest poses are required for every body; unbound poses do not acquire invented skin bindings. Interior nodes carry explicit absence of surface ancestry. Stable deep attachments map by source identity rather than old vertex rows.

Facet maps derive from exact boundary node IDs and tetrahedral incidence. Remaining external facets require retained, parent-backed authored head-wall rows. New arbitrary holes cannot be silently capped. Preparation is transactional on failure. Physical volume, conservative mass mapping, and runtime admission remain their existing owners.

Build on SSH Numi:

```sh
cmake --build build-native --target numi-human-physical-skin-registered-volume-import-check numi-human-physical-skin-registered-volume-csv-check numi-human-physical-skin-registered-facet-maps-check -j2
build-native/numi-human-physical-skin-registered-volume-import-check
build-native/numi-human-physical-skin-registered-volume-csv-check
build-native/numi-human-physical-skin-registered-facet-maps-check
```

`checks.json` records source/binary/input hashes and executed fixture checks, including affine parent residual conversion, malformed input rejection and exact interface/head-wall topology. The six-tet prism and seven-node producer fixture are contract tests. No replacement body volume, mass transfer, motion, or whole-scene qualification is claimed. These helpers do not change per-step physics or install new geometry in the scene.
