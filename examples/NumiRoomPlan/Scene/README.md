# NumiProperty RoomPlan scene

This scene preserves the supplied RoomPlan USDZ and adds the native Numi Lab
visual-scene representation beside it.

- `NumiProperty-2E2B8FAC-2E2C-465C-85FC-D4B02503E02C.usdz` is the unchanged
  RoomPlan source (`sha256:0e4b6c07bd9acfb1137c4b33a6cd29ae29b5f2653a8ba4cb1efa3b43afa65dfc`).
- `NumiProperty-2E2B8FAC-2E2C-465C-85FC-D4B02503E02C.mrvpack` is the native
  `VisualAssetPackV2` cooked from that source
  (`sha256:d46264c9d9f50c776439c5265f694cd337e2889ae061d0e5351f411891e9efc`).
- `NumiProperty-2E2B8FAC-2E2C-465C-85FC-D4B02503E02C.visual.v3.json` is the
  validated `VisualSceneManifestV3` with fingerprint
  `2852369877894046265`.

The scene contains 1,932 vertices, 2,022 indices, 2 materials, and no external
textures. It is authored presentation geometry bound to the Numi Lab
`workspace` asset slot with the indoor-area light rig. This import does not
invent collision or dynamics data from the visual scan; a physics-ready room
would need a separate authored WorldPack collision layer.

The repeatable authoring path is:

```sh
build/bin/metalrobo_visual_cook --id numi_property_roomplan_scan \
  --provenance numi_roomplan_import/v1 \
  NumiProperty-2E2B8FAC-2E2C-465C-85FC-D4B02503E02C.usdz \
  NumiProperty-2E2B8FAC-2E2C-465C-85FC-D4B02503E02C.mrvpack

build/bin/metalrobo_room_scan_scene \
  NumiProperty-2E2B8FAC-2E2C-465C-85FC-D4B02503E02C.mrvpack \
  NumiProperty-2E2B8FAC-2E2C-465C-85FC-D4B02503E02C.visual.v3.json
```
