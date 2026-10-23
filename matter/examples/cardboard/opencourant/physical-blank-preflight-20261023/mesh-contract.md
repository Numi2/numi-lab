# OpenCourant cardboard HEX8 mesh contract

The geometry entry points live in `matter/tools/cardboard_opencourant_hex.py`:

- `build_score_fold_strip(config=None) -> HexMesh` returns a small source-recipe coupon for score/fold instrumentation.
- `build_fefco0201_blank(config=None) -> HexMesh` returns the unscored RSC blank with a 25 mm manufacturer tab, 1 mm slots, score/tool face regions, and explicit solid liner, flute-medium, and starch geometry.
- `export_mesh_binary(mesh, path)` writes a compact geometry archive. `export_mesh_json(mesh, path)` writes the expanded audit sidecar and should not be used for the full blank.

Both builders return the same in-memory `HexMesh` contract. `nodes` are dictionaries `{id, xyz_mm, part}` with one-based IDs. Each `HexElement` has `{id, nodes[8], part, material_role, frame_angle_rad, frame_axes[3]}`; HEX8 node order is `x0_y0_z0, x1_y0_z0, x1_y1_z0, x0_y1_z0, x0_y0_z1, x1_y0_z1, x1_y1_z1, x0_y1_z1`. `frame_axes` stores the local MD, CD, and thickness vectors as three columns in global XYZ. The corrected flute frame follows the positive profile tangent: `MD=(cos(theta),0,+sin(theta))`, `CD=(0,1,0)`, `thickness=(-sin(theta),0,cos(theta))`.

`interface_pairs` contain independent coincident four-node face references on each side of finite starch bridges (`medium_starch` and `starch_liner`). They are candidate TYPE43 surfaces; `law_material_id` remains unset until the solver adapter binds a qualified cohesive material. `surface_sets` map boundary-set names to `FaceRef(element_id, face, nodes[4])`; `node_sets` map boundary/grip names to one-based node IDs. `parameters`, `starch_patches`, `score`, `tool_targets`, `frame_convention`, `provenance`, and `quality` carry the audit metadata. Score and tool sets describe where fixture contact can occur; they do not mean the score was already formed.

The binary file is little-endian and writer-only at present. Its prefix is eight bytes `OCHEX8\0\1`, then a uint32 header length and compact UTF-8 JSON header. Records follow `nodes`, `hex8_elements`, `interface_pairs`, `surface_sets`, `node_sets`:

| Record | Encoding | Meaning |
| --- | --- | --- |
| Node | `<dddH` | x/y/z in mm as float64, then uint16 part ID |
| HEX8 | `<8IHBf` | eight one-based uint32 node IDs, uint16 part ID, uint8 material-role ID, float32 frame angle in radians |
| Interface pair | `<BHIBIB` | uint8 interface ID, uint16 patch ID, side-A uint32 element ID and uint8 face ID, side-B uint32 element ID and uint8 face ID |
| Surface-set header and faces | `<HQ`, UTF-8 name, then repeated `<IB` | uint16 name byte count, uint64 face count, then uint32 element ID and uint8 face ID per face |
| Node-set header and nodes | `<HQ`, UTF-8 name, then repeated `<I` | uint16 name byte count, uint64 node count, then one-based uint32 node ID per entry |

The header provides the part/material/face/interface ID maps and the record order. Surface and node set records appear in the mesh dictionary insertion order; a reader should use the names in each record rather than assuming sorted header-key order. The element record carries the angle, not all three expanded frame vectors; reconstruct using the header frame convention. The binary is geometry interchange, not a solver deck or material assignment.

For FEFCO 0201 closure, the present substrate grid does not provide direct one-to-one matching overlap faces after ideal folding. The tab's local 25 mm edge has no corresponding panel station (nearest are 24.6875 and 25.675 mm). Across an end-flap overlap, one axis has 153 versus 12 stations and the orthogonal axis 12 versus 154; slot kerfs also leave cut boundaries. Do not move substrate FEM nodes to force a match.

A possible controlled fixture is to mesh matching fine TYPE43 faces as an independent overlay on each side, with overlay nodes kinematically tied/interpolated to each unmodified substrate through standard TYPE2. A minimal test would use one coarse H8 substrate face per side and a 2-by-2 overlay of four cohesive H8s per side (nine overlay nodes each), then test contact, a common rigid rotation, accepted birth, and nonuniform mixed opening/shear. Measure substrate reaction versus cohesive resultant and global/cohesive work balance. Zero-mass auxiliary-node support and energy/work correctness are not yet verified, so this remains a fixture proposal; no closure mapping is emitted by the blank builder.
