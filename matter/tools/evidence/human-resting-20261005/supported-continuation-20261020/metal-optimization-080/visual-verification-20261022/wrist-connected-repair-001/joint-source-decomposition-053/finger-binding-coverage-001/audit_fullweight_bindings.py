import hashlib
import json
import pathlib
import struct

root = pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
skin = pathlib.Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
m4 = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/native-initial-force-report-source-1187/docs/media/numi-human-whole-body-all-dof-v1/m4-pro.json")
regional = root / "registered-arm-skin-bone-weight-audit-v2.json"
out = pathlib.Path(__file__).with_name("fullweight-binding-coverage.json")
raw = skin.read_bytes()
header = struct.Struct("<8s5I32s")
magic, abi, binding_count, vertex_count, index_count, registration, source_hash = header.unpack_from(raw)
if magic[:7] != b"NHSKIN1" or magic[7] != 0 or abi != 5 or binding_count != 86:
    raise SystemExit("unexpected NHSKIN header identity")
header_bytes = header.size
bind_record = struct.Struct("<I8f")
bindings = [bind_record.unpack_from(raw, header_bytes + i * bind_record.size)[0] for i in range(binding_count)]
vertex_bytes = 56
index_bytes = 4 * index_count
weights_offset = header_bytes + binding_count * bind_record.size + vertex_count * vertex_bytes + index_bytes
if len(raw) != weights_offset + vertex_count * binding_count * 4:
    raise SystemExit("NHSKIN full-weight byte count mismatch")
weights = struct.unpack_from("<" + str(vertex_count * binding_count) + "f", raw, weights_offset)
if len(set(bindings)) != len(bindings):
    raise SystemExit("duplicate body index in binding table")
m4_data = json.loads(m4.read_text())
names = {entry.get("child_body"): entry.get("child_body_name") for entry in m4_data["residuals"] if entry.get("child_body_name")}
# The retained registration donor audit names body IDs 49/99 as right/left capitate.
names.update({49: "capitate_r", 99: "capitate_l"})

def stats(slot):
    values = [weights[row * binding_count + slot] for row in range(vertex_count)]
    return {
        "body_index": bindings[slot],
        "body_name": names.get(bindings[slot], "unresolved"),
        "weight_sum_over_source_vertices": sum(values),
        "maximum_vertex_weight": max(values),
        "vertex_counts_weight_gt_threshold": {
            str(t): sum(value > t for value in values) for t in (0.001, 0.01, 0.05, 0.25, 0.5)
        },
    }

finger_stats = [stats(i) for i, body in enumerate(bindings) if any(token in names.get(body, "") for token in ("ph", "thumb"))]
capitate_stats = [stats(i) for i, body in enumerate(bindings) if body in (49, 99)]
dominant = {name: 0 for name in sorted(set(names.get(body, "unresolved") for body in bindings))}
for row in range(vertex_count):
    slot = max(range(binding_count), key=lambda col: weights[row * binding_count + col])
    dominant[names.get(bindings[slot], "unresolved")] += 1
regional_data = json.loads(regional.read_text())
result = {
    "schema": "numi.human_skin_fullweight_finger_binding_audit.v1",
    "scope": "source NHSKIN full-weight binding inventory; descriptive only, not a physical qualification",
    "skin_file": str(skin),
    "skin_file_sha256": hashlib.sha256(raw).hexdigest(),
    "skin_source_payload_sha256": source_hash.hex(),
    "skin_header": {"payload_abi": abi, "binding_count": binding_count, "vertex_count": vertex_count, "index_count": index_count, "registration_fingerprint": registration},
    "source_oracle": {"path": str(m4), "sha256": hashlib.sha256(m4.read_bytes()).hexdigest(), "revision": m4_data["provenance"]["source"]["revision"], "archive_sha256": m4_data["provenance"]["source"]["archive_sha256"]},
    "regional_weight_audit": {"path": str(regional), "sha256": hashlib.sha256(regional.read_bytes()).hexdigest(), "method": regional_data["method"], "left_digits": regional_data["regional"]["left"]["regions"]["digits"], "right_digits": regional_data["regional"]["right"]["regions"]["digits"]},
    "finger_bindings": finger_stats,
    "capitate_bindings": capitate_stats,
    "dominant_binding_vertex_counts": dominant,
    "interpretation": [
        "The full ABI-5 table binds articulated proximal, middle, and distal phalanges on both hands; raw nonzero tails are not treated as meaningful support.",
        "Threshold counts report per-vertex weight greater than 0.01 as a descriptive cutoff, not an acceptance gate.",
        "The separate registered spatial audit finds capitate-dominant digit regions and low dominant-binding/nearest-bone agreement; this indicates weak or misplaced finger-region influence despite phalange bindings being present."
    ]
}
out.write_text(json.dumps(result, indent=2) + chr(10))
