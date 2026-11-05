#!/usr/bin/env python3
"""Active-row current 217 adapter for the frozen 433 exporter.

The source O/I/D NPZs keep their full authoring point IDs. Only points used by
current surface faces are compacted into the 433 role domain. Every omitted row
is retained explicitly as source provenance and must map to -1 in the pinned
217 full-domain crosswalk.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE_DIR = HERE.parent
ADAPTER_1021_PATH = pathlib.Path('/Users/numi/Work/human-integrated-gaps-20261104/joint-boundary-rebuild-64c/compiler/typed-adapter/typed-current-schema.py')
ADAPTER_1021_SHA256 = '1d734c4c69068ba48a78face65955a1cd31b4d2ecc419dbe872d4ce1d5129166'
ROLE_ORDER = ("O", "I", "D")
CROSSWALK = None
CROSSWALK_PATH = None
CROSSWALK_SHA256 = None


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_1021():
    if not ADAPTER_1021_PATH.is_file() or sha256_file(ADAPTER_1021_PATH) != ADAPTER_1021_SHA256:
        raise ValueError("current 1021 adapter source pin changed")
    spec = importlib.util.spec_from_file_location("current_433_schema_adapter_1021_pinned", ADAPTER_1021_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(str(ADAPTER_1021_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_surface_crosswalk(path: pathlib.Path, expected_sha256: str) -> dict:
    import numpy as np
    path = pathlib.Path(path).resolve()
    if len(str(expected_sha256)) != 64 or sha256_file(path) != str(expected_sha256):
        raise ValueError("217 registered-surface node-map hash pin mismatch")
    with np.load(path, allow_pickle=False) as z:
        required = {"inactive_sentinel"}
        for role in ROLE_ORDER:
            required.update({
                f"{role}_source_row_to_joint_node",
                f"{role}_active_source_vertex_rows",
                f"{role}_inactive_source_vertex_rows",
            })
        missing = required - set(z.files)
        if missing:
            raise ValueError("217 registered-surface node-map lacks fields: " + ",".join(sorted(missing)))
        sentinel = np.asarray(z["inactive_sentinel"])
        if sentinel.shape != () or sentinel.dtype.kind not in "iu" or int(sentinel) != -1:
            raise ValueError("217 registered-surface node-map must pin inactive sentinel -1")
        maps = {}
        active = {}
        inactive = {}
        for role in ROLE_ORDER:
            for key in (f"{role}_source_row_to_joint_node",
                        f"{role}_active_source_vertex_rows",
                        f"{role}_inactive_source_vertex_rows"):
                a = np.asarray(z[key])
                if a.dtype.kind not in "iu" or a.ndim != 1:
                    raise ValueError(f"217 crosswalk {key} must be a one-dimensional integer row map")
            maps[role] = np.asarray(z[f"{role}_source_row_to_joint_node"], dtype=np.int64).copy()
            active[role] = np.asarray(z[f"{role}_active_source_vertex_rows"], dtype=np.int64).copy()
            inactive[role] = np.asarray(z[f"{role}_inactive_source_vertex_rows"], dtype=np.int64).copy()
            if (np.any(maps[role] < -1) or np.any(active[role] < 0) or
                    np.any(inactive[role] < 0) or
                    len(np.unique(active[role])) != len(active[role]) or
                    len(np.unique(inactive[role])) != len(inactive[role])):
                raise ValueError(f"217 {role} crosswalk has invalid IDs or duplicate row lists")
        return {
            "path": str(path), "sha256": str(expected_sha256),
            "schema": "numi.h706.217-registered-surface-node-map.v1",
            "inactive_sentinel": -1, "source_row_to_joint_node": maps,
            "active_source_vertex_rows": active,
            "inactive_source_vertex_rows": inactive,
        }


def validate_role_crosswalk(role_asset: dict, crosswalk: dict) -> None:
    import numpy as np
    role = role_asset["role"]
    if role not in ROLE_ORDER:
        raise ValueError("crosswalk validation received unknown role")
    source_count = len(role_asset["source_points_world_m_f32"])
    source_active = np.asarray(role_asset["active_source_vertex_rows"], dtype=np.int64)
    source_inactive = np.asarray(role_asset["inactive_source_vertex_rows"], dtype=np.int64)
    mapping = np.asarray(crosswalk["source_row_to_joint_node"][role], dtype=np.int64)
    active = np.asarray(crosswalk["active_source_vertex_rows"][role], dtype=np.int64)
    inactive = np.asarray(crosswalk["inactive_source_vertex_rows"][role], dtype=np.int64)
    exact_active = np.unique(np.asarray(role_asset["source_faces"], dtype=np.int64).reshape(-1))
    exact_inactive = np.setdiff1d(np.arange(source_count, dtype=np.int64), exact_active,
                                  assume_unique=True)
    if (mapping.shape != (source_count,) or not np.array_equal(source_active, exact_active) or
            not np.array_equal(active, exact_active) or
            not np.array_equal(source_inactive, exact_inactive) or
            not np.array_equal(inactive, exact_inactive)):
        raise ValueError(f"217 {role} inactive/active source row sets differ from exact current face use")
    if (np.any(mapping[active] < 0) or np.any(mapping[inactive] != -1) or
            len(np.unique(mapping[active])) != len(active)):
        raise ValueError(f"217 {role} crosswalk does not map every active row injectively and only active rows")


def load_active_current_typed_role_asset(role, path, expected_sha256):
    import numpy as np
    base_adapter = _load_1021()
    asset = base_adapter.load_current_typed_role_asset(role, path, expected_sha256)
    full_points = np.asarray(asset["points_world_m_f32"], dtype=np.float32)
    full_faces = np.asarray(asset["faces"], dtype=np.int64)
    active = np.unique(full_faces.reshape(-1))
    inactive = np.setdiff1d(np.arange(len(full_points), dtype=np.int64), active,
                             assume_unique=True)
    source_to_compact = np.full(len(full_points), -1, dtype=np.int64)
    source_to_compact[active] = np.arange(len(active), dtype=np.int64)
    compact_faces = source_to_compact[full_faces]
    if np.any(compact_faces < 0):
        raise ValueError("current face references a source point excluded from active role rows")
    asset["source_points_world_m_f32"] = full_points.copy()
    asset["source_faces"] = full_faces.copy()
    asset["active_source_vertex_rows"] = active.copy()
    asset["inactive_source_vertex_rows"] = inactive.copy()
    asset["source_vertex_to_compact_role_row"] = source_to_compact
    asset["points_world_m_f32"] = full_points[active].copy()
    asset["faces"] = compact_faces
    if CROSSWALK is None:
        raise ValueError("current typed 433 mode requires the pinned full-domain 217 surface crosswalk")
    validate_role_crosswalk(asset, CROSSWALK)
    asset["registered_surface_node_map_path"] = CROSSWALK["path"]
    asset["registered_surface_node_map_sha256"] = CROSSWALK["sha256"]
    return asset


def _compact_node_map_rows(node_map_rows, role_assets):
    import numpy as np
    if CROSSWALK is None:
        raise ValueError("current typed node map requires a full-domain 217 crosswalk")
    active_rows_by_role = {}
    for role in ROLE_ORDER:
        active = np.asarray(role_assets[role]["active_source_vertex_rows"], dtype=np.int64)
        active_rows_by_role[role] = {
            int(source_row): compact for compact, source_row in enumerate(active)
        }
    compact_rows = []
    observed = {role: set() for role in ROLE_ORDER}
    for raw in node_map_rows:
        row = dict(raw)
        role = row.get("role")
        if role in ROLE_ORDER:
            try:
                source_row = int(row["registered_surface_vertex_row"])
                joint_node_id = int(row["joint_node_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("217 role map lacks source row or canonical joint node ID") from exc
            mapping = CROSSWALK["source_row_to_joint_node"][role]
            if source_row < 0 or source_row >= len(mapping):
                raise ValueError(f"217 {role} source surface row is outside its full-domain map")
            expected_joint_row = int(mapping[source_row])
            if expected_joint_row < 0:
                raise ValueError(f"217 node map emitted inactive {role} source row {source_row} as a volume node")
            if joint_node_id - 1 != expected_joint_row:
                raise ValueError(f"217 {role} row {source_row} joint-node crosswalk differs from the active child map")
            if source_row not in active_rows_by_role[role]:
                raise ValueError(f"217 node map includes non-face-active {role} source row {source_row}")
            compact = int(active_rows_by_role[role][source_row])
            row["current_typed_source_surface_vertex_row"] = source_row
            row["registered_surface_vertex_row"] = compact
            observed[role].add(source_row)
        else:
            row["current_typed_source_surface_vertex_row"] = -1
        compact_rows.append(row)
    for role in ROLE_ORDER:
        expected = set(map(int, role_assets[role]["active_source_vertex_rows"]))
        if observed[role] != expected:
            missing = sorted(expected - observed[role])[:8]
            extra = sorted(observed[role] - expected)[:8]
            raise ValueError(f"217 {role} active surface crosswalk coverage mismatch missing={missing} extra={extra}")
    return compact_rows



def _replace_node_registration_source_row(source):
    """Keep local surface row dense while exporting source_vertex_row in its original namespace."""
    old = (
        "        registration_rows.append([role, local, IDENTITY_KIND, cell, *beta,\n"
        "                                      *parent_residual[node], norm(parent_residual[node]), 0, FRAME_METHOD,\n"
        "                                      ROLE_DOMAIN[role], local, -1, 0, 0, 0, 0, 0, 0])"
    )
    new = (
        "        source_vertex_row = int(row.get(\"current_typed_source_surface_vertex_row\", local))\n"
        "        registration_rows.append([role, local, IDENTITY_KIND, cell, *beta,\n"
        "                                      *parent_residual[node], norm(parent_residual[node]), 0, FRAME_METHOD,\n"
        "                                      ROLE_DOMAIN[role], source_vertex_row, -1, 0, 0, 0, 0, 0, 0])"
    )
    count = source.count(old)
    if count != 1:
        raise ValueError(f"433 source registration patch expected one match, found {count}")
    return source.replace(old, new, 1)


def _patch_export_for_original_surface_rows(owner):
    """Patch only serialized source IDs; local geometry rows remain dense."""
    import inspect

    source = inspect.getsource(owner.export)
    source = _replace_node_registration_source_row(source)
    replacements = [
        (
            '{"source_vertex_row": i, "child_volume_node_row": int(row["volume_node_row"]), "parent_volume_node_row":int(pn[int(row["volume_node_row"])]),',
            '{"source_vertex_row": int(row.get("current_typed_source_surface_vertex_row", i)),\n'
            '                 "registered_surface_local_row": i,\n'
            '                 "child_volume_node_row": int(row["volume_node_row"]), "parent_volume_node_row":int(pn[int(row["volume_node_row"])]),'
        ),
        (
            '"source_vertex_count": len(role_rows[role]),\n            "source_face_count": len(parent_face_info),',
            '"source_vertex_count": len(role_rows[role]),\n'
            '            "source_point_domain_count": (len(current_typed_role_assets[role]["source_points_world_m_f32"])\n'
            '                if current_typed_requested else len(role_rows[role])),\n'
            '            "source_face_count": len(parent_face_info),'
        ),
        (
            'nrows.append([*prefix, i, -1, 1, 0, 0, 0, i, -1, 0, 0, 0,',
            'nrows.append([*prefix, i, -1, 1, 0, 0, 0,\n'
            '                              int(row.get("current_typed_source_surface_vertex_row", i)), -1, 0, 0, 0,'
        ),
        (
            '*local_point, ROLE_DOMAIN.get(role,"derived_volume_interior_no_surface_ancestry"), "vertex" if role!="interior" else "none", local, -1, 0, 0, 0,',
            '*local_point, ROLE_DOMAIN.get(role,"derived_volume_interior_no_surface_ancestry"), "vertex" if role!="interior" else "none", source_vertex_row, -1, 0, 0, 0,'
        ),
        (
            '        common = [node + 1, None, None, local if role == "I" else -1,',
            '        source_vertex_row = (int(layer_row.get("current_typed_source_surface_vertex_row", local))\n'
            '                             if role != "interior" else -1)\n'
            '        common = [node + 1, None, None, local if role == "I" else -1,'
        ),
        (
            '"current_role_asset_pins":{r:{"path":a["path"],"sha256":a["sha256"]} for r,a in current_typed_role_assets.items()},\n                "child_label_reference_source_sha256":',
            '"current_role_asset_pins":{r:{"path":a["path"],"sha256":a["sha256"]} for r,a in current_typed_role_assets.items()},\n'
            '            "current_typed_surface_row_domains": ({\n'
            '                "registered_surface_node_map": {\n'
            '                    "path": current_typed_role_assets["O"]["registered_surface_node_map_path"],\n'
            '                    "sha256": current_typed_role_assets["O"]["registered_surface_node_map_sha256"],\n'
            '                    "schema": "numi.h706.217-registered-surface-node-map.v1",\n'
            '                    "joint_row_base": 0,\n'
            '                },\n'
            '                "roles": {\n'
            '                    r: {\n'
            '                        "source_point_count": int(len(current_typed_role_assets[r]["source_points_world_m_f32"])),\n'
            '                        "active_source_vertex_rows": current_typed_role_assets[r]["active_source_vertex_rows"].tolist(),\n'
            '                        "inactive_source_vertex_rows": current_typed_role_assets[r]["inactive_source_vertex_rows"].tolist(),\n'
            '                        "source_row_to_joint_node": current_typed_role_assets[r]["source_row_to_joint_node"].tolist(),\n'
            '                    } for r in ROLE_ORDER\n'
            '                },\n'
            '                "inactive_policy": "unreferenced full-source points remain pinned provenance rows and are not 433 physical volume nodes",\n'
            '            } if current_typed_requested else None),\n'
            '            "child_label_reference_source_sha256":'
        ),
        (
            '        else:\n            nheader += ["source_domain", "d37_parent_node_row",\n'
            '                        "d37_parent_surface_face_row", "beta0", "beta1", "beta2",\n'
            '                        "source_residual_x_m", "source_residual_y_m",\n'
            '                        "source_residual_z_m", "frame_owner_method"]',
            '        else:\n            nheader += ["source_domain", "d37_parent_node_row",\n'
            '                        "d37_parent_surface_face_row", "beta0", "beta1", "beta2",\n'
            '                        "source_residual_x_m", "source_residual_y_m",\n'
            '                        "source_residual_z_m", "frame_owner_method"]\n'
            '            if current_typed_requested:\n'
            '                nheader += ["source_vertex_row"]'
        ),
        (
            '                nrows.append([*prefix, domain, i, -1, 0, 0, 0, 0, 0, 0, FRAME_METHOD])',
            '                nrows.append([*prefix, domain, i, -1, 0, 0, 0, 0, 0, 0, FRAME_METHOD] +\n'
            '                    ([int(row.get("current_typed_source_surface_vertex_row", i))]\n'
            '                     if current_typed_requested else []))'
        ),
    ]
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise ValueError(f"433 source-row export patch expected one match, found {count}: {old[:90]!r}")
        source = source.replace(old, new, 1)
    namespace = owner.__dict__
    exec(compile(source, str(owner.__file__), "exec"), namespace)
    owner.export = namespace["export"]

def _install_original(owner):
    global CROSSWALK
    owner.load_current_typed_role_asset = load_active_current_typed_role_asset
    original_adapt = owner.adapt_current_typed_role_rows

    def adapt_current_typed_role_rows(child_points, child_tets, child_labels, node_map_rows,
                                      role_assets, expected_role_rows=None):
        compact_rows = _compact_node_map_rows(node_map_rows, role_assets)
        adapted = original_adapt(child_points, child_tets, child_labels, compact_rows,
                                 role_assets, expected_role_rows)
        for row in adapted:
            role = row["role"]
            if role in ROLE_ORDER:
                compact = int(row["role_row"])
                row["current_typed_source_surface_vertex_row"] = int(
                    role_assets[role]["active_source_vertex_rows"][compact])
            else:
                row["current_typed_source_surface_vertex_row"] = -1
        return adapted

    owner.adapt_current_typed_role_rows = adapt_current_typed_role_rows
    original_validate = owner.validate_current_typed_candidate_binding

    def validate_current_typed_candidate_binding(candidate, candidate_node_map_rows,
            candidate_node_map_sha256, inserted_ancestry_sha256, role_asset_pins,
            current_typed_enabled, operational_enabled):
        import numpy as np
        result = original_validate(candidate, candidate_node_map_rows,
            candidate_node_map_sha256, inserted_ancestry_sha256, role_asset_pins,
            current_typed_enabled, operational_enabled)
        if not result:
            return result
        if CROSSWALK is None:
            raise ValueError("current typed 433 candidate lacks its full-domain 217 crosswalk")
        seen = {role: set() for role in ROLE_ORDER}
        for row in candidate_node_map_rows:
            role = row.get("role")
            if role not in ROLE_ORDER:
                if int(row.get("current_typed_source_surface_vertex_row", -1)) != -1:
                    raise ValueError("interior candidate node claims a current source surface row")
                continue
            try:
                source_row = int(row["current_typed_source_surface_vertex_row"])
                volume_node = int(row["volume_node_row"])
                compact_row = int(row["role_row"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("current typed candidate node map omits full source surface row") from exc
            asset = role_asset_pins[role]["loaded_asset"]
            active = np.asarray(asset["active_source_vertex_rows"], dtype=np.int64)
            crosswalk = np.asarray(CROSSWALK["source_row_to_joint_node"][role], dtype=np.int64)
            if (source_row < 0 or source_row >= len(crosswalk) or
                    compact_row < 0 or compact_row >= len(active) or
                    int(active[compact_row]) != source_row or
                    int(crosswalk[source_row]) != volume_node):
                raise ValueError(f"current typed {role} candidate row does not reproduce the pinned 217 source crosswalk")
            if source_row in seen[role]:
                raise ValueError(f"current typed {role} active source row is mapped more than once")
            seen[role].add(source_row)
        for role in ROLE_ORDER:
            if seen[role] != set(map(int, role_asset_pins[role]["loaded_asset"]["active_source_vertex_rows"])):
                raise ValueError(f"current typed {role} candidate omits or adds a face-active source row")
        return result

    owner.validate_current_typed_candidate_binding = validate_current_typed_candidate_binding
    owner.CURRENT_TYPED_NODE_MAP_HEADER = list(owner.CURRENT_TYPED_NODE_MAP_HEADER)
    if "current_typed_source_surface_vertex_row" not in owner.CURRENT_TYPED_NODE_MAP_HEADER:
        owner.CURRENT_TYPED_NODE_MAP_HEADER.append("current_typed_source_surface_vertex_row")

    original_build = owner.build_current_typed_volume_candidate_payload

    def build_current_typed_volume_candidate_payload(child_points, child_tets, child_label_rows,
            child_label_source_rows, node_map_rows, parent_points, parent_tets, parent_labels,
            parent_full86_weights, role_assets, expected_role_rows=None, p1_anchor=None):
        result = original_build(child_points, child_tets, child_label_rows,
            child_label_source_rows, node_map_rows, parent_points, parent_tets, parent_labels,
            parent_full86_weights, role_assets, expected_role_rows, p1_anchor)
        # Keep legacy count field as the physical active domain; preserve full
        # source rows and their exact crosswalk separately.
        result["role_source_row_counts"] = {
            role: int(len(role_assets[role]["source_points_world_m_f32"]))
            for role in ROLE_ORDER
        }
        result["role_rows"] = {
            role: int(len(role_assets[role]["active_source_vertex_rows"]))
            for role in ROLE_ORDER
        }
        result["current_typed_role_domains"] = {
            role: {
                "source_point_count": int(len(role_assets[role]["source_points_world_m_f32"])),
                "active_source_vertex_rows": role_assets[role]["active_source_vertex_rows"].tolist(),
                "inactive_source_vertex_rows": role_assets[role]["inactive_source_vertex_rows"].tolist(),
            } for role in ROLE_ORDER
        }
        return result

    owner.build_current_typed_volume_candidate_payload = build_current_typed_volume_candidate_payload
    original_write = owner.write_current_typed_volume_candidate_bundle

    def write_current_typed_volume_candidate_bundle(result, npz_path, node_map_path,
            ancestry_path, receipt_path, role_assets, input_pins):
        import numpy as np
        if CROSSWALK is None:
            raise ValueError("current typed candidate writer lacks full-domain 217 crosswalk")
        payload = dict(result["payload"])
        for role in ROLE_ORDER:
            asset = role_assets[role]
            payload[f"current_typed_role_active_source_vertex_rows_{role}"] = np.asarray(
                asset["active_source_vertex_rows"], dtype=np.int64)
            payload[f"current_typed_role_inactive_source_vertex_rows_{role}"] = np.asarray(
                asset["inactive_source_vertex_rows"], dtype=np.int64)
            payload[f"current_typed_role_source_vertex_to_compact_row_{role}"] = np.asarray(
                asset["source_vertex_to_compact_role_row"], dtype=np.int64)
            payload[f"current_typed_role_source_row_to_joint_node_{role}"] = np.asarray(
                CROSSWALK["source_row_to_joint_node"][role], dtype=np.int64)
        result = dict(result)
        result["payload"] = payload
        input_pins = dict(input_pins)
        input_pins["registered_surface_node_map_path"] = CROSSWALK["path"]
        input_pins["registered_surface_node_map_sha256"] = CROSSWALK["sha256"]
        receipt = original_write(result, npz_path, node_map_path, ancestry_path,
                                 receipt_path, role_assets, input_pins)
        receipt_path = pathlib.Path(receipt_path)
        receipt["role_source_point_rows"] = dict(result.get("role_source_row_counts", {}))
        receipt["current_typed_surface_row_domains"] = {
            "registered_surface_node_map": {
                "path": CROSSWALK["path"], "sha256": CROSSWALK["sha256"],
                "schema": CROSSWALK["schema"], "joint_row_base": 0,
                "node_map_csv_id_base": 1, "inactive_sentinel": -1,
            },
            "roles": {
                role: {
                    "full_source_point_count": int(len(role_assets[role]["source_points_world_m_f32"])),
                    "active_face_referenced_point_count": int(len(role_assets[role]["active_source_vertex_rows"])),
                    "inactive_unreferenced_point_count": int(len(role_assets[role]["inactive_source_vertex_rows"])),
                    "active_source_vertex_rows": role_assets[role]["active_source_vertex_rows"].tolist(),
                    "inactive_unreferenced_source_vertex_rows": role_assets[role]["inactive_source_vertex_rows"].tolist(),
                    "active_row_set_sha256": hashlib.sha256(
                        np.asarray(role_assets[role]["active_source_vertex_rows"], dtype="<i8").tobytes()).hexdigest(),
                    "inactive_row_set_sha256": hashlib.sha256(
                        np.asarray(role_assets[role]["inactive_source_vertex_rows"], dtype="<i8").tobytes()).hexdigest(),
                    "source_asset_path": role_assets[role]["path"],
                    "source_asset_sha256": role_assets[role]["sha256"],
                } for role in ROLE_ORDER
            },
            "policy": "only face-referenced current role rows enter the 217/433 physical volume node domain; inactive full-source rows remain in the pinned role NPZ and explicit row maps, with no invented volume node",
            "physical_admission": False,
        }
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return receipt

    owner.write_current_typed_volume_candidate_bundle = write_current_typed_volume_candidate_bundle
    _patch_export_for_original_surface_rows(owner)


def _install(owner):
    # 938 handoffs publish large maps and label tables as .csv.gz. Preserve
    # the pinned owner's CSV behavior for plain files and add streaming gzip.
    import csv
    import gzip

    original_read_csv = owner.read_csv

    def read_csv_compatible(path):
        path = pathlib.Path(path)
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        return original_read_csv(path)

    owner.read_csv = read_csv_compatible
    _install_original(owner)


def main() -> int:
    global CROSSWALK, CROSSWALK_PATH, CROSSWALK_SHA256
    original_argv = list(sys.argv)
    path = None
    sha = None
    cleaned = []
    i = 0
    while i < len(original_argv):
        item = original_argv[i]
        if item == "--current-surface-node-map":
            if path is not None or i + 2 >= len(original_argv):
                raise ValueError("provide one --current-surface-node-map PATH SHA256")
            path = pathlib.Path(original_argv[i + 1])
            sha = original_argv[i + 2]
            i += 3
            continue
        cleaned.append(item)
        i += 1
    current_typed = ("--build-current-typed-volume-candidate" in cleaned or
                     "--current-typed-volume-candidate" in cleaned)
    if current_typed:
        if path is None or sha is None:
            raise ValueError("current typed 433 mode requires --current-surface-node-map PATH SHA256")
        CROSSWALK = load_surface_crosswalk(path, sha)
        CROSSWALK_PATH = CROSSWALK["path"]
        CROSSWALK_SHA256 = CROSSWALK["sha256"]
    sys.argv = cleaned
    base_adapter = _load_1021()
    owner = base_adapter.load_pinned_433_owner()
    _install(owner)
    return int(owner.main())


if __name__ == "__main__":
    raise SystemExit(main())

