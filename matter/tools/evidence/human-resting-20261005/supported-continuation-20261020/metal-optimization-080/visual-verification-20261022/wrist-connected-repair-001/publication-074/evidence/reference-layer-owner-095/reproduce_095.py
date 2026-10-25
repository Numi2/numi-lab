#!/usr/bin/env python3
"""Reproduce the pinned, read-only measurements in reference-layer-owner-095."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/Users/n/Work/numi-human-connected-surface-069")
EVIDENCE = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
NODES = EVIDENCE / "wholebody-attachment-source-rebind-028-final080-omit-stale-inferred-hand-sites-001/generated/nodes.csv"
DONORS = EVIDENCE / "wholebody-attachment-source-rebind-028-final080-omit-stale-inferred-hand-sites-001/generated/donor-frames-fullweights.csv"
QUERY = EVIDENCE / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/exact-emitted.nodes.csv"
HEAD_REPORT = ROOT / "anatomy-feasibility-090/report.json"
TETS = EVIDENCE / "connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight/compiled-skin-fem-tets.csv"
INPUTS = {
    NODES: "5ae63af6c1f43fc38a53422452b685b6b1644c890b6b88d9d108cef64a208a11",
    DONORS: "3a1c4220172a5b4050e6502031c80d9c93ea23cee7d0a6e4a751fac6c2f7e4e7",
    QUERY: "99e58d6afb68e0939670f69d4132bfa730cf3e988a7f70a179a41a89eac3802c",
    HEAD_REPORT: "7ec30993a076c936204dfb816bca8f0e6c28f39140fa3ba8fcc3d3806e6ddf90",
    TETS: "a7dc2d97713e43ed2bfae2aa805bb5281fdf2361108cf208df4cbdea10084461",
}
HAND_BODY_IDS = {45,46,47,48,49,50,52,53,54,55,56,57,59,60,61,62,64,65,66,67,69,70,71,72,74,75,76,95,96,97,98,99,100,102,103,104,105,106,107,109,110,111,112,114,115,116,117,119,120,121,122,124,125,126}
FOREARM_BODY_IDS = {41,42,43,91,92,93}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        raise ValueError("empty measurement group")
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2

def stats_mm(xs):
    values = [float(x) * 1000.0 for x in xs]
    return {"count": len(values), "min_mm": min(values), "median_mm": median(values), "max_mm": max(values)}

def dist(a, b):
    return math.sqrt(math.fsum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

def main():
    pins = {}
    for path, expected in INPUTS.items():
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"input pin mismatch: {path}: expected {expected}, got {actual}")
        pins[str(path)] = actual

    nodes = read_csv(NODES)
    donors = read_csv(DONORS)
    queries = read_csv(QUERY)
    head_report = json.loads(HEAD_REPORT.read_text())
    tets = read_csv(TETS)
    if len(nodes) != 2411 or len(donors) != 2411 or len(queries) != 7233 or len(tets) != 28632:
        raise SystemExit("input row-count mismatch")
    node_by_row = {int(r["row"]): r for r in nodes}
    donor_by_row = {int(r["row"]): int(r["donor"]) for r in donors}
    if set(node_by_row) != set(range(2411)) or set(donor_by_row) != set(range(2411)):
        raise SystemExit("source rows are not the expected contiguous 0..2410 domain")

    q = {}
    for r in queries:
        layer, row = int(r["layer"]), int(r["row"])
        if layer not in (0, 1, 2) or row in q.setdefault(layer, {}):
            raise SystemExit("invalid or duplicate emitted layer row")
        q[layer][row] = tuple(float(r[k]) for k in ("x_m", "y_m", "z_m"))
    if any(set(q[layer]) != set(range(2411)) for layer in (0, 1, 2)):
        raise SystemExit("emitted query does not cover all 2411 rows in all three layers")

    rest = {
        layer: {
            int(r["row"]): tuple(float(r[k]) for k in keys)
            for r in nodes
        }
        for layer, keys in ((0, ("ox","oy","oz")), (1, ("ix","iy","iz")), (2, ("dx","dy","dz")))
    }
    skin_field = [float(r["skin_thickness_m"]) for r in nodes]
    fat_field = [float(r["fat_thickness_m"]) for r in nodes]
    selected_rows = [
        int(r["source_column_row"])
        for r in head_report["head_endpoint_counts_iteration003"]["ordinary_columns"]
        if r["endpoint_status"]["D_deep"] == "inside"
    ]
    if len(selected_rows) != 75:
        raise SystemExit(f"expected 75 selected head D-inside rows, got {len(selected_rows)}")
    if any(donor_by_row[row] != 23 for row in selected_rows):
        raise SystemExit("selected head witness rows do not all map to body index 23")

    def group(body_ids):
        rows = [i for i, donor in donor_by_row.items() if donor in body_ids]
        return {
            "rows": len(rows),
            "authored_rest_field_OI_mm": stats_mm([skin_field[i] for i in rows]),
            "authored_rest_field_ID_mm": stats_mm([fat_field[i] for i in rows]),
            "authored_rest_geometry_OI_mm": stats_mm([dist(rest[0][i], rest[1][i]) for i in rows]),
            "authored_rest_geometry_ID_mm": stats_mm([dist(rest[1][i], rest[2][i]) for i in rows]),
            "emitted_current_OI_mm": stats_mm([dist(q[0][i], q[1][i]) for i in rows]),
            "emitted_current_ID_mm": stats_mm([dist(q[1][i], q[2][i]) for i in rows]),
        }

    by_material = {}
    for material in (15, 16):
        rows = [r for r in tets if int(r["material_index"]) == material]
        if len(rows) != 14316:
            raise SystemExit(f"unexpected cell count for material {material}")
        vols = [float(r["inverse_rest_w_m3"]) for r in rows]
        masses = [v * float(r["compiled_bulk_x"]) for v, r in zip(vols, rows)]
        densities = {float(r["compiled_bulk_x"]) for r in rows}
        if len(densities) != 1:
            raise SystemExit(f"unexpected compiled density variation for material {material}")
        by_material[str(material)] = {
            "cell_count": len(rows),
            "reference_volume_m3": math.fsum(vols),
            "compiled_density_kg_m3": densities.pop(),
            "compiled_mass_kg": math.fsum(masses),
        }

    result = {
        "status": "REPRODUCED_FROM_HASH_PINNED_INPUTS",
        "inputs_sha256": pins,
        "field_ranges_mm": {
            "row_count": len(nodes),
            "skin": stats_mm(skin_field),
            "fat": stats_mm(fat_field),
        },
        "head_selected_iteration003_D_inside": {
            "row_count": len(selected_rows),
            "body_index": 23,
            "rest_field_OI_mm": stats_mm([skin_field[i] for i in selected_rows]),
            "rest_field_ID_mm": stats_mm([fat_field[i] for i in selected_rows]),
            "rest_geometry_OI_mm": stats_mm([dist(rest[0][i], rest[1][i]) for i in selected_rows]),
            "rest_geometry_ID_mm": stats_mm([dist(rest[1][i], rest[2][i]) for i in selected_rows]),
            "current_OI_mm": stats_mm([dist(q[0][i], q[1][i]) for i in selected_rows]),
            "current_ID_mm": stats_mm([dist(q[1][i], q[2][i]) for i in selected_rows]),
        },
        "regional_by_donor_body_index": {
            "head_23": group({23}),
            "bilateral_arm_bone_donors_41_42_43_91_92_93": group(FOREARM_BODY_IDS),
            "bilateral_hands": group(HAND_BODY_IDS),
        },
        "compiled_native_mass_by_material": by_material,
        "interpretation": "Current emitted distances are posed query spacing, not authored reference thickness or anatomical measurements."
    }
    target = ROOT / "D/reference-layer-owner-095/reproduction-output.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
