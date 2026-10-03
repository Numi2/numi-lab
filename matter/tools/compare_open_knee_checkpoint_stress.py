#!/usr/bin/env python3
"""Compare the assembled knee's first FEM force evaluation with XPLT state 1.

This is a rejected-root diagnostic. It never qualifies a Matter checkpoint.
The three arguments must be the pinned source volume mesh, exported state-1
NPZ, and first-assembly element-force dump from the full coupled runner.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


EXPECTED_SOURCE = (
    "39b86f2f55853c74968f36a8d6c67eaed94639a3d42b558bb17787e2af8ffdba",
    "de0ab6ec1de1a421187027beca0153b9b5768af14f90d19802c794e72cc10fb4",
)
EXPECTED_FORCE_DUMPS = {
    "83c7c8e28db0ff172443212d4d693327dcb5a7be36c838d6c157d51d2cbcc654": "rounded_position_baseline",
    "86be1d86771cc5dfceef021151a3a0dca5f4e75ef8b4bd40bdad1902b0ed6c4d": "split_position_candidate",
    "1d7e4c571a6f7a561b7f8d4ecba0b5f90117c8848fca04b1bc9f6c7336918d90": "split_position_incremental_tie_candidate",
}
EXPECTED_TIES = "cb06f3e1d566ac5d551c17daea6bb222e74c0c84ac33111e02f9e5f0f5d83221"
DOMAINS = (1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 16)
FIBROUS = {
    10: (8, 1.0017, (.2862545445153689, -.03912610167242837,
                      .9573544191741205)),
    14: (12, 1.00135, (.22214778048527417, .16495808684018612,
                        .9609574356918686)),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(value):
    return float(np.sqrt(np.mean(value * value)))


def source_stress(reference, current, axial, fiber):
    """Pinned uncoupled trans-iso source law for MCL/LCL at time 0.05, MPa."""
    a = np.asarray(fiber, dtype=np.float64)
    a /= np.linalg.norm(a)
    rest_edges = np.stack([reference[:, i] - reference[:, 0]
                           for i in (1, 2, 3)], axis=2)
    current_edges = np.stack([current[:, i] - current[:, 0]
                              for i in (1, 2, 3)], axis=2)
    f = current_edges @ np.linalg.inv(rest_edges)
    fp = np.eye(3) / np.sqrt(axial) + (axial - 1 / np.sqrt(axial)) * np.outer(a, a)
    fe = f @ fp
    j = np.linalg.det(fe)
    b = fe @ np.transpose(fe, (0, 2, 1)) * j[:, None, None] ** (-2 / 3)
    matrix = 2 * 1.44 / j[:, None, None] * (
        b - np.trace(b, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3)
    v = np.einsum("eij,j->ei", fe, a)
    length = np.linalg.norm(v, axis=1)
    lam = length * j ** (-1 / 3)
    q = np.where(lam < 1, 0, np.where(lam < 1.063,
        .57 * np.expm1(48 * (lam - 1)),
        .57 * np.expm1(48 * .063) + 467.1 * (lam - 1.063)))
    unit = v / length[:, None]
    tension = q[:, None, None] / j[:, None, None] * (
        unit[:, :, None] * unit[:, None, :] - np.eye(3) / 3)
    pressure = 793.65 * np.log(j) / j
    stress = matrix + tension + pressure[:, None, None] * np.eye(3)
    return stress[:, (0, 1, 2, 0, 1, 0), (0, 1, 2, 1, 2, 2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_volume_mesh", type=Path)
    parser.add_argument("xplt_state_1_npz", type=Path)
    parser.add_argument("matter_first_element_forces", type=Path)
    parser.add_argument("--source-rigid-ties", type=Path)
    args = parser.parse_args()
    paths = (args.source_volume_mesh, args.xplt_state_1_npz,
             args.matter_first_element_forces)
    hashes = tuple(sha256(path) for path in paths)
    if hashes[:2] != EXPECTED_SOURCE or hashes[2] not in EXPECTED_FORCE_DUMPS:
        parser.error("source mesh, XPLT checkpoint, or Matter force dump hash changed")
    tied_nodes = None
    if args.source_rigid_ties is not None:
        if sha256(args.source_rigid_ties) != EXPECTED_TIES:
            parser.error("source rigid ties hash changed")
        tie_bytes = args.source_rigid_ties.read_bytes()
        if tie_bytes[:8] != b"NHTIES1\0" or len(tie_bytes) != 88 + 29427 * 16:
            parser.error("source rigid ties layout changed")
        tied_nodes = np.frombuffer(tie_bytes, dtype="<u4", offset=88).reshape(-1, 4)[:, 0]

    checkpoint = np.load(paths[1])
    displacement = checkpoint["node_01"].reshape(-1, 3).astype(np.float64)
    forces = np.memmap(paths[2], dtype="<f4", mode="r").reshape(-1, 4, 4)
    source = paths[0].read_bytes()
    offset = first = 0
    result = {"status": "rejected_root_diagnostic_only", "hashes": hashes,
              "force_dump_variant": EXPECTED_FORCE_DUMPS[hashes[2]],
              "source_rigid_ties_sha256": EXPECTED_TIES if tied_nodes is not None else None,
              "stress_component_order": "xx yy zz xy yz xz", "tissues": []}
    for group, domain in enumerate(DOMAINS):
        magic, version, material, nnode, ntet = struct.unpack_from("<4s4I", source, offset)
        if (magic, version, material) != (b"NOKT", 1, group + 5):
            raise ValueError("source volume group order changed")
        offset += 20
        node_dtype = np.dtype([("id", "<u4"), ("xyz", "<f8", 3)])
        tet_dtype = np.dtype([("id", "<u4"), ("node", "<u4", 4)])
        nodes = np.frombuffer(source, dtype=node_dtype, count=nnode, offset=offset)
        offset += 28 * nnode
        tets = np.frombuffer(source, dtype=tet_dtype, count=ntet, offset=offset)
        offset += 20 * ntet
        local = np.searchsorted(nodes["id"], tets["node"])
        if not np.array_equal(nodes["id"][local], tets["node"]):
            raise ValueError("source tet references an unknown node")
        reference = nodes["xyz"][local]
        current = reference + displacement[tets["node"] - 1]
        edges = np.stack([current[:, i] - current[:, 0] for i in (1, 2, 3)], axis=2)
        volume = np.abs(np.linalg.det(edges)) / 6
        element_forces = forces[first:first + ntet, :, :3].astype(np.float64)
        stress = -np.einsum("eai,eaj->eij", element_forces, current) / volume[:, None, None]
        native = stress[:, (0, 1, 2, 0, 1, 0), (0, 1, 2, 1, 2, 2)]
        archived = checkpoint[f"element_01_domain_{domain:02d}"].reshape(-1, 6).astype(np.float64)
        row = {"material": material, "domain": domain, "elements": ntet,
               "xplt_rms_mpa": rms(archived),
               "matter_first_assembly_difference_rms_mpa": rms(native - archived)}
        if tied_nodes is not None:
            touches_tie = np.isin(tets["node"], tied_nodes).any(axis=1)
            row["elements_touching_rigid_tie"] = int(touches_tie.sum())
            row["tied_difference_rms_mpa"] = (rms((native - archived)[touches_tie])
                                               if touches_tie.any() else None)
            row["untied_difference_rms_mpa"] = (rms((native - archived)[~touches_tie])
                                                 if (~touches_tie).any() else None)
        if material in FIBROUS:
            _, axial, fiber = FIBROUS[material]
            exact = source_stress(reference, current, axial, fiber)
            rounded_reference = (reference * .001).astype(np.float32).astype(np.float64) * 1000
            rounded_current = (current * .001).astype(np.float32).astype(np.float64) * 1000
            rounded = source_stress(rounded_reference, rounded_current, axial, fiber)
            row["double_source_law_difference_rms_mpa"] = rms(exact - archived)
            row["float32_coordinate_difference_rms_mpa"] = rms(rounded - archived)
        result["tissues"].append(row)
        first += ntet
    if offset != len(source) or first != len(forces):
        raise ValueError("source mesh or Matter force dump has trailing data")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
