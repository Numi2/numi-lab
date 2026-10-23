#!/usr/bin/env python3
"""Rebind existing NHSKIN v5 weights without changing its reference surface.

This is source preparation, not physical admission. Stored normals stay byte
identical; regenerated coarse tissue frames, mass and pose gates remain required.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


class RebindError(ValueError):
    pass


def decode(raw):
    if len(raw) < 60:
        raise RebindError("truncated NHSKIN header")
    magic, abi, nb, nv, ni, fingerprint = struct.unpack_from("<8s5I", raw)
    if magic != b"NHSKIN1\0" or abi != 5 or nb == 0 or nv == 0 or ni % 3:
        raise RebindError("requires an existing NHSKIN v5 triangle surface")
    vo = 60 + nb * 36
    io = vo + nv * 56
    wo = io + ni * 4
    if len(raw) != wo + nv * nb * 4:
        raise RebindError("NHSKIN byte count does not match its header")
    bindings = np.frombuffer(raw, dtype=[("body", "<u4"), ("v", "<f4", (8,))],
                             count=nb, offset=60)
    vertices = np.frombuffer(raw, dtype="<f4", count=nv * 14, offset=vo).reshape(nv, 14)
    faces = np.frombuffer(raw, dtype="<u4", count=ni, offset=io).reshape(-1, 3)
    weights = np.frombuffer(raw, dtype="<f4", count=nv * nb, offset=wo).reshape(nv, nb)
    if len(set(map(int, bindings["body"]))) != nb:
        raise RebindError("duplicate binding body identity")
    if faces.size and int(faces.max()) >= nv:
        raise RebindError("face index out of range")
    if not np.isfinite(vertices[:, :6]).all() or not np.isfinite(bindings["v"]).all():
        raise RebindError("nonfinite geometry or binding")
    validate_weights(weights)
    return dict(raw=raw, nb=nb, nv=nv, ni=ni, fingerprint=fingerprint,
                source_sha256=raw[28:60], vertoff=vo, indexoff=io,
                weightoff=wo, binds=bindings, verts=vertices, inds=faces, weights=weights)


def validate_weights(w):
    if not np.isfinite(w).all() or np.any(w < 0):
        raise RebindError("weights must be finite and nonnegative")
    if np.max(np.abs(w.astype(float).sum(axis=1) - 1)) > 5e-7:
        raise RebindError("weight partition is not unity")


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.dot(q, q) - 1) > 1e-5:
        raise RebindError("invalid rigid quaternion")
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def reference_transforms(payload, pose, root_lift_m):
    lift = np.asarray(root_lift_m, dtype=float)
    if lift.shape != (3,) or not np.isfinite(lift).all():
        raise RebindError("root lift requires three finite metres")
    rows = pose["source_poses"]
    by_body = {int(p["core_body"]): p for p in rows}
    if len(rows) != len(by_body):
        raise RebindError("duplicate pose body identity")
    matrices, offsets = [], []
    for rec in payload["binds"]:
        body = int(rec["body"])
        if body not in by_body:
            raise RebindError("missing reference pose body " + str(body))
        p = by_body[body]
        vals = rec["v"].astype(float)
        r = rotation(p["quaternion_xyzw"])
        t = np.asarray(p["position"], dtype=float) + lift
        if not np.isfinite(t).all() or t.shape != (3,) or vals[7] <= 0:
            raise RebindError("invalid binding scale or reference translation")
        matrices.append(r @ (rotation(vals[3:7]) * vals[7]))
        offsets.append(t + r @ vals[:3])
    return np.asarray(matrices), np.asarray(offsets)


def positions(payload, a, b):
    w = payload["weights"].astype(float)
    return np.einsum("nk,kij,nj->ni", w, a, payload["verts"][:, :3].astype(float)) + w @ b


def replacement_weights(payload, other):
    if (payload["nb"], payload["nv"]) != (other["nb"], other["nv"]):
        raise RebindError("replacement weight dimensions differ")
    if (payload["fingerprint"] != other["fingerprint"] or
            payload["source_sha256"] != other["source_sha256"]):
        raise RebindError("replacement source ancestry or registration differs")
    if payload["binds"].tobytes() != other["binds"].tobytes():
        raise RebindError("replacement binding identities/transforms differ")
    return other["weights"].copy()


def local_rigid_weights(payload, target_body, seeds, shell_gates):
    gates = np.asarray(shell_gates, dtype=float)
    if (not len(gates) or not np.isfinite(gates).all() or
            np.any(gates < 0) or np.any(gates > 1)):
        raise RebindError("shell gates must be finite fractions")
    lanes = {int(b["body"]): i for i, b in enumerate(payload["binds"])}
    if target_body not in lanes:
        raise RebindError("target body has no skin binding")
    seeds = set(map(int, seeds))
    if not seeds or min(seeds) < 0 or max(seeds) >= payload["nv"]:
        raise RebindError("invalid local seed vertex")
    adjacency = [set() for _ in range(payload["nv"])]
    for a, b, c in payload["inds"]:
        for i, j in [(int(a), int(b)), (int(b), int(c)), (int(c), int(a))]:
            adjacency[i].add(j)
            adjacency[j].add(i)
    distance = {i: 0 for i in seeds}
    front = seeds
    for ring in range(1, len(gates)):
        front = {j for i in front for j in adjacency[i] if j not in distance}
        distance.update((j, ring) for j in front)
    w = payload["weights"].copy()
    rows = np.array(sorted(distance), dtype=int)
    gate = np.array([gates[distance[int(i)]] for i in rows])
    target = np.zeros((len(rows), payload["nb"]))
    target[:, lanes[target_body]] = 1
    w[rows] = (w[rows].astype(float) * (1 - gate[:, None]) +
               target * gate[:, None]).astype("<f4")
    return w


def rebind(payload, weights, a, b, maximum_error_m=2e-6):
    weights = np.asarray(weights, dtype="<f4")
    if weights.shape != payload["weights"].shape:
        raise RebindError("weight dimensions differ")
    validate_weights(weights)
    if (a.shape != (payload["nb"], 3, 3) or b.shape != (payload["nb"], 3)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise RebindError("invalid reference transforms")
    if not np.isfinite(maximum_error_m) or maximum_error_m <= 0:
        raise RebindError("reference error limit must be finite and positive")
    rows = np.flatnonzero(np.any(weights != payload["weights"], axis=1))
    if not len(rows):
        return payload["raw"], dict(changed_rows=[], reference_error_max_m=0.0)
    target = positions(payload, a, b)
    aa = np.einsum("nk,kij->nij", weights[rows].astype(float), a)
    bb = weights[rows].astype(float) @ b
    condition = np.linalg.cond(aa)
    if not np.isfinite(condition).all() or condition.max() >= 10:
        raise RebindError("singular or ill-conditioned inverse skinning")
    xyz = np.linalg.solve(aa, (target[rows] - bb)[..., None])[..., 0].astype("<f4")
    blob = bytearray(payload["raw"])
    for k, i in enumerate(rows):
        struct.pack_into("<3f", blob, payload["vertoff"] + int(i)*56, *map(float, xyz[k]))
        order = np.lexsort((np.arange(payload["nb"]), -weights[i].astype(float)))[:4]
        # The format always carries four lanes, even for tiny synthetic fixtures.
        if len(order) < 4:
            order = np.pad(order, (0, 4-len(order)), mode="constant")
            top = np.r_[weights[i, order[:payload["nb"]]], np.zeros(4-payload["nb"])]
        else:
            top = weights[i, order].astype(float)
        top = top.astype(float) / float(top.sum())
        struct.pack_into("<4I4f", blob, payload["vertoff"] + int(i)*56 + 24,
                         *map(int, order), *map(float, top))
        struct.pack_into("<" + str(payload["nb"]) + "f", blob,
                         payload["weightoff"] + int(i)*payload["nb"]*4,
                         *map(float, weights[i]))
    result = decode(bytes(blob))
    error = np.linalg.norm(positions(result, a, b) - target, axis=1)
    if not np.isfinite(error).all() or error.max() > maximum_error_m:
        raise RebindError("Float32 emission exceeds reference surface tolerance")
    return bytes(blob), dict(changed_rows=list(map(int, rows)),
                             reference_error_max_m=float(error.max()),
                             inverse_condition_max=float(condition.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--reference-pose", type=Path, required=True)
    parser.add_argument("--root-lift-m", type=float, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--weights-from", type=Path)
    mode.add_argument("--local-body", type=int)
    parser.add_argument("--seed-vertices", type=int, nargs="+")
    parser.add_argument("--shell-gates", type=float, nargs="+")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.input_sha256:
        raise RebindError("input hash mismatch")
    if args.output.exists() or args.receipt.exists():
        raise RebindError("refusing to overwrite source or evidence")
    p = decode(raw)
    if args.weights_from:
        w = replacement_weights(p, decode(args.weights_from.read_bytes()))
    else:
        if args.seed_vertices is None or args.shell_gates is None:
            parser.error("--local-body requires --seed-vertices and --shell-gates")
        w = local_rigid_weights(p, args.local_body, args.seed_vertices, args.shell_gates)
    a, b = reference_transforms(p, json.loads(args.reference_pose.read_text()), args.root_lift_m)
    result, report = rebind(p, w, a, b)
    args.output.write_bytes(result)
    report.update(input_sha256=args.input_sha256,
                  output_sha256=hashlib.sha256(result).hexdigest(),
                  reference_pose_sha256=hashlib.sha256(args.reference_pose.read_bytes()).hexdigest(),
                  root_lift_m=args.root_lift_m,
                  admission="SOURCE_PREPARATION_ONLY; physical mappings, mass and pose gates required")
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
