#!/usr/bin/env python3
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools" / "rebind_human_tissue_reference.py"
spec = importlib.util.spec_from_file_location("rebind_human_tissue_reference_396", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _rotate(q, p):
    q = np.asarray(q, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    x, y, z = np.moveaxis(p, -1, 0)
    a, b, c, w = q
    u = 2 * (b * z - c * y)
    v = 2 * (c * x - a * z)
    t = 2 * (a * y - b * x)
    return np.stack([x + ((w * u + b * t) - c * v),
                     y + ((w * v + c * u) - a * t),
                     z + ((w * t + a * v) - b * u)], axis=-1).astype(np.float32)


def _bodyworld(body, p):
    state = _POSES[int(body)]
    return (np.asarray(state["position_world_m"], dtype=np.float32) +
            _rotate(state["orientation_world_xyzw"], p)).astype(np.float32)


def _quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = angle * 0.5
    return np.asarray([*(axis * np.sin(half)), np.cos(half)], dtype=np.float32)


def _make_fixture(tmp: Path):
    global _POSES
    nr, per_record_bindings, per_record_vertices = 9, 2, 3
    nb, nv, ni = nr * per_record_bindings, nr * per_record_vertices, nr * 3
    registration = 77
    source_id = bytes(range(32))
    records = np.zeros((nr, 8), dtype="<u4")
    bindings = np.zeros(nb, dtype=module.BIND_DTYPE)
    vertices = np.zeros(nv, dtype=module.VERT_DTYPE)
    indices = np.zeros(ni, dtype="<u4")
    pose_rows = []
    _POSES = {}
    for r in range(nr):
        body_a, body_b = 1000 + 2 * r, 1001 + 2 * r
        fb, fv, fi = 2 * r, 3 * r, 3 * r
        records[r] = [fb, 2, fv, 3, fi, 3, 100 + r, 1]
        for lane, body in enumerate((body_a, body_b)):
            # Distinct input bind rotations/translations; selected output must
            # replace these with coherent inverses of the pinned current pose.
            bind_q = _quat([1 + lane, 2, 3], 0.03 * (r + lane + 1))
            bindings[fb + lane]["body"] = body
            bindings[fb + lane]["transform"] = np.asarray(
                [0.001 * r, -0.002 * lane, 0.0005, *bind_q, 0.001], dtype=np.float32)
            current_q = _quat([1, 2 + lane, 3], 0.08 * (r + lane + 1))
            current_p = np.asarray([1.2 + 0.01 * r, -0.8 + 0.02 * lane, 0.4 - 0.003 * r], dtype=np.float32)
            state = {"position_world_m": current_p.tolist(),
                     "orientation_world_xyzw": current_q.tolist()}
            _POSES[body] = state
            pose_rows.append({"body_index": body, "current": state})
        local = np.asarray([[0, 0, 0], [23, 0, 0], [0, 17, 0]], dtype=np.float32)
        local += np.asarray([r * 3, -r, 2 * r], dtype=np.float32)
        for j in range(3):
            i = fv + j
            vertices[i]["p"] = local[j]
            vertices[i]["n"] = np.asarray([0, 0, 1], dtype=np.float32)
            vertices[i]["binding"] = np.asarray([0, 1, 0, 0], dtype=np.uint32)
            vertices[i]["w"] = np.asarray([0.5, 0.5, 0, 0], dtype=np.float32)
        indices[fi:fi + 3] = np.asarray([fv, fv + 1, fv + 2], dtype=np.uint32)
    pose_data = {"registration_fingerprint32": registration, "all_bodies": pose_rows}
    header = module.HEADER.pack(b"NHTISS4\0", 5, nr, nb, nv, ni, registration, source_id)
    raw = header + records.tobytes() + bindings.tobytes() + vertices.tobytes() + indices.tobytes()
    source = module.decode_nhtiss(raw)
    pose_map = module.load_pose_map(pose_data, registration)
    cache_arrays = {}
    banks = []
    for r, rec in enumerate(records):
        sid = int(rec[6])
        p, _ = module.emit_current(source, rec, pose_map, _rotate, _bodyworld)
        key = f"bank_{r:04d}"
        cache_arrays[key + "_vertices"] = p
        cache_arrays[key + "_faces"] = np.asarray([[0, 1, 2]], dtype=np.uint32)
        banks.append({"kind": "soft_tissue", "stable_id": sid, "key": key,
                      "vertices": 3, "triangles": 1})
    baseline_path = tmp / "baseline.nhtissue"
    weights_path = tmp / "weights.nhtissue"
    pose_path = tmp / "pose.json"
    cache_path = tmp / "cache.npz"
    inventory_path = tmp / "inventory.json"
    emitter_path = tmp / "emitter.py"
    baseline_path.write_bytes(raw)
    candidate = bytearray(raw)
    candidate_data = module.decode_nhtiss(raw)
    for r in range(8):
        for j in range(3):
            vertex = 3 * r + j
            candidate_data["vertices"]["w"][vertex] = np.asarray([0.25, 0.75, 0, 0], dtype=np.float32)
            struct.pack_into("<4f", candidate, candidate_data["vertex_offset"] + vertex * 56 + 40,
                             0.25, 0.75, 0.0, 0.0)
    weights_path.write_bytes(bytes(candidate))
    pose_path.write_text(json.dumps(pose_data, sort_keys=True) + "\n")
    np.savez(cache_path, **cache_arrays)
    baseline_sha = hashlib.sha256(raw).hexdigest()
    pose_sha = hashlib.sha256(pose_path.read_bytes()).hexdigest()
    cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    inventory = {"pose_sha256": pose_sha, "cache_sha256": cache_sha,
                 "inputs": [{"kind": "soft_tissue", "sha256": baseline_sha, "record_count": nr}],
                 "banks": banks}
    inventory_path.write_text(json.dumps(inventory, sort_keys=True) + "\n")
    # This deliberately mirrors the exact helper expressions loaded from 191.
    emitter_path.write_text(
        "import numpy as np\n"
        "def rotate(q,p):\n"
        " q=np.asarray(q,dtype=np.float32); p=np.asarray(p,dtype=np.float32)\n"
        " x,y,z=np.moveaxis(p,-1,0); a,b,c,w=q\n"
        " u=2*(b*z-c*y);v=2*(c*x-a*z);t=2*(a*y-b*x)\n"
        " return np.stack([x+((w*u+b*t)-c*v),y+((w*v+c*u)-a*t),z+((w*t+a*v)-b*u)],axis=-1).astype(np.float32)\n"
        "def bodyworld(body,p):\n"
        " pose=poses[int(body)]\n"
        " return (np.asarray(pose['position_world_m'],dtype=np.float32)+rotate(pose['orientation_world_xyzw'],p)).astype(np.float32)\n"
    )
    return {"baseline": baseline_path, "weights": weights_path, "pose": pose_path,
            "cache": cache_path, "inventory": inventory_path, "emitter": emitter_path,
            "pose_data": pose_data, "cache_arrays": cache_arrays, "inventory_data": inventory,
            "baseline_raw": raw, "weights_raw": bytes(candidate), "source": source,
            "selected_ids": list(range(100, 108))}


def _args_from_fixture(fx, out, report):
    values = {
        "--baseline": fx["baseline"], "--baseline-sha256": hashlib.sha256(fx["baseline"].read_bytes()).hexdigest(),
        "--weights": fx["weights"], "--weights-sha256": hashlib.sha256(fx["weights"].read_bytes()).hexdigest(),
        "--pose": fx["pose"], "--pose-sha256": hashlib.sha256(fx["pose"].read_bytes()).hexdigest(),
        "--inventory": fx["inventory"], "--inventory-sha256": hashlib.sha256(fx["inventory"].read_bytes()).hexdigest(),
        "--cache": fx["cache"], "--cache-sha256": hashlib.sha256(fx["cache"].read_bytes()).hexdigest(),
        "--emitter": fx["emitter"], "--emitter-sha256": hashlib.sha256(fx["emitter"].read_bytes()).hexdigest(),
        "--inferred-reference-label": "inferred common registered reference from pinned test current pose",
        "--output": out, "--report": report,
    }
    argv = []
    for flag, value in values.items():
        argv.extend([flag, str(value)])
    for sid in fx["selected_ids"]:
        argv.extend(["--stable-id", str(sid)])
    return argv


class CoherentReferenceTests(unittest.TestCase):
    def test_centered_translation_rotation_cancellation_and_eight_bank_byte_scope(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fx = _make_fixture(tmp)
            out, report = module.construct_candidate(
                fx["baseline_raw"], fx["weights_raw"], stable_ids=fx["selected_ids"],
                pose_data=fx["pose_data"], inventory=fx["inventory_data"], cache=fx["cache_arrays"],
                baseline_sha256=hashlib.sha256(fx["baseline_raw"]).hexdigest(),
                pose_sha256=hashlib.sha256(fx["pose"].read_bytes()).hexdigest(),
                inventory_sha256=hashlib.sha256(fx["inventory"].read_bytes()).hexdigest(),
                cache_sha256=hashlib.sha256(fx["cache"].read_bytes()).hexdigest(),
                emitter_sha256=hashlib.sha256(fx["emitter"].read_bytes()).hexdigest(),
                reference_label="inferred registered-world frame test", rotate=_rotate, bodyworld=_bodyworld)
            parsed = module.decode_nhtiss(out)
            self.assertEqual(report["gate"]["physical_admission"], False)
            self.assertEqual(report["provenance"]["reference_kind"], "inferred")
            self.assertEqual(report["scope"]["stable_surface_ids"], fx["selected_ids"])
            for sid in fx["selected_ids"]:
                self.assertLessEqual(report["per_surface"][str(sid)]["current_position_delta_max_m"], 1e-6)
                self.assertLessEqual(report["per_surface"][str(sid)]["per_binding_position_spread_max_m"], 1e-6)
            # Every byte outside selected binding transforms and selected p/n is
            # copied from the proposed-weight input. This includes the ninth bank.
            original = fx["weights_raw"]
            permitted = np.zeros(len(original), dtype=bool)
            source = module.decode_nhtiss(original)
            for row in source["records"]:
                fb, bc, fv, vc, _fi, _ic, sid, _layer = map(int, row)
                if sid not in fx["selected_ids"]:
                    continue
                for bi in range(fb, fb + bc):
                    permitted[source["binding_offset"] + bi * 36 + 4:source["binding_offset"] + (bi + 1) * 36] = True
                for vi in range(fv, fv + vc):
                    permitted[source["vertex_offset"] + vi * 56:source["vertex_offset"] + vi * 56 + 24] = True
            changed = np.frombuffer(original, dtype=np.uint8) != np.frombuffer(out, dtype=np.uint8)
            self.assertFalse(np.any(changed & ~permitted))
            self.assertTrue(np.array_equal(parsed["vertices"]["w"], source["vertices"]["w"]))
            self.assertTrue(np.array_equal(parsed["vertices"]["binding"], source["vertices"]["binding"]))

    def test_bad_hash_pin_creates_no_output(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fx = _make_fixture(tmp)
            output, report = tmp / "must-not-exist.nhtissue", tmp / "must-not-exist.json"
            args = _args_from_fixture(fx, output, report)
            args[args.index("--baseline-sha256") + 1] = "0" * 64
            with contextlib.redirect_stderr(io.StringIO()):
                status = module.main(args)
            self.assertEqual(status, 2)
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())

    def test_zero_and_nonunit_quaternions_fail_without_output(self):
        for bad_quaternion in ([0, 0, 0, 0], [0, 0, 0, 2]):
            with self.subTest(quaternion=bad_quaternion), tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                fx = _make_fixture(tmp)
                pose = json.loads(fx["pose"].read_text())
                pose["all_bodies"][0]["current"]["orientation_world_xyzw"] = bad_quaternion
                bad_pose_path = tmp / "invalid-pose.json"
                bad_pose_path.write_text(json.dumps(pose, sort_keys=True) + "\n")
                inv = dict(fx["inventory_data"])
                inv["pose_sha256"] = hashlib.sha256(bad_pose_path.read_bytes()).hexdigest()
                bad_inv_path = tmp / "invalid-pose-inventory.json"
                bad_inv_path.write_text(json.dumps(inv, sort_keys=True) + "\n")
                output, report = tmp / "must-not-exist.nhtissue", tmp / "must-not-exist.json"
                args = _args_from_fixture(fx, output, report)
                args[args.index("--pose") + 1] = str(bad_pose_path)
                args[args.index("--pose-sha256") + 1] = hashlib.sha256(bad_pose_path.read_bytes()).hexdigest()
                args[args.index("--inventory") + 1] = str(bad_inv_path)
                args[args.index("--inventory-sha256") + 1] = hashlib.sha256(bad_inv_path.read_bytes()).hexdigest()
                with contextlib.redirect_stderr(io.StringIO()):
                    status = module.main(args)
                self.assertEqual(status, 2)
                self.assertFalse(output.exists())
                self.assertFalse(report.exists())

    @unittest.skipUnless(all(os.environ.get(key) for key in (
        "NUMI_NHTISS_REPLAY_BASELINE", "NUMI_NHTISS_REPLAY_BASELINE_SHA256",
        "NUMI_NHTISS_REPLAY_WEIGHTS", "NUMI_NHTISS_REPLAY_WEIGHTS_SHA256",
        "NUMI_NHTISS_REPLAY_POSE", "NUMI_NHTISS_REPLAY_POSE_SHA256",
        "NUMI_NHTISS_REPLAY_INVENTORY", "NUMI_NHTISS_REPLAY_INVENTORY_SHA256",
        "NUMI_NHTISS_REPLAY_CACHE", "NUMI_NHTISS_REPLAY_CACHE_SHA256",
        "NUMI_NHTISS_REPLAY_EMITTER", "NUMI_NHTISS_REPLAY_EMITTER_SHA256",
        "NUMI_NHTISS_REPLAY_STABLE_IDS", "NUMI_NHTISS_REPLAY_EXPECTED",
        "NUMI_NHTISS_REPLAY_EXPECTED_SHA256")), "set NUMI_NHTISS_REPLAY_* pins to replay frozen 393")
    def test_optional_retained_393_byte_exact_replay(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            args = [
                "--baseline", os.environ["NUMI_NHTISS_REPLAY_BASELINE"],
                "--baseline-sha256", os.environ["NUMI_NHTISS_REPLAY_BASELINE_SHA256"],
                "--weights", os.environ["NUMI_NHTISS_REPLAY_WEIGHTS"],
                "--weights-sha256", os.environ["NUMI_NHTISS_REPLAY_WEIGHTS_SHA256"],
                "--pose", os.environ["NUMI_NHTISS_REPLAY_POSE"],
                "--pose-sha256", os.environ["NUMI_NHTISS_REPLAY_POSE_SHA256"],
                "--inventory", os.environ["NUMI_NHTISS_REPLAY_INVENTORY"],
                "--inventory-sha256", os.environ["NUMI_NHTISS_REPLAY_INVENTORY_SHA256"],
                "--cache", os.environ["NUMI_NHTISS_REPLAY_CACHE"],
                "--cache-sha256", os.environ["NUMI_NHTISS_REPLAY_CACHE_SHA256"],
                "--emitter", os.environ["NUMI_NHTISS_REPLAY_EMITTER"],
                "--emitter-sha256", os.environ["NUMI_NHTISS_REPLAY_EMITTER_SHA256"],
                "--inferred-reference-label", "inferred registered-world reference replay of frozen 393",
                "--output", str(tmp / "replay.nhtissue"), "--report", str(tmp / "replay.json"),
            ]
            args += sum((["--stable-id", sid.strip()] for sid in os.environ["NUMI_NHTISS_REPLAY_STABLE_IDS"].split(",") if sid.strip()), [])
            with contextlib.redirect_stdout(io.StringIO()):
                status = module.main(args)
            self.assertEqual(status, 0)
            expected = Path(os.environ["NUMI_NHTISS_REPLAY_EXPECTED"])
            self.assertEqual(hashlib.sha256(expected.read_bytes()).hexdigest(), os.environ["NUMI_NHTISS_REPLAY_EXPECTED_SHA256"])
            self.assertEqual((tmp / "replay.nhtissue").read_bytes(), expected.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
