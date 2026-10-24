#!/usr/bin/env python3
import importlib.util
import struct
import subprocess
import tempfile
import json
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "skin_rebind", Path(__file__).resolve().parents[1] / "tools/rebind_human_skin_reference.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(nb=4):
    nv, ni = 5, 6
    raw = bytearray(60 + nb*36 + nv*56 + ni*4 + nv*nb*4)
    struct.pack_into("<8s5I", raw, 0, b"NHSKIN1\0", 5, nb, nv, ni, 123)
    for i in range(nb):
        struct.pack_into("<I8f", raw, 60+i*36, 10+i, 0, 0, 0, 0, 0, 0, 1, 1)
    vo = 60+nb*36
    for i, xyz in enumerate([(0,0,0), (1,0,0), (0,1,0), (1,1,0), (10,10,0)]):
        struct.pack_into("<6f4I4f", raw, vo+i*56, *xyz, 0, 0, 1,
                         0, 0, 0, 0, 1, 0, 0, 0)
    io = vo+nv*56
    struct.pack_into("<6I", raw, io, 0, 1, 2, 1, 3, 2)
    for i in range(nv):
        struct.pack_into("<" + str(nb) + "f", raw, io+ni*4+i*nb*4,
                         *([1]+[0]*(nb-1)))
    return m.decode(bytes(raw))


class RebindTests(unittest.TestCase):

    def _registered_snapshot(self, payload):
        rows = [dict(body_index=int(binding["body"]),
                     current=dict(position_world_m=[1,2,3],
                                  orientation_world_xyzw=[0,0,1,0]),
                     rest=dict(position_world_m=[0,0,0],
                               orientation_world_xyzw=[0,0,0,1]))
                for binding in payload["binds"]]
        return dict(registration_fingerprint32=payload["fingerprint"],
                    binding_count=payload["nb"], vertex_count=payload["nv"],
                    payload_abi=5, all_bodies=rows)

    def test_selected_normals_use_native_rest_not_current_frame(self):
        payload = fixture()
        snap = self._registered_snapshot(payload)
        # Current X rotation reverses the visible plane; ABI5 stores REST normals.
        for row in snap["all_bodies"]:
            row["current"]["orientation_world_xyzw"] = [1,0,0,0]
        out = m.registered_rest_vertex_normals_float32(payload, snap, [3,0,1])
        np.testing.assert_array_equal(out, [[0,0,1]]*3)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(payload["verts"], fixture()["verts"])
        # A rest rotation must transform the constructed normals.
        for row in snap["all_bodies"]:
            row["rest"]["orientation_world_xyzw"] = [1,0,0,0]
        np.testing.assert_array_equal(
            m.registered_rest_vertex_normals_float32(payload, snap, [0,3]),
            [[0,0,-1]]*2)

    def test_patch_normals_include_retained_incident_faces(self):
        payload = fixture()
        verts = payload["verts"].copy()
        verts[3,2] = 1
        payload = dict(payload, verts=verts)
        snap = self._registered_snapshot(payload)
        out = m.registered_rest_vertex_normals_float32(payload, snap, [1])
        expected = np.array([[-1,-1,2]],dtype=float)
        expected /= np.linalg.norm(expected,axis=1)[:,None]
        np.testing.assert_allclose(out,expected,rtol=1e-7)
        # Restricting accumulation to the first changed face would return +Z.
        self.assertFalse(np.array_equal(out,[[0,0,1]]))

    def test_rest_normals_reject_invalid_and_unreferenced_rows(self):
        payload = fixture()
        snap = self._registered_snapshot(payload)
        for rows in ([4], [0,0], [-1], [5], [0.5]):
            with self.subTest(rows=rows), self.assertRaises(m.RebindError):
                m.registered_rest_vertex_normals_float32(payload,snap,rows)
        collapsed = dict(payload, verts=payload["verts"].copy())
        collapsed["verts"][2,:3] = collapsed["verts"][1,:3]
        with self.assertRaisesRegex(m.RebindError,"degenerate"):
            m.registered_rest_vertex_normals_float32(collapsed,snap,[0])
        empty = m.registered_rest_vertex_normals_float32(payload,snap,np.array([],dtype=int))
        self.assertEqual(empty.shape,(0,3))

    def test_registered_frames_are_explicit_and_distinct(self):
        payload = fixture()
        snapshot = self._registered_snapshot(payload)
        rest = m.registered_positions_float32(payload, snapshot, frame="rest")
        posed = m.registered_positions_float32(payload, snapshot, frame="current")
        np.testing.assert_array_equal(rest, payload["verts"][:, :3])
        expected = rest * np.array([-1,-1,1], dtype=np.float32) + [1,2,3]
        np.testing.assert_array_equal(posed, expected)
        self.assertEqual(posed.dtype, np.float32)
        for frame in ("neutral", "", None):
            with self.subTest(frame=frame), self.assertRaises(m.RebindError):
                m.registered_positions_float32(payload, snapshot, frame=frame)
        with self.assertRaises(TypeError):
            m.registered_positions_float32(payload, snapshot)

    def test_registered_binding_and_weight_order(self):
        payload = fixture()
        payload = dict(payload, binds=payload["binds"].copy(),
                       weights=payload["weights"].copy())
        payload["binds"]["v"][:, :3] = [0.25,-0.5,0.125]
        payload["binds"]["v"][:, 7] = 2
        payload["weights"][:] = [0.25,0.75,0,0]
        snapshot = self._registered_snapshot(payload)
        snapshot["all_bodies"][1]["current"]["position_world_m"] = [5,2,3]
        out = m.registered_positions_float32(payload, snapshot, frame="current")
        local = payload["verts"][:,:3]*2 + [0.25,-0.5,0.125]
        expected = local*[-1,-1,1] + [4,2,3]
        np.testing.assert_array_equal(out, expected)
        # Snapshot order may differ; binding order is authoritative.
        snapshot["all_bodies"].reverse()
        np.testing.assert_array_equal(out,
            m.registered_positions_float32(payload, snapshot, frame="current"))

    def test_registered_snapshot_rejects_stale_and_missing_frames(self):
        payload = fixture()
        for key in ("registration_fingerprint32", "binding_count",
                    "vertex_count", "payload_abi"):
            snap = self._registered_snapshot(payload)
            snap[key] += 1
            with self.subTest(key=key), self.assertRaises(m.RebindError):
                m.registered_positions_float32(payload, snap, frame="rest")
        for mode in ("duplicate", "missing", "wrong_frame", "nonrigid"):
            snap = self._registered_snapshot(payload)
            if mode == "duplicate": snap["all_bodies"].append(snap["all_bodies"][0])
            elif mode == "missing": snap["all_bodies"].pop(0)
            elif mode == "wrong_frame": del snap["all_bodies"][0]["rest"]
            else: snap["all_bodies"][0]["rest"]["orientation_world_xyzw"] = [0,0,0,2]
            with self.subTest(mode=mode), self.assertRaises(m.RebindError):
                m.registered_positions_float32(payload, snap, frame="rest")


    def test_retained_faces_remap_by_oriented_vertices_not_ordinal(self):
        source = fixture()
        target = dict(source)
        target["inds"] = np.roll(source["inds"][::-1], 1, axis=1)
        np.testing.assert_array_equal(
            m.remap_unchanged_source_faces(source, target, [0, 1, 0]), [1, 0, 1])
        # An unrelated regional coordinate edit is permitted.
        target["verts"] = source["verts"].copy()
        target["verts"][4, 0] += .1
        np.testing.assert_array_equal(
            m.remap_unchanged_source_faces(source, target, [0]), [1])

    def test_retained_faces_reject_wrong_orientation_and_duplicates(self):
        source = fixture()
        for faces in (source["inds"][:, ::-1], source["inds"][:1],
                      np.r_[source["inds"], source["inds"][:1]]):
            target = dict(source, inds=faces)
            with self.subTest(faces=faces.tolist()), self.assertRaises(m.RebindError):
                m.remap_unchanged_source_faces(source, target, [0, 1])

    def test_retained_faces_reject_changed_coordinates_or_registration(self):
        source = fixture()
        target = dict(source, verts=source["verts"].copy())
        target["verts"][0, 0] += .001
        with self.assertRaisesRegex(m.RebindError, "coordinates changed"):
            m.remap_unchanged_source_faces(source, target, [0])
        for key, value in (("source_sha256", b"wrong"), ("fingerprint", 124)):
            with self.subTest(key=key), self.assertRaises(m.RebindError):
                m.remap_unchanged_source_faces(source, dict(source, **{key:value}), [0])
        for ids in ([-1], [2], [.5], [[0]]):
            with self.subTest(ids=ids), self.assertRaises(m.RebindError):
                m.remap_unchanged_source_faces(source, source, ids)



    def test_protected_faces_require_oriented_surface_not_just_old_vertices(self):
        args = self._append_patch_args()
        data,report = m.append_registered_surface_patch(*args,protected_source_face_ids=[1])
        self.assertEqual(report["protected_source_faces_checked"],1)
        # All original vertex records survive, but triangle 0 was replaced.
        with self.assertRaisesRegex(m.RebindError,"protected source face 0"):
            m.append_registered_surface_patch(*args,protected_source_face_ids=[0])
        p = args[0]
        rotated = np.roll(p["inds"],1,axis=1)
        self.assertEqual(m.validate_protected_source_faces(p,rotated,[0,1]),2)
        for bad in [p["inds"][:,::-1],p["inds"][:1],
                    np.r_[p["inds"],p["inds"][:1]]]:
            with self.subTest(faces=bad.tolist()),self.assertRaises(m.RebindError):
                m.validate_protected_source_faces(p,bad,[0,1])
        with self.assertRaises(m.RebindError):
            m.validate_protected_source_faces(p,p["inds"],[0,0])

    def _append_patch_args(self):
        p = fixture()
        w = p["weights"].copy()
        w[1] = [0,1,0,0]
        p = m.decode(m.rebind(p,w,np.repeat(np.eye(3)[None],4,0),np.zeros((4,3)))[0])
        pose = {"source_poses":[]}
        rest = {"source_poses":[]}
        for i in range(4):
            theta = .6 + .2*i
            pose["source_poses"].append(dict(core_body=10+i,
                position=[.05*i,0,0],quaternion_xyzw=[0,np.sin(theta/2),0,np.cos(theta/2)]))
            rest["source_poses"].append(dict(core_body=10+i,
                position=[0,0,0],quaternion_xyzw=[0,0,0,1]))
        a,b = m.reference_transforms(p,pose,[0,0,.35])
        x = m.positions(p,a,b)
        point = x[[0,1,2]].mean(0) + np.array([.01,0,.01])
        x = np.r_[x,point[None]]
        f = np.array([[0,1,5],[1,2,5],[2,0,5],[1,3,2]],dtype=np.int64)
        return [p,pose,rest,[0,0,.35],x,f,np.array([0,1,2,3,4,5,8]),
                np.array([0,1,2,3,4,0,1,2]),np.array([1,1,1,1,1,1/3,1/3,1/3])]

    def test_appended_patch_preserves_original_records_and_reference(self):
        args = self._append_patch_args()
        p,pose,rest,lift,x,f = args[:6]
        data,report = m.append_registered_surface_patch(*args)
        q = m.decode(data)
        self.assertEqual(q["nv"],6)
        self.assertEqual(report["added_vertices"],1)
        self.assertFalse(report["physical_admission"])
        self.assertEqual(q["raw"][q["vertoff"]:q["vertoff"]+p["nv"]*56],
                         p["raw"][p["vertoff"]:p["indexoff"]])
        np.testing.assert_array_equal(q["weights"][:p["nv"]],p["weights"])
        np.testing.assert_array_equal(q["binds"],p["binds"])
        a,b = m.reference_transforms(p,pose,lift)
        np.testing.assert_allclose(m.positions(q,a,b),x,atol=2e-7)
        np.testing.assert_allclose(q["weights"][5],[2/3,1/3,0,0],atol=1e-7)
        current = {v["core_body"]:m.rotation(v["quaternion_xyzw"]) for v in pose["source_poses"]}
        normal_map = sum(float(q["weights"][5,i])*current[10+i] for i in range(4))
        actual = normal_map@q["verts"][5,3:6].astype(float)
        actual /= np.linalg.norm(actual)
        tri = x[f[:3]]
        target = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]).sum(0)
        target /= np.linalg.norm(target)
        np.testing.assert_allclose(actual,target,atol=2e-7)
        self.assertGreater(report["inferred_displacement_from_ancestry_max_m"],.01)

    def test_appended_patch_rejects_ancestry_or_old_geometry_changes(self):
        for what in ["partition","identity","geometry","unreferenced"]:
            args = self._append_patch_args()
            if what == "partition":
                args[8][-1] = .1
            elif what == "identity":
                args[7][0] = 1
            elif what == "geometry":
                args[4][0,0] += .001
            else:
                args[5] = args[0]["inds"].copy()
            with self.subTest(what=what), self.assertRaises(m.RebindError):
                m.append_registered_surface_patch(*args)

    def test_local_support_transfer_preserves_partition_and_outside(self):
        p = fixture()
        w = p["weights"].copy()
        w[:] = [[.2,.3,.5,0],[.3,.3,.4,0],[.4,.2,.4,0],[.8,.1,.1,0],[0,0,1,0]]
        a = np.repeat(np.eye(3)[None],4,0)
        b = np.array([[0,0,0],[.1,.2,.3],[.3,.1,0],[0,0,0]])
        raw, _ = m.rebind(p,w,a,b)
        p = m.decode(raw)
        out = m.localize_weight_support(p,10,[12],[.2,.4])
        np.testing.assert_array_equal(out[[0,4]],w[[0,4]])
        np.testing.assert_allclose(out[1],[.5,.3,.2,0],atol=2e-7)
        np.testing.assert_allclose(out[2],[.8,.2,0,0],atol=2e-7)
        np.testing.assert_array_equal(out[:,1],w[:,1])
        np.testing.assert_allclose(out.sum(axis=1),w.sum(axis=1),atol=1e-7)
        rebound, receipt = m.rebind(p,out,a,b)
        np.testing.assert_allclose(m.positions(m.decode(rebound),a,b),
                                   m.positions(p,a,b),atol=2e-7)
        self.assertNotIn(0,receipt["changed_rows"])
        self.assertNotIn(4,receipt["changed_rows"])

    def test_local_support_zero_strength_is_identity(self):
        p = fixture()
        np.testing.assert_array_equal(m.localize_weight_support(p,10,[11,12],[.2,.4],0),
                                      p["weights"])

    def test_local_support_rejects_invalid_policy(self):
        p = fixture()
        for ids, thresholds, strength in [([10],[.2,.4],1),([99],[.2,.4],1),
                ([11,11],[.2,.4],1),([],[.2,.4],1),([11],[.4,.2],1),
                ([11],[.2,float("nan")],1),([11],[.2,.4],-1)]:
            with self.assertRaises(m.RebindError):
                m.localize_weight_support(p,10,ids,thresholds,strength)

    def test_reference_preserved_and_unedited_bytes_exact(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], p["nb"], axis=0)
        b = np.array([[0,0,0], [.1,.2,.3], [0,0,0], [0,0,0]])
        w = p["weights"].copy()
        w[1] = [0,1,0,0]
        raw, report = m.rebind(p, w, a, b)
        q = m.decode(raw)
        np.testing.assert_allclose(m.positions(p, a, b), m.positions(q, a, b), atol=1e-7)
        self.assertEqual(report["changed_rows"], [1])
        self.assertEqual(raw[:p["vertoff"]], p["raw"][:p["vertoff"]])
        self.assertEqual(raw[p["indexoff"]:p["weightoff"]], p["raw"][p["indexoff"]:p["weightoff"]])
        for i in [0,2,3,4]:
            o = p["vertoff"]+i*56
            self.assertEqual(raw[o:o+56], p["raw"][o:o+56])
        np.testing.assert_array_equal(q["verts"][:,3:6], p["verts"][:,3:6])
        ids = np.frombuffer(raw, "<u4", 4, p["vertoff"]+56+24)
        self.assertEqual(int(ids[0]), 1)

    def test_patch_follows_topology_and_keeps_disconnected_vertex(self):
        p = fixture()
        w = m.local_rigid_weights(p, 11, [0], [1,.5,0])
        np.testing.assert_array_equal(w[0], [0,1,0,0])
        np.testing.assert_array_equal(w[1], [.5,.5,0,0])
        np.testing.assert_array_equal(w[2], [.5,.5,0,0])
        np.testing.assert_array_equal(w[3:], p["weights"][3:])

    def test_singular_inverse_rejected(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], 4, axis=0)
        a[1] = -a[0]
        w = p["weights"].copy()
        w[0] = [.5,.5,0,0]
        with self.assertRaisesRegex(m.RebindError, "conditioned"):
            m.rebind(p, w, a, np.zeros((4,3)))

    def test_negative_nonfinite_and_bad_partition_rejected(self):
        for row in [[-1,2,0,0], [np.nan,0,0,1], [.5,0,0,0]]:
            w = fixture()["weights"].copy()
            w[0] = row
            with self.assertRaises(m.RebindError):
                m.validate_weights(w)

    def test_identity_replacement_keeps_complete_file_exact(self):
        p = fixture()
        raw, report = m.rebind(p, p["weights"], np.repeat(np.eye(3)[None],4,0),
                               np.zeros((4,3)))
        self.assertEqual(raw, p["raw"])
        self.assertEqual(report["changed_rows"], [])

    def test_wrong_bindings_rejected(self):
        p = fixture()
        raw = bytearray(p["raw"])
        struct.pack_into("<I", raw, 60, 999)
        with self.assertRaisesRegex(m.RebindError, "identities"):
            m.replacement_weights(p, m.decode(bytes(raw)))

    def test_replacement_source_and_registration_must_match(self):
        p = fixture()
        for offset in [24, 28]:
            raw = bytearray(p["raw"])
            raw[offset] ^= 1
            with self.assertRaisesRegex(m.RebindError, "ancestry"):
                m.replacement_weights(p, m.decode(bytes(raw)))

    def test_weight_only_merge_rejects_reordered_or_reoriented_faces(self):
        p = fixture()
        for faces in [p["inds"][::-1], p["inds"][:, ::-1]]:
            raw = bytearray(p["raw"])
            raw[p["indexoff"]:p["weightoff"]] = faces.astype("<u4").tobytes()
            with self.assertRaisesRegex(m.RebindError, "topology"):
                m.merge_weight_repairs(p, [m.decode(bytes(raw))])
        np.testing.assert_array_equal(p["weights"], fixture()["weights"])

    def test_cli_composition_receipt_pins_every_repair(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], 4, 0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp/"base.nhskin"; source.write_bytes(p["raw"])
            paths = []
            for row, weights in [(1,[0,1,0,0]), (3,[.5,0,.5,0])]:
                w = p["weights"].copy(); w[row] = weights
                raw, _ = m.rebind(p,w,a,np.zeros((4,3)))
                path=tmp/("repair"+str(row)+".nhskin");path.write_bytes(raw);paths.append(path)
            pose=tmp/"pose.json"
            pose.write_text(json.dumps({"source_poses":[
                {"core_body":10+i,"position":[0,0,0],"quaternion_xyzw":[0,0,0,1]}
                for i in range(4)]}))
            output,receipt=tmp/"combined.nhskin",tmp/"receipt.json"
            subprocess.run([sys.executable,str(Path(m.__file__)),
                "--input",str(source),"--input-sha256",hashlib.sha256(p["raw"]).hexdigest(),
                "--reference-pose",str(pose),"--root-lift-m","0","0","0",
                "--weights-from",*map(str,paths),"--output",str(output),"--receipt",str(receipt)],
                check=True,capture_output=True,text=True)
            report=json.loads(receipt.read_text())
            self.assertEqual(report["changed_rows"],[1,3])
            self.assertEqual(report["output_sha256"],hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(len(report["weight_repairs"]),2)
            for actual,path in zip(report["weight_repairs"],paths):
                self.assertEqual(actual["sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(actual["path"],str(path.resolve()))
                self.assertEqual(actual["changed_weight_rows"],1)
            self.assertIn("SOURCE_PREPARATION_ONLY",report["admission"])

    def test_bad_indices_and_truncation_rejected(self):
        p = fixture()
        raw = bytearray(p["raw"])
        struct.pack_into("<I", raw, p["indexoff"], p["nv"])
        for data in [bytes(raw), p["raw"][:-1]]:
            with self.assertRaises(m.RebindError):
                m.decode(data)

    def test_small_binding_count_top_four_padding(self):
        p = fixture(2)
        a = np.repeat(np.eye(3)[None],2,0)
        w = p["weights"].copy()
        w[0] = [.25,.75]
        raw, _ = m.rebind(p, w, a, np.zeros((2,3)))
        values = struct.unpack_from("<4I4f", raw, p["vertoff"]+24)
        self.assertEqual(values[:2], (1,0))
        np.testing.assert_allclose(values[4:], [.75,.25,0,0])

    def test_missing_and_duplicate_reference_poses_rejected(self):
        p = fixture()
        pose = {"source_poses": []}
        with self.assertRaises(m.RebindError):
            m.reference_transforms(p, pose, [0,0,0])
        row = {"core_body":10,"position":[0,0,0],"quaternion_xyzw":[0,0,0,1]}
        with self.assertRaisesRegex(m.RebindError, "duplicate"):
            m.reference_transforms(p, {"source_poses":[row,row]}, [0,0,0])

    def test_disjoint_repairs_compose_without_overwrite(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], 4, 0)
        b = np.array([[0,0,0], [.1,.2,.3], [0,0,0], [0,0,0]])
        repairs = []
        for row, weights in [(1, [0,1,0,0]), (3, [.5,0,.5,0])]:
            w = p["weights"].copy()
            w[row] = weights
            raw, _ = m.rebind(p, w, a, b)
            repairs.append(m.decode(raw))
        w, counts = m.merge_weight_repairs(p, repairs)
        self.assertEqual(counts, [1,1])
        np.testing.assert_array_equal(w[1], [0,1,0,0])
        np.testing.assert_array_equal(w[3], [.5,0,.5,0])
        raw, report = m.rebind(p, w, a, b)
        self.assertEqual(report["changed_rows"], [1,3])
        np.testing.assert_allclose(m.positions(m.decode(raw), a, b),
                                   m.positions(p, a, b), atol=1e-7)

    def test_conflicting_repairs_rejected_identical_overlap_allowed(self):
        p = fixture()
        a = np.repeat(np.eye(3)[None], 4, 0)
        repairs = []
        for weights in [[0,1,0,0], [0,0,1,0]]:
            w = p["weights"].copy()
            w[1] = weights
            raw, _ = m.rebind(p, w, a, np.zeros((4,3)))
            repairs.append(m.decode(raw))
        with self.assertRaisesRegex(m.RebindError, "vertex 1"):
            m.merge_weight_repairs(p, repairs)
        w, counts = m.merge_weight_repairs(p, [repairs[0], repairs[0]])
        np.testing.assert_array_equal(w, repairs[0]["weights"])
        self.assertEqual(counts, [1,1])
        np.testing.assert_array_equal(p["weights"], fixture()["weights"])

    def test_empty_composition_is_exact_identity(self):
        p = fixture()
        w, counts = m.merge_weight_repairs(p, [])
        np.testing.assert_array_equal(w, p["weights"])
        self.assertEqual(counts, [])

    def test_unknown_body_and_invalid_shell_rejected(self):
        p = fixture()
        for body, seeds, gates in [(999,[0],[1]), (10,[5],[1]), (10,[0],[1.1])]:
            with self.assertRaises(m.RebindError):
                m.local_rigid_weights(p, body, seeds, gates)


if __name__ == "__main__":
    unittest.main()
