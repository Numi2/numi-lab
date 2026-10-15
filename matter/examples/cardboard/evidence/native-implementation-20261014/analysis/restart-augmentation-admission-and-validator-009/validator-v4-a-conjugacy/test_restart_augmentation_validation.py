#!/usr/bin/env python3
"""CPU contract tests for the restart-augmentation offline validator.

The synthetic augmentation in this file checks validator plumbing only. It is
not Matter output and cannot qualify a native restart augmentation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor

ROOT = Path("/Users/home/cardboard-implementation-20261014")
PACKAGE = ROOT / "analysis/restart-augmentation-admission-and-validator-009/validator-v4-a-conjugacy"
CAPTURE_DIR = ROOT / "solver/selected-newton-regional-rap-and-operator-capture-182"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "restart_augmentation_validation_v4", PACKAGE / "validate_restart_augmentation.py")
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
validator = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = validator
MODULE_SPEC.loader.exec_module(validator)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rich_arrays():
    manifest = json.loads((CAPTURE_DIR / "operator-capture.regional.json").read_text())
    raw = Path(manifest["payload"]["file"]).read_bytes()
    output = {}
    for desc in manifest["arrays"]:
        start = desc["offset"]
        data = raw[start:start + desc["byte_count"]]
        dtype = desc["dtype"]
        if dtype in ("<f4x4", "<u4x4"):
            scalar = "<f4" if dtype == "<f4x4" else "<u4"
            shape = tuple(desc["shape"]) + (4,)
            output[desc["name"]] = np.frombuffer(data, dtype=scalar).copy().reshape(shape)
        elif dtype == "<f4":
            output[desc["name"]] = np.frombuffer(data, dtype="<f4").copy().reshape(desc["shape"])
        elif dtype == "<u4":
            output[desc["name"]] = np.frombuffer(data, dtype="<u4").copy().reshape(desc["shape"])
        else:
            output[desc["name"]] = np.frombuffer(data, dtype=np.uint8).copy()
    return manifest, output


def write_conditioning(directory: Path, arrays: dict[str, bytes], base_manifest: dict,
                       augmentation: dict | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, raw in sorted(arrays.items()):
        path = directory / name
        path.write_bytes(raw)
        stride = 16 if name in (
            "P.bin", "effective_P.bin", "AP.bin", "node_block_inverse.bin",
            "restart_q_by_environment.bin", "restart_Aq_by_environment.bin") else 4
        if name == "candidate_nodes.bin":
            stride = 80
        count = len(raw) // stride
        files.append({"name": name, "count": count, "stride_bytes": stride,
                      "bytes": len(raw), "sha256": sha(raw)})
    manifest = dict(base_manifest)
    manifest["files"] = files
    manifest["byte_order"] = "native_little_endian"
    if augmentation is not None:
        manifest.update({
            "restart_augmentation": augmentation,
            "regional_basis_storage_mode_stride": augmentation["augmented_mode_stride"],
            "exported_base_modes_compacted": True,
        })
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def synthetic_augmented_payload(tmp: Path):
    operator_path = CAPTURE_DIR / "operator-capture"
    regional_manifest_path = CAPTURE_DIR / "operator-capture.regional.json"
    base_cond_path = CAPTURE_DIR / "output/rejected-conditioning-000004/manifest.json"
    regional, rarrays = rich_arrays()
    base_manifest = json.loads(base_cond_path.read_text())
    envs, modes, nodes = 2, 26, 882
    stride = modes + 1
    arr: dict[str, bytes] = {}
    for name in ("candidate_nodes.bin", "node_block_inverse.bin", "pivoted_RAP.bin",
                 "pivots.bin", "factor_valid.bin"):
        arr[name] = (base_cond_path.parent / name).read_bytes()
    # Use the rich capture's independently recorded regional P/AP; the older
    # conditioning AP file for env0 is known to be inconsistent and is retained
    # as a failed artifact, never silently repaired in place.
    arr["P.bin"] = np.asarray(rarrays["P"], dtype="<f4").tobytes()
    arr["effective_P.bin"] = np.asarray(rarrays["effective_P"], dtype="<f4").tobytes()
    arr["AP.bin"] = np.asarray(rarrays["AP"], dtype="<f4").tobytes()

    capture = validator.load_rich_capture(operator_path)
    assembler = validator.load_operator_assembler(validator.DEFAULT_ASSEMBLER)
    q = np.zeros((envs, nodes, 4), dtype="<f4")
    aq = np.zeros_like(q)
    gens = np.full((envs, 4), validator.UINT32_MAX, dtype="<u4")
    attempts = np.zeros(envs, dtype="<u4")
    target = np.zeros(envs, dtype="<f4")
    raw_norm = np.zeros(envs, dtype="<f4")
    mode_scale = np.zeros(envs, dtype="<f4")
    mode_valid = np.zeros(envs, dtype="<u4")
    factor_valid = np.zeros(envs, dtype="<u4")
    aug_lu = np.zeros((envs, stride, stride), dtype="<f4")
    aug_pivots = np.zeros((envs, stride), dtype="<u4")
    header = capture.header
    control_step = int(base_manifest["control_step"])

    for env in (0,):
        _, _, matrix, _, meta = assembler.validate_and_assemble(
            capture.header, capture.nodes, capture.tets, capture.integrity, env)
        free = meta["free_nodes"]
        rng = np.random.default_rng(20261015 + env)
        raw_q = rng.standard_normal((len(free), 3))
        pmat = rarrays["effective_P"][env][:, free, :3].astype(np.float64).transpose(1, 2, 0).reshape(-1, modes)
        apmat = rarrays["AP"][env][:, free, :3].astype(np.float64).transpose(1, 2, 0).reshape(-1, modes)
        raw_vector = raw_q.reshape(-1)
        raw_image = np.asarray(matrix @ raw_vector).reshape(-1)
        base_rap = pmat.T @ apmat
        coefficients = np.linalg.solve(base_rap, pmat.T @ raw_image)
        qf = raw_vector - pmat @ coefficients
        qf /= np.linalg.norm(qf)
        q[env, free, :3] = qf.reshape(-1, 3).astype("<f4")
        qf32 = q[env, free, :3].astype(np.float64).reshape(-1)
        aqf = np.asarray(matrix @ qf32).reshape(-1)
        aq[env, free, :3] = aqf.reshape(-1, 3).astype("<f4")

        # The authored restart mode is A-conjugate to P. Euclidean P^Tq=0 is
        # not the producer invariant when A is nonsymmetric.
        self_cross = pmat.T @ aq[env, free, :3].astype(np.float64).reshape(-1)
        self_scale = max(np.linalg.norm(pmat) * np.linalg.norm(aq[env, free, :3]), 1e-300)
        assert np.linalg.norm(self_cross) / self_scale < 2e-3
        p_aug = np.column_stack((pmat, qf32))
        ap_aug = np.column_stack((apmat, aq[env, free, :3].astype(np.float64).reshape(-1)))
        rap = p_aug.T @ ap_aug
        lu, steps = lu_factor(rap)
        permutation = np.arange(stride, dtype="<u4")
        for row, col in enumerate(steps):
            permutation[[row, int(col)]] = permutation[[int(col), row]]
        aug_lu[env] = lu.astype("<f4")
        aug_pivots[env] = permutation
        gens[env] = np.asarray([control_step, 0, int(header["newton_iteration"]), 0], dtype="<u4")
        attempts[env] = 1
        target[env] = 1.0
        raw_norm[env] = np.float32(qf32 @ qf32)
        mode_scale[env] = 1.0
        mode_valid[env] = 1
        factor_valid[env] = 1

    arr.update({
        "restart_generations_by_environment.bin": gens.tobytes(),
        "restart_attempt_counts_by_environment.bin": attempts.tobytes(),
        "restart_target_scales_by_environment.bin": target.tobytes(),
        "restart_raw_norm_squared_by_environment.bin": raw_norm.tobytes(),
        "restart_mode_scales_by_environment.bin": mode_scale.tobytes(),
        "restart_mode_valid_by_environment.bin": mode_valid.tobytes(),
        "restart_factor_valid_by_environment.bin": factor_valid.tobytes(),
        "restart_q_by_environment.bin": q.tobytes(),
        "restart_Aq_by_environment.bin": aq.tobytes(),
        "restart_pivoted_RAP_by_environment.bin": aug_lu.tobytes(),
        "restart_pivots_by_environment.bin": aug_pivots.tobytes(),
    })
    augmentation = {"diagnostics_available": True,
                    "diagnostic_environment_count": envs,
                    "nodes_per_environment": nodes,
                    "augmented_mode_stride": stride}
    cond_path = write_conditioning(tmp / "conditioning", arr, base_manifest, augmentation)
    return operator_path, regional_manifest_path, cond_path


class RestartValidatorTests(unittest.TestCase):
    def test_float4_uint4_vector_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f = np.arange(24, dtype="<f4").reshape(2, 3, 4)
            u = np.arange(8, dtype="<u4").reshape(2, 4)
            scalar = np.array([2.0, 3.0], dtype="<f4")
            payload = f.tobytes() + u.tobytes() + scalar.tobytes()
            (root / "capture.bin").write_bytes(payload)
            descriptors = []
            offset = 0
            for name, dtype, shape, itemsize, data in (
                    ("f", "<f4x4", [2, 3], 16, f.tobytes()),
                    ("u", "<u4x4", [2], 16, u.tobytes()),
                    ("s", "<f4", [2], 4, scalar.tobytes())):
                descriptors.append({"name": name, "dtype": dtype, "shape": shape,
                    "element_bytes": itemsize, "offset": offset,
                    "byte_count": len(data), "sha256": sha(data)})
                offset += len(data)
            manifest = {"schema": validator.SCHEMA_REGIONAL,
                        "byte_order": "little_endian",
                        "payload": {"file": str(root / "capture.bin"),
                                    "byte_count": len(payload), "sha256": sha(payload)},
                        "arrays": descriptors}
            mp = root / "capture.json"
            mp.write_text(json.dumps(manifest))
            loaded = validator.load_regional(mp)
            self.assertEqual(loaded.arrays["f"].shape, (2, 3, 4))
            self.assertEqual(loaded.arrays["u"].shape, (2, 4))
            np.testing.assert_array_equal(loaded.arrays["f"], f)
            np.testing.assert_array_equal(loaded.arrays["u"], u)

    def test_capture182_decodes_and_exposes_inconsistent_conditioning(self):
        regional, arrays = rich_arrays()
        self.assertEqual(arrays["P"].shape, (26, 882, 4))
        self.assertEqual(arrays["effective_P"].shape, (2, 26, 882, 4))
        self.assertEqual(arrays["AP"].shape, (2, 26, 882, 4))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            args = [sys.executable, str(PACKAGE / "validate_restart_augmentation.py"),
                    "--operator-capture", str(CAPTURE_DIR / "operator-capture"),
                    "--regional-json", str(CAPTURE_DIR / "operator-capture.regional.json"),
                    "--conditioning-manifest", str(CAPTURE_DIR / "output/rejected-conditioning-000004/manifest.json"),
                    "--output", str(output)]
            import subprocess
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            report = json.loads(output.read_text())
            self.assertIn("conditioning compact AP differs", report["error"])

    def test_synthetic_attempt_and_quiet_environment_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            old_err = np.seterr(all="ignore")
            try:
                operator, regional, conditioning = synthetic_augmented_payload(Path(temp))
                report = validator.validate(operator, regional, conditioning, None,
                    validator.DEFAULT_ASSEMBLER, True, 2e-3, 2e-4)
            finally:
                np.seterr(**old_err)
            self.assertEqual(report["status"], "internally_checked_diagnostic_capture")
            self.assertTrue(report["environments"][0]["attempt"])
            self.assertTrue(report["environments"][0]["fresh_Aq_gate"]["passed"])
            self.assertTrue(report["environments"][0]["augmented_RAP_gate"]["passed"])
            self.assertFalse(report["environments"][1]["attempt"])
            self.assertEqual(report["environments"][1]["generation_words"],
                             [validator.UINT32_MAX] * 4)
            self.assertTrue(all(np.isfinite(report["environments"][0]["fresh_Aq"][key])
                for key in ("relative_l2", "difference_l2", "max_abs")))
            same_capture = validator.compare_centers(
                validator.load_rich_capture(operator),
                validator.load_rich_capture(operator))
            self.assertEqual(same_capture["status"], "exact_center_match")

            cond_root = conditioning.parent
            manifest = json.loads(conditioning.read_text())
            q_path = cond_root / "restart_q_by_environment.bin"
            q_values = np.fromfile(q_path, dtype="<f4")
            # The second environment is explicitly no-attempt and must remain
            # all-zero, including its q arena.
            q_values[882 * 4] = 1.0
            q_path.write_bytes(q_values.tobytes())
            for desc in manifest["files"]:
                if desc["name"] == q_path.name:
                    raw = q_path.read_bytes()
                    desc["bytes"] = len(raw)
                    desc["sha256"] = sha(raw)
                    break
            conditioning.write_text(json.dumps(manifest, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "quiet no-attempt payload"):
                validator.validate(operator, regional, conditioning, None,
                    validator.DEFAULT_ASSEMBLER, True, 2e-3, 2e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
