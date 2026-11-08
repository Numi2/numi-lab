#!/usr/bin/env python3
"""Export a deterministic PointWorld CUDA oracle corpus from the pinned release.

This file intentionally lives outside the NVIDIA checkout.  Copy it into, or run
it with ``--source-root`` pointing at, an unmodified PointWorld source tree.
It loads the release evaluator, takes one canonical evaluation batch, records
selected intermediate tensors through forward hooks, and emits immutable NPZ
and JSON receipts for the Apple-native implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


PINNED_SOURCE_REVISION = "05484826dfef74cbe278a3974179a5a16705d35d"
PINNED_DINOV3_REVISION = "54694f7627fd815f62a5dcc82944ffa6153bbb76"
DEFAULT_CAPTURE_REGEX = (
    r"^(scene_feature_encoder|robot_proj|time_embed|dynamics_predictor"
    r"(\.predictor_model(\.(enc|dec))?|\.dynamics_head|\.log_var_head)?)$"
)


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _collect_tensors(torch: Any, value: Any, prefix: str, out: dict[str, np.ndarray]) -> None:
    """Flatten tensor-bearing module outputs without depending on Point internals."""
    if isinstance(value, torch.Tensor):
        out[prefix] = value.detach().to("cpu").contiguous().numpy()
        return
    if isinstance(value, dict):
        for key in sorted(value):
            if not str(key).startswith("_"):
                _collect_tensors(torch, value[key], f"{prefix}.{key}", out)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_tensors(torch, item, f"{prefix}.{index}", out)
        return
    # Pointcept's Point is dict-like in the pinned release.  Attribute fallback
    # captures the topology-bearing fields needed by the native PTv3 port.
    for name in ("coord", "feat", "batch", "offset", "serialized_code",
                 "serialized_order", "serialized_inverse", "pooling_inverse"):
        if hasattr(value, name):
            _collect_tensors(torch, getattr(value, name), f"{prefix}.{name}", out)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit one deterministic golden corpus from NVIDIA PointWorld",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--capture-regex", default=DEFAULT_CAPTURE_REGEX)
    parser.add_argument(
        "pointworld_args", nargs=argparse.REMAINDER,
        help="Arguments forwarded to PointWorld eval.py (precede them with --)",
    )
    return parser


def main() -> int:
    outer = _parser().parse_args()
    if outer.repeats < 2:
        raise ValueError("--repeats must be at least 2 to measure CUDA variation")
    source_root = outer.source_root.resolve()
    checkpoint = outer.checkpoint.resolve()
    if _git_revision(source_root) != PINNED_SOURCE_REVISION:
        raise RuntimeError("PointWorld source revision does not match the frozen contract")
    dino_root = source_root / "third_party" / "dinov3"
    if _git_revision(dino_root) != PINNED_DINOV3_REVISION:
        raise RuntimeError("DINOv3 submodule revision does not match the frozen contract")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sys.path.insert(0, str(source_root))
    import re
    import torch
    from arguments import parse_args
    from evaluation.tester import Tester

    forwarded = list(outer.pointworld_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    sys.argv = ["eval.py", "--model_path", str(checkpoint), *forwarded]
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    tester = Tester(args)
    loader, loader_info = tester._build_eval_loader(outer.split)
    batch = next(iter(loader))
    device_batch = {
        key: value.to(tester.device, non_blocking=True)
        if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    arrays: dict[str, np.ndarray] = {}
    for key, value in sorted(batch.items()):
        if isinstance(value, torch.Tensor):
            arrays[f"input.{key}"] = value.detach().cpu().contiguous().numpy()

    capture_pattern = re.compile(outer.capture_regex)
    capture_enabled = True
    handles = []

    def make_hook(module_name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if capture_enabled:
                _collect_tensors(torch, output, f"stage.{module_name}", arrays)
        return hook

    for name, module in tester.model.named_modules():
        if name and capture_pattern.search(name):
            handles.append(module.register_forward_hook(make_hook(name)))
    if not handles:
        raise RuntimeError("capture expression matched no model modules")

    outputs_per_run: list[dict[str, np.ndarray]] = []
    timings_ms: list[float] = []
    tester.model.eval()
    try:
        with torch.no_grad():
            for run_index in range(outer.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = tester.model(device_batch, training=False)
                torch.cuda.synchronize()
                timings_ms.append((time.perf_counter() - started) * 1000.0)
                run_arrays: dict[str, np.ndarray] = {}
                for key in ("scene_relative_norm", "scene_relative", "scene_flows",
                            "log_var", "confidence"):
                    _collect_tensors(torch, outputs[key], f"output.{key}", run_arrays)
                outputs_per_run.append(run_arrays)
                if run_index == 0:
                    arrays.update(run_arrays)
                capture_enabled = False
    finally:
        for handle in handles:
            handle.remove()

    variation: dict[str, dict[str, float]] = {}
    for key in sorted(outputs_per_run[0]):
        stack = np.stack([run[key].astype(np.float64) for run in outputs_per_run])
        delta = np.abs(stack - stack[0:1])
        variation[key] = {
            "max_abs": float(delta.max(initial=0.0)),
            "mean_abs": float(delta.mean()),
        }

    array_receipts = {
        key: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_fingerprint(value),
            "finite": bool(np.isfinite(value).all()) if value.dtype.kind in "fc" else True,
        }
        for key, value in sorted(arrays.items())
    }
    output_prefix = outer.output.resolve()
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")
    _atomic_npz(npz_path, arrays)
    receipt = {
        "contract": "PointWorldCUDAOracleV1",
        "source_revision": PINNED_SOURCE_REVISION,
        "dinov3_revision": PINNED_DINOV3_REVISION,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "split": outer.split,
        "pointworld_args": forwarded,
        "arrays": array_receipts,
        "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        "repeats": outer.repeats,
        "timings_ms": timings_ms,
        "run_to_run_variation": variation,
        "loader_info": str(loader_info),
        "runtime": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    _atomic_json(json_path, receipt)
    print(json.dumps({"receipt": str(json_path), "npz": str(npz_path),
                      "npz_sha256": receipt["npz"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
