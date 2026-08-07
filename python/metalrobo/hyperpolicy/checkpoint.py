"""Atomic compiler checkpoints for the MLX ARDY hyper-policy model."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlx.core as mx

from .base import HyperBasePolicy
from .common import _atomic_directory, _canonical_json
from .mlx_production_model import ARDYHyperNetwork, ARDYHyperPolicyConfiguration

_FORMAT = "numi.ardy-hyperpolicy-checkpoint.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration_record(
    configuration: ARDYHyperPolicyConfiguration,
) -> dict[str, Any]:
    record = asdict(configuration)
    record["coefficient_limits"] = list(configuration.coefficient_limits)
    return record


def _configuration_from_record(
    record: Mapping[str, Any],
) -> ARDYHyperPolicyConfiguration:
    values = dict(record)
    values["coefficient_limits"] = tuple(
        float(value) for value in values.get("coefficient_limits", ())
    )
    configuration = ARDYHyperPolicyConfiguration(**values)
    configuration.validate()
    return configuration


def write_compiler_checkpoint(
    directory: str | Path,
    *,
    hypernetwork: ARDYHyperNetwork,
    hyper_base: HyperBasePolicy,
    configuration: ARDYHyperPolicyConfiguration,
    feature_schema: Sequence[str],
    training_updates: int,
    metrics: Mapping[str, float] | None = None,
) -> Path:
    configuration.validate()
    base = hyper_base.with_fingerprint()
    base.validate(require_fingerprint=True)
    if (
        configuration.action_count != base.action_count
        or configuration.coefficient_count != base.coefficient_count
        or len(feature_schema) != configuration.feature_count
        or training_updates < 0
    ):
        raise ValueError("compiler checkpoint contracts are inconsistent")
    target = Path(directory)
    with _atomic_directory(target) as staging:
        weights = staging / "hypernetwork.npz"
        hypernetwork.save_weights(str(weights))
        mx.eval(hypernetwork.parameters())
        base.write(staging / "hyper-base")
        manifest = {
            "format": _FORMAT,
            "configuration": _configuration_record(configuration),
            "feature_schema": [str(value) for value in feature_schema],
            "hyper_base_fingerprint": base.fingerprint,
            "training_updates": int(training_updates),
            "metrics": {
                str(key): float(value) for key, value in (metrics or {}).items()
            },
            "weights_sha256": _sha256(weights),
        }
        core = _canonical_json(manifest)
        manifest["fingerprint"] = hashlib.sha256(core).hexdigest()
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
    return target


def read_compiler_checkpoint(
    directory: str | Path,
) -> tuple[
    ARDYHyperNetwork,
    HyperBasePolicy,
    ARDYHyperPolicyConfiguration,
    tuple[str, ...],
    dict[str, Any],
]:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("format") != _FORMAT:
        raise ValueError("ARDY hyper-policy checkpoint format is unsupported")
    fingerprint = manifest.pop("fingerprint", None)
    if hashlib.sha256(_canonical_json(manifest)).hexdigest() != fingerprint:
        raise ValueError("ARDY hyper-policy checkpoint fingerprint is invalid")
    weights = source / "hypernetwork.npz"
    if _sha256(weights) != manifest["weights_sha256"]:
        raise ValueError("hypernetwork weight fingerprint is invalid")
    configuration = _configuration_from_record(manifest["configuration"])
    feature_schema = tuple(str(value) for value in manifest["feature_schema"])
    if len(feature_schema) != configuration.feature_count:
        raise ValueError("checkpoint feature schema width is invalid")
    base = HyperBasePolicy.read(source / "hyper-base")
    if base.fingerprint != manifest["hyper_base_fingerprint"]:
        raise ValueError("checkpoint hyper-base fingerprint is invalid")
    network = ARDYHyperNetwork(configuration)
    network.load_weights(str(weights), strict=True)
    mx.eval(network.parameters())
    return network, base, configuration, feature_schema, dict(manifest)
