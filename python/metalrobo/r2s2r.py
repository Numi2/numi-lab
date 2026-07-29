"""Durable coordinator for the executable real-to-sim-to-real loop.

The immutable world, replay-fit population, policy feedback, and episode
outcomes are separate content-addressed artifacts. SQLite stores searchable
metadata only; large arrays, telemetry, video, and learned parameters stay in
the artifact store.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from .mlx_r2s2r import (
    CompactedEpisodeRecords,
    FailureTrainingBatch,
    FiveMemberFailureEnsemble,
    SMCConfig,
    compile_feedback_regions,
    deterministic_candidate_scenarios,
    fit_alignment_smc,
)
from .worlds import FrankaPickPlaceWorldFamily, ScenarioSchema


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fingerprint64(value: str | bytes) -> int:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _hex64(value: int) -> str:
    return f"{int(value) & ((1 << 64) - 1):016x}"


def _parse_hex64(value: str | int | None, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        if not 0 <= value < 1 << 64:
            raise ValueError("fingerprint must fit in a uint64")
        return value
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if not text:
        return default
    if len(text) > 16 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"invalid uint64 fingerprint: {value!r}")
    return int(text, 16)


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int = 1024,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap interval requires observations")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return value, value
    generator = np.random.default_rng(seed & ((1 << 64) - 1))
    selections = generator.integers(
        0,
        array.size,
        size=(replicates, array.size),
    )
    estimates = np.mean(array[selections], axis=1)
    lower, upper = np.quantile(
        estimates,
        (0.025, 0.975),
        method="linear",
    )
    return float(lower), float(upper)


def _mean_maximum_rank_violation(
    real_values: Sequence[float],
    simulation_values: Sequence[float],
) -> float:
    real = np.asarray(real_values, dtype=np.float64)
    simulation = np.asarray(simulation_values, dtype=np.float64)
    if real.shape != simulation.shape or real.ndim != 1 or real.size < 2:
        raise ValueError("MMRV requires paired policy vectors")
    violations = np.abs(real[:, None] - real[None, :]) * (
        (simulation[:, None] < simulation[None, :])
        != (real[:, None] < real[None, :])
    )
    return float(np.mean(np.max(violations, axis=1)))


def _rank_correlation(
    real_values: Sequence[float],
    simulation_values: Sequence[float],
) -> float | None:
    real = np.asarray(real_values, dtype=np.float64)
    simulation = np.asarray(simulation_values, dtype=np.float64)
    if real.shape != simulation.shape or real.ndim != 1 or real.size < 2:
        return None

    def ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        order = np.argsort(values, kind="stable")
        result = np.empty_like(values)
        begin = 0
        while begin < order.size:
            end = begin + 1
            while (
                end < order.size
                and values[order[end]] == values[order[begin]]
            ):
                end += 1
            result[order[begin:end]] = 0.5 * (begin + end - 1)
            begin = end
        return result

    real_ranks = ranks(real)
    simulation_ranks = ranks(simulation)
    real_centered = real_ranks - np.mean(real_ranks)
    simulation_centered = simulation_ranks - np.mean(simulation_ranks)
    denominator = math.sqrt(
        float(np.sum(real_centered * real_centered))
        * float(np.sum(simulation_centered * simulation_centered))
    )
    if denominator <= 0.0:
        return None
    return float(
        np.sum(real_centered * simulation_centered) / denominator
    )


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    content_hash: str
    kind: str
    media_type: str
    byte_count: int
    path: str


class ContentAddressedArtifactStore:
    """Atomic SHA-256 object store with immutable object paths."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        media_type: str,
        suffix: str = "",
    ) -> ArtifactRef:
        if not kind:
            raise ValueError("artifact kind cannot be empty")
        content_hash = _hash_bytes(data)
        object_path = (
            self.objects
            / content_hash[:2]
            / f"{content_hash}{suffix}"
        )
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if not object_path.exists():
            with tempfile.NamedTemporaryFile(
                dir=object_path.parent,
                prefix=f".{content_hash}.",
                delete=False,
            ) as temporary:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            try:
                os.replace(temporary_path, object_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        return ArtifactRef(
            content_hash=content_hash,
            kind=kind,
            media_type=media_type,
            byte_count=len(data),
            path=str(object_path),
        )

    def put_json(self, value: Any, *, kind: str) -> ArtifactRef:
        return self.put_bytes(
            _canonical_json(value),
            kind=kind,
            media_type="application/json",
            suffix=".json",
        )

    def put_npz(
        self,
        *,
        kind: str,
        arrays: Mapping[str, npt.ArrayLike],
    ) -> ArtifactRef:
        payload = io.BytesIO()
        np.savez_compressed(
            payload,
            **{
                name: np.asarray(value)
                for name, value in arrays.items()
            },
        )
        return self.put_bytes(
            payload.getvalue(),
            kind=kind,
            media_type="application/x-npz",
            suffix=".npz",
        )

    def read_json(self, content_hash: str) -> Any:
        path = self._find(content_hash)
        return json.loads(path.read_text(encoding="utf-8"))

    def read_npz(self, content_hash: str) -> dict[str, np.ndarray]:
        path = self._find(content_hash)
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name].copy() for name in payload.files}

    def _find(self, content_hash: str) -> Path:
        if (
            len(content_hash) != 64
            or any(character not in "0123456789abcdef" for character in content_hash)
        ):
            raise ValueError("artifact hash must be lowercase SHA-256")
        matches = tuple(
            (self.objects / content_hash[:2]).glob(f"{content_hash}*")
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"content-addressed artifact is unavailable: {content_hash}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    id: str
    content_hash: str
    observation_schema: str
    action_schema: str
    embodiment: str

    @property
    def fingerprint(self) -> int:
        return _fingerprint64(_canonical_json(asdict(self)))

    @property
    def embodiment_fingerprint(self) -> int:
        return _fingerprint64(self.embodiment)


@dataclass(slots=True)
class EpisodeOutcome:
    id: str
    run_id: str
    source: str
    policy_fingerprint: int
    task_fingerprint: int
    embodiment_fingerprint: int
    scenario_schema: str
    policy_id: str = ""
    robot_id: str = ""
    task_id: str = ""
    scenario_key: int = 0
    episode_counter: int = 0
    family_fingerprint: int = 0
    alignment_fingerprint: int = 0
    feedback_fingerprint: int = 0
    termination: str = "horizon"
    success: bool = False
    failure_tags: tuple[str, ...] = ()
    physics_status: int = 0
    step_count: int = 0
    episode_return: float = 0.0
    task_margin: float = 0.0
    safety_margin: float = 0.0
    duration_seconds: float = 0.0
    minimum_visibility: float = 0.0
    integrated_contact_load: float = 0.0
    peak_contact_load: float = 0.0
    scenario_values: dict[str, float | None] = field(default_factory=dict)
    missing_value_mask: dict[str, bool] = field(default_factory=dict)
    scenario_quantiles: list[float] | None = None
    artifacts: tuple[dict[str, str], ...] = ()

    def validate(self, schema: ScenarioSchema) -> None:
        if not self.id or not self.run_id:
            raise ValueError("outcome id and run_id cannot be empty")
        if self.source not in {"simulation", "hardware"}:
            raise ValueError("outcome source must be simulation or hardware")
        if self.source == "hardware" and (
            not self.policy_id or not self.robot_id or not self.task_id
        ):
            raise ValueError(
                "hardware outcome requires policy_id, robot_id, and task_id"
            )
        if self.termination not in {
            "success",
            "horizon",
            "policy",
            "physics",
            "safety",
            "external",
        }:
            raise ValueError("outcome termination is invalid")
        if self.scenario_schema != schema.id:
            raise ValueError(
                f"outcome schema {self.scenario_schema!r} does not match "
                f"{schema.id!r}"
            )
        if self.scenario_quantiles is not None and len(
            self.scenario_quantiles
        ) != len(schema.features):
            raise ValueError("scenario_quantiles has the wrong feature width")
        feature_ids = {feature.id for feature in schema.features}
        if (
            not set(self.scenario_values).issubset(feature_ids)
            or not set(self.missing_value_mask).issubset(feature_ids)
        ):
            raise ValueError("outcome contains an unknown scenario feature")
        for feature in schema.features:
            missing = self.missing_value_mask.get(
                feature.id,
                self.scenario_values.get(feature.id) is None,
            )
            if missing != (
                self.scenario_values.get(feature.id) is None
            ):
                raise ValueError(
                    f"missing-value mask conflicts for {feature.id!r}"
                )
        if any(not tag for tag in self.failure_tags):
            raise ValueError("failure tags cannot be empty")
        if (
            self.source == "simulation"
            and self.physics_status != 0
        ):
            raise ValueError(
                "a physics-invalid simulation episode cannot be recorded"
            )
        if self.success and self.physics_status != 0:
            raise ValueError(
                "a physics-invalid episode cannot be recorded as success"
            )
        for value in (
            self.episode_return,
            self.task_margin,
            self.safety_margin,
            self.duration_seconds,
            self.minimum_visibility,
            self.integrated_contact_load,
            self.peak_contact_load,
        ):
            if not math.isfinite(value):
                raise ValueError("outcome metrics must be finite")

    def as_manifest(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "scenario_key",
            "episode_counter",
            "family_fingerprint",
            "alignment_fingerprint",
            "feedback_fingerprint",
            "policy_fingerprint",
            "task_fingerprint",
            "embodiment_fingerprint",
        ):
            result[key] = _hex64(result[key])
        return result


@dataclass(frozen=True, slots=True)
class AlignmentArtifact:
    content_hash: str
    fingerprint: int
    schema_fingerprint: int
    particle_count: int
    replay_artifact_hash: str


@dataclass(frozen=True, slots=True)
class FeedbackArtifact:
    content_hash: str
    model_content_hash: str
    fingerprint: int
    schema_fingerprint: int
    policy_fingerprint: int
    task_fingerprint: int
    embodiment_fingerprint: int
    region_count: int
    hardware_available: bool
    hardware_evidence_count: int = 0
    hardware_predictive_variance: float | None = None


class R2S2RCoordinator:
    """Owns one repeatable R2S2R artifact graph and searchable outcome log."""

    schema_version = 1

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = ContentAddressedArtifactStore(
            self.root / "artifacts"
        )
        self.database_path = self.root / "r2s2r.sqlite3"
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as database:
            database.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    content_hash TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    created_unix REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policies (
                    fingerprint TEXT PRIMARY KEY,
                    id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    descriptor_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alignments (
                    content_hash TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    schema_fingerprint TEXT NOT NULL,
                    particle_count INTEGER NOT NULL,
                    replay_artifact_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback_programs (
                    content_hash TEXT PRIMARY KEY,
                    model_content_hash TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    schema_fingerprint TEXT NOT NULL,
                    policy_fingerprint TEXT NOT NULL,
                    task_fingerprint TEXT NOT NULL,
                    embodiment_fingerprint TEXT NOT NULL,
                    region_count INTEGER NOT NULL,
                    hardware_available INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    scenario_schema TEXT NOT NULL,
                    scenario_key TEXT NOT NULL,
                    episode_counter TEXT NOT NULL,
                    policy_fingerprint TEXT NOT NULL,
                    task_fingerprint TEXT NOT NULL,
                    embodiment_fingerprint TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    task_margin REAL NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS outcomes_policy_task
                    ON outcomes(
                        policy_fingerprint,
                        task_fingerprint,
                        embodiment_fingerprint,
                        source
                    );
                CREATE INDEX IF NOT EXISTS outcomes_pairing
                    ON outcomes(task_fingerprint, scenario_key, source);
                CREATE TABLE IF NOT EXISTS iterations (
                    id TEXT PRIMARY KEY,
                    command TEXT NOT NULL,
                    created_unix REAL NOT NULL,
                    world_hash TEXT NOT NULL,
                    alignment_hash TEXT NOT NULL,
                    sampling_hash TEXT NOT NULL,
                    policy_hash TEXT NOT NULL,
                    engine_hash TEXT NOT NULL,
                    hardware_hash TEXT NOT NULL,
                    provenance_json TEXT NOT NULL
                );
                """
            )

    def _register_artifact(self, artifact: ArtifactRef) -> None:
        with self._connect() as database:
            database.execute(
                """
                INSERT OR IGNORE INTO artifacts
                (content_hash, kind, media_type, byte_count, path, created_unix)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.content_hash,
                    artifact.kind,
                    artifact.media_type,
                    artifact.byte_count,
                    artifact.path,
                    time.time(),
                ),
            )

    def record_iteration(
        self,
        command: str,
        *,
        world_hash: str = "",
        alignment_hash: str = "",
        sampling_hash: str = "",
        policy_hash: str = "",
        engine_hash: str = "",
        hardware_hash: str = "",
        provenance: Mapping[str, Any] | None = None,
    ) -> str:
        record = {
            "schema_version": self.schema_version,
            "command": command,
            "created_unix": time.time(),
            "world_hash": world_hash,
            "alignment_hash": alignment_hash,
            "sampling_hash": sampling_hash,
            "policy_hash": policy_hash,
            "engine_hash": engine_hash,
            "hardware_hash": hardware_hash,
            "provenance": dict(provenance or {}),
        }
        iteration_id = _hash_bytes(_canonical_json(record))
        with self._connect() as database:
            database.execute(
                """
                INSERT OR IGNORE INTO iterations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iteration_id,
                    command,
                    record["created_unix"],
                    world_hash,
                    alignment_hash,
                    sampling_hash,
                    policy_hash,
                    engine_hash,
                    hardware_hash,
                    json.dumps(record["provenance"], sort_keys=True),
                ),
            )
        return iteration_id

    def register_policy(self, policy: PolicyDescriptor) -> None:
        if not policy.id or not policy.content_hash:
            raise ValueError("policy id and content hash cannot be empty")
        descriptor = _canonical_json(asdict(policy)).decode("utf-8")
        with self._connect() as database:
            database.execute(
                """
                INSERT OR REPLACE INTO policies
                (fingerprint, id, content_hash, descriptor_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    _hex64(policy.fingerprint),
                    policy.id,
                    policy.content_hash,
                    descriptor,
                ),
            )

    def ingest_replay_trace(
        self,
        path: str | os.PathLike[str],
    ) -> ArtifactRef:
        """Copy a telemetry trace into immutable artifact storage."""

        trace_path = Path(path).expanduser().resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(
                f"physical replay trace is unavailable: {trace_path}"
            )
        reference = self.artifacts.put_bytes(
            trace_path.read_bytes(),
            kind="physical-replay-trace",
            media_type="application/x-npz",
            suffix=".npz",
        )
        self._register_artifact(reference)
        return reference

    def align(
        self,
        schema: ScenarioSchema,
        replay_artifact: Mapping[str, Any],
        evaluator: Callable[[mx.array], mx.array],
        *,
        initial_quantiles: mx.array | None = None,
        residual_count: int,
        config: SMCConfig = SMCConfig(),
        world_hash: str = "",
        engine_hash: str = "",
    ) -> AlignmentArtifact:
        if residual_count <= 0:
            raise ValueError("residual_count must be positive")
        replay_ref = self.artifacts.put_json(
            dict(replay_artifact),
            kind="physical-replay-manifest",
        )
        self._register_artifact(replay_ref)
        if initial_quantiles is None:
            initial_quantiles = deterministic_candidate_scenarios(
                len(schema.features),
                count=4096,
            )
        population = fit_alignment_smc(
            initial_quantiles,
            residual_count,
            evaluator,
            config=config,
        )
        # This is the declared artifact-publication boundary.
        quantiles = np.asarray(population.quantiles)
        weights = np.asarray(population.weights)
        residuals = np.asarray(population.replay_residuals)
        attempted_particle_count = int(initial_quantiles.shape[0])
        valid_particle_count = int(quantiles.shape[0])
        array_ref = self.artifacts.put_npz(
            kind="world-alignment-population-arrays",
            arrays={
                "quantiles": quantiles,
                "weights": weights,
                "replay_residuals": residuals,
            },
        )
        self._register_artifact(array_ref)
        metadata = {
            "schema_version": self.schema_version,
            "kind": "WorldAlignmentPopulation",
            "scenario_schema": schema.id,
            "schema_fingerprint": _hex64(schema.fingerprint),
            "particle_count": int(quantiles.shape[0]),
            "attempted_particle_count": attempted_particle_count,
            "rejected_particle_count": (
                attempted_particle_count - valid_particle_count
            ),
            "feature_count": int(quantiles.shape[1]),
            "rounds": config.rounds,
            "replay_artifact_hash": replay_ref.content_hash,
            "array_artifact_hash": array_ref.content_hash,
            "replay_residual_names": list(
                replay_artifact.get("residual_names", ())
            ),
            "multimodal": True,
        }
        metadata_ref = self.artifacts.put_json(
            metadata,
            kind="world-alignment-population",
        )
        self._register_artifact(metadata_ref)
        fingerprint = _fingerprint64(metadata_ref.content_hash)
        with self._connect() as database:
            database.execute(
                """
                INSERT OR REPLACE INTO alignments
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata_ref.content_hash,
                    _hex64(fingerprint),
                    _hex64(schema.fingerprint),
                    int(quantiles.shape[0]),
                    replay_ref.content_hash,
                    _canonical_json(metadata).decode("utf-8"),
                ),
            )
        self.record_iteration(
            "align",
            world_hash=world_hash,
            alignment_hash=metadata_ref.content_hash,
            engine_hash=engine_hash,
            hardware_hash=replay_ref.content_hash,
            provenance={
                "scenario_schema": schema.id,
                "schema_fingerprint": _hex64(schema.fingerprint),
                "array_artifact_hash": array_ref.content_hash,
            },
        )
        return AlignmentArtifact(
            content_hash=metadata_ref.content_hash,
            fingerprint=fingerprint,
            schema_fingerprint=schema.fingerprint,
            particle_count=int(quantiles.shape[0]),
            replay_artifact_hash=replay_ref.content_hash,
        )

    def load_alignment(
        self,
        content_hash: str,
    ) -> tuple[AlignmentArtifact, dict[str, np.ndarray]]:
        metadata = self.artifacts.read_json(content_hash)
        arrays = self.artifacts.read_npz(
            metadata["array_artifact_hash"]
        )
        artifact = AlignmentArtifact(
            content_hash=content_hash,
            fingerprint=_fingerprint64(content_hash),
            schema_fingerprint=_parse_hex64(
                metadata["schema_fingerprint"]
            ),
            particle_count=int(metadata["particle_count"]),
            replay_artifact_hash=metadata["replay_artifact_hash"],
        )
        return artifact, arrays

    def record_outcomes(
        self,
        schema: ScenarioSchema,
        outcomes: Sequence[EpisodeOutcome],
        *,
        command: str = "record-sim",
        world_hash: str = "",
        engine_hash: str = "",
        policy_hash: str = "",
    ) -> ArtifactRef:
        if not outcomes:
            raise ValueError("at least one outcome is required")
        for outcome in outcomes:
            if outcome.scenario_quantiles is None:
                outcome.scenario_quantiles = _normalize_scenario_values(
                    schema,
                    outcome.scenario_values,
                )
            if not outcome.missing_value_mask:
                outcome.missing_value_mask = {
                    feature.id: (
                        outcome.scenario_values.get(feature.id) is None
                    )
                    for feature in schema.features
                }
            outcome.validate(schema)
        manifest = {
            "schema_version": self.schema_version,
            "scenario_schema": schema.id,
            "schema_fingerprint": _hex64(schema.fingerprint),
            "outcomes": [outcome.as_manifest() for outcome in outcomes],
        }
        artifact = self.artifacts.put_json(
            manifest,
            kind=(
                "hardware-outcome-batch"
                if command == "ingest-real"
                else "simulation-outcome-batch"
            ),
        )
        self._register_artifact(artifact)
        with self._connect() as database:
            for outcome in outcomes:
                database.execute(
                    """
                    INSERT OR REPLACE INTO outcomes
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outcome.id,
                        outcome.run_id,
                        outcome.source,
                        outcome.scenario_schema,
                        _hex64(outcome.scenario_key),
                        _hex64(outcome.episode_counter),
                        _hex64(outcome.policy_fingerprint),
                        _hex64(outcome.task_fingerprint),
                        _hex64(outcome.embodiment_fingerprint),
                        int(outcome.success),
                        outcome.task_margin,
                        _canonical_json(
                            outcome.as_manifest()
                        ).decode("utf-8"),
                    ),
                )
        first = outcomes[0]
        self.record_iteration(
            command,
            world_hash=world_hash,
            alignment_hash=_hex64(first.alignment_fingerprint),
            sampling_hash=_hex64(first.feedback_fingerprint),
            policy_hash=policy_hash or _hex64(first.policy_fingerprint),
            engine_hash=engine_hash,
            hardware_hash=(
                artifact.content_hash
                if command == "ingest-real"
                else ""
            ),
            provenance={
                "outcome_artifact": artifact.content_hash,
                "episode_count": len(outcomes),
                "source": first.source,
            },
        )
        return artifact

    def ingest_hardware_manifest(
        self,
        schema: ScenarioSchema,
        path: str | os.PathLike[str],
    ) -> ArtifactRef:
        manifest_path = Path(path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else [payload]
        outcomes = [
            self._outcome_from_manifest(
                entry,
                schema,
                source="hardware",
            )
            for entry in entries
        ]
        return self.record_outcomes(
            schema,
            outcomes,
            command="ingest-real",
        )

    def ingest_simulation_manifest(
        self,
        schema: ScenarioSchema,
        path: str | os.PathLike[str],
    ) -> ArtifactRef:
        payload = json.loads(
            Path(path).expanduser().resolve().read_text(encoding="utf-8")
        )
        entries = payload.get("outcomes", payload)
        if not isinstance(entries, list):
            entries = [entries]
        outcomes = [
            self._outcome_from_manifest(
                entry,
                schema,
                source="simulation",
            )
            for entry in entries
        ]
        return self.record_outcomes(schema, outcomes)

    def _outcome_from_manifest(
        self,
        value: Mapping[str, Any],
        schema: ScenarioSchema,
        *,
        source: str,
    ) -> EpisodeOutcome:
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("unsupported hardware-outcome schema version")
        scenario_schema = value.get("scenario_schema", schema.id)
        scenario_values = dict(value.get("scenario_values", {}))
        if not all(
            isinstance(key, str)
            and (raw is None or isinstance(raw, (int, float)))
            for key, raw in scenario_values.items()
        ):
            raise ValueError("scenario_values must map names to numbers/null")
        quantiles = value.get("scenario_quantiles")
        if quantiles is None:
            quantiles = _normalize_scenario_values(
                schema,
                scenario_values,
            )
        return EpisodeOutcome(
            id=str(value["id"]),
            run_id=str(value["run_id"]),
            source=source,
            policy_fingerprint=_parse_hex64(
                value["policy_fingerprint"]
            ),
            task_fingerprint=_parse_hex64(
                value["task_fingerprint"]
            ),
            embodiment_fingerprint=_parse_hex64(
                value["embodiment_fingerprint"]
            ),
            scenario_schema=str(scenario_schema),
            policy_id=str(value.get("policy_id", "")),
            robot_id=str(value.get("robot_id", "")),
            task_id=str(value.get("task_id", "")),
            scenario_key=_parse_hex64(value.get("scenario_key")),
            episode_counter=_parse_hex64(
                value.get("episode_counter")
            ),
            family_fingerprint=_parse_hex64(
                value.get("family_fingerprint")
            ),
            alignment_fingerprint=_parse_hex64(
                value.get("alignment_fingerprint")
            ),
            feedback_fingerprint=_parse_hex64(
                value.get("feedback_fingerprint")
            ),
            termination=str(value.get("termination", "horizon")),
            success=bool(value["success"]),
            failure_tags=tuple(value.get("failure_tags", ())),
            physics_status=int(value.get("physics_status", 0)),
            step_count=int(value.get("step_count", 0)),
            episode_return=float(value.get("episode_return", 0.0)),
            task_margin=float(value.get("task_margin", 0.0)),
            safety_margin=float(value.get("safety_margin", 0.0)),
            duration_seconds=float(
                value.get("duration_seconds", 0.0)
            ),
            minimum_visibility=float(
                value.get("minimum_visibility", 0.0)
            ),
            integrated_contact_load=float(
                value.get("integrated_contact_load", 0.0)
            ),
            peak_contact_load=float(
                value.get("peak_contact_load", 0.0)
            ),
            scenario_values=scenario_values,
            missing_value_mask={
                str(key): bool(missing)
                for key, missing in value.get(
                    "missing_value_mask",
                    {
                        feature.id: scenario_values.get(feature.id) is None
                        for feature in schema.features
                    },
                ).items()
            },
            scenario_quantiles=[
                float(component) for component in quantiles
            ],
            artifacts=tuple(value.get("artifacts", ())),
        )

    def outcomes(
        self,
        *,
        policy_fingerprint: int,
        task_fingerprint: int,
        embodiment_fingerprint: int,
    ) -> list[EpisodeOutcome]:
        with self._connect() as database:
            rows = database.execute(
                """
                SELECT manifest_json FROM outcomes
                WHERE policy_fingerprint = ?
                  AND task_fingerprint = ?
                  AND embodiment_fingerprint = ?
                ORDER BY id
                """,
                (
                    _hex64(policy_fingerprint),
                    _hex64(task_fingerprint),
                    _hex64(embodiment_fingerprint),
                ),
            ).fetchall()
        return [
            _outcome_from_stored_manifest(
                json.loads(row["manifest_json"])
            )
            for row in rows
        ]

    def fit_feedback(
        self,
        schema: ScenarioSchema,
        *,
        policy_fingerprint: int,
        task_fingerprint: int,
        embodiment_fingerprint: int,
        steps: int = 300,
        candidate_count: int = 65536,
        maximum_regions: int = 64,
        seed: int = 1,
    ) -> FeedbackArtifact:
        outcomes = self.outcomes(
            policy_fingerprint=policy_fingerprint,
            task_fingerprint=task_fingerprint,
            embodiment_fingerprint=embodiment_fingerprint,
        )
        simulation_count = sum(
            outcome.source == "simulation" for outcome in outcomes
        )
        if simulation_count == 0:
            raise ValueError(
                "feedback fitting requires simulation outcomes for the "
                "simulation head"
            )
        quantiles = np.asarray(
            [outcome.scenario_quantiles for outcome in outcomes],
            dtype=np.float32,
        )
        success = np.asarray(
            [outcome.success for outcome in outcomes],
            dtype=np.float32,
        )
        margins = np.asarray(
            [outcome.task_margin for outcome in outcomes],
            dtype=np.float32,
        )
        hardware_mask = np.asarray(
            [outcome.source == "hardware" for outcome in outcomes],
            dtype=np.float32,
        )
        ensemble = FiveMemberFailureEnsemble(
            len(schema.features),
            seed=seed,
        )
        history = ensemble.fit(
            FailureTrainingBatch(
                quantiles=mx.array(quantiles),
                success=mx.array(success),
                task_margin=mx.array(margins),
                hardware_mask=mx.array(hardware_mask),
            ),
            steps=steps,
        )
        regions, scores = compile_feedback_regions(
            ensemble,
            candidate_count=candidate_count,
            maximum_regions=maximum_regions,
        )
        simulation_quantiles = mx.array(
            quantiles[hardware_mask == 0.0],
        )
        observed_scenario_scores = ensemble.score(
            simulation_quantiles
        )
        mx.eval(
            observed_scenario_scores.simulation_success,
            observed_scenario_scores.uncertainty_score,
            scores.simulation_success,
            scores.uncertainty_score,
        )
        observed_predictive_variance = float(
            mx.mean(observed_scenario_scores.uncertainty_score).item()
        )
        candidate_predictive_variance = float(
            mx.mean(scores.uncertainty_score).item()
        )
        predicted_hardware_rate: float | None = None
        candidate_grid_hardware_rate: float | None = None
        if observed_scenario_scores.hardware_success is not None:
            mx.eval(
                observed_scenario_scores.hardware_success,
                scores.hardware_success,
            )
            predicted_hardware_rate = float(
                mx.mean(
                    observed_scenario_scores.hardware_success
                ).item()
            )
            candidate_grid_hardware_rate = float(
                mx.mean(scores.hardware_success).item()
            )
        # MLX parameter publication is a chunk/final-artifact boundary.
        model_arrays = {
            name: np.asarray(array)
            for name, array in ensemble.parameters.items()
        }
        model_ref = self.artifacts.put_npz(
            kind="policy-failure-ensemble",
            arrays=model_arrays,
        )
        self._register_artifact(model_ref)
        metadata = {
            "schema_version": self.schema_version,
            "kind": "WorldFeedbackProgram",
            "scenario_schema": schema.id,
            "schema_fingerprint": _hex64(schema.fingerprint),
            "policy_fingerprint": _hex64(policy_fingerprint),
            "task_fingerprint": _hex64(task_fingerprint),
            "embodiment_fingerprint": _hex64(
                embodiment_fingerprint
            ),
            "source_model_hash": model_ref.content_hash,
            "hardware_available": ensemble.hardware_available,
            "hardware_evidence_count": (
                ensemble.hardware_evidence_count
            ),
            "simulation_episode_count": simulation_count,
            "hardware_episode_count": len(outcomes) - simulation_count,
            "candidate_count": candidate_count,
            "training_history": history,
            "mixture": {
                "broad": 0.5,
                "failure": 0.3,
                "uncertainty": 0.2,
            },
            "regions": [
                {
                    "kind": (
                        "failure" if region.kind == 0 else "uncertainty"
                    ),
                    "weight": region.weight,
                    "lower": region.lower,
                    "upper": region.upper,
                    "score": region.score,
                }
                for region in regions
            ],
            "hardware_prediction": (
                "available_with_uncertainty"
                if ensemble.hardware_available
                else "unavailable_sim_only"
            ),
            "observed_scenario_predictive_variance": (
                observed_predictive_variance
            ),
            "candidate_grid_predictive_variance": (
                candidate_predictive_variance
            ),
            "observed_scenario_simulation_success_prediction": float(
                mx.mean(
                    observed_scenario_scores.simulation_success
                ).item()
            ),
            "observed_scenario_hardware_success_prediction": (
                predicted_hardware_rate
            ),
            "candidate_grid_hardware_success_prediction": (
                candidate_grid_hardware_rate
            ),
        }
        feedback_ref = self.artifacts.put_json(
            metadata,
            kind="world-feedback-program",
        )
        self._register_artifact(feedback_ref)
        fingerprint = _fingerprint64(feedback_ref.content_hash)
        with self._connect() as database:
            database.execute(
                """
                INSERT OR REPLACE INTO feedback_programs
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_ref.content_hash,
                    model_ref.content_hash,
                    _hex64(fingerprint),
                    _hex64(schema.fingerprint),
                    _hex64(policy_fingerprint),
                    _hex64(task_fingerprint),
                    _hex64(embodiment_fingerprint),
                    len(regions),
                    int(ensemble.hardware_available),
                    _canonical_json(metadata).decode("utf-8"),
                ),
            )
        self.record_iteration(
            "fit-feedback",
            sampling_hash=feedback_ref.content_hash,
            policy_hash=_hex64(policy_fingerprint),
            hardware_hash=(
                "available"
                if ensemble.hardware_available
                else ""
            ),
            provenance={
                "task_fingerprint": _hex64(task_fingerprint),
                "embodiment_fingerprint": _hex64(
                    embodiment_fingerprint
                ),
                "model_artifact": model_ref.content_hash,
                "candidate_count": candidate_count,
            },
        )
        return FeedbackArtifact(
            content_hash=feedback_ref.content_hash,
            model_content_hash=model_ref.content_hash,
            fingerprint=fingerprint,
            schema_fingerprint=schema.fingerprint,
            policy_fingerprint=policy_fingerprint,
            task_fingerprint=task_fingerprint,
            embodiment_fingerprint=embodiment_fingerprint,
            region_count=len(regions),
            hardware_available=ensemble.hardware_available,
            hardware_evidence_count=(
                ensemble.hardware_evidence_count
            ),
            hardware_predictive_variance=(
                observed_predictive_variance
                if ensemble.hardware_available
                else None
            ),
        )

    def configure_world_family(
        self,
        family: FrankaPickPlaceWorldFamily,
        *,
        alignment_hash: str,
        feedback_hash: str | None = None,
    ) -> None:
        alignment, arrays = self.load_alignment(alignment_hash)
        if alignment.schema_fingerprint != family.scenario_schema.fingerprint:
            raise ValueError("alignment and world-family schemas differ")
        arguments: dict[str, Any] = {
            "alignment_fingerprint": alignment.fingerprint,
            "particle_quantiles": arrays["quantiles"],
            "particle_weights": arrays["weights"],
            "particle_residuals": arrays["replay_residuals"],
        }
        if feedback_hash:
            feedback = self.artifacts.read_json(feedback_hash)
            if _parse_hex64(
                feedback["schema_fingerprint"]
            ) != family.scenario_schema.fingerprint:
                raise ValueError("feedback and world-family schemas differ")
            regions = feedback["regions"]
            bounds = np.asarray(
                [
                    list(zip(region["lower"], region["upper"]))
                    for region in regions
                ],
                dtype=np.float32,
            )
            arguments.update(
                feedback_fingerprint=_fingerprint64(feedback_hash),
                region_kinds=np.asarray(
                    [
                        0 if region["kind"] == "failure" else 1
                        for region in regions
                    ],
                    dtype=np.uint32,
                ),
                region_weights=np.asarray(
                    [region["weight"] for region in regions],
                    dtype=np.float32,
                ),
                region_bounds=bounds,
            )
        family.configure_sampling(**arguments)

    def evaluate(
        self,
        schema: ScenarioSchema,
        *,
        task_fingerprint: int,
        policy_fingerprints: Sequence[int],
    ) -> dict[str, Any]:
        if len(policy_fingerprints) < 2:
            raise ValueError("paired evaluation requires at least two policies")
        all_outcomes: dict[int, list[EpisodeOutcome]] = {}
        for policy in policy_fingerprints:
            with self._connect() as database:
                rows = database.execute(
                    """
                    SELECT manifest_json FROM outcomes
                    WHERE policy_fingerprint = ?
                      AND task_fingerprint = ?
                    ORDER BY id
                    """,
                    (_hex64(policy), _hex64(task_fingerprint)),
                ).fetchall()
            all_outcomes[policy] = [
                _outcome_from_stored_manifest(
                    json.loads(row["manifest_json"])
                )
                for row in rows
            ]
        paired_keys: set[int] | None = None
        for outcomes in all_outcomes.values():
            keys = {
                outcome.scenario_key
                for outcome in outcomes
                if outcome.source == "simulation"
                and outcome.scenario_key != 0
            }
            paired_keys = keys if paired_keys is None else paired_keys & keys
        paired_keys = paired_keys or set()
        ordered_paired_keys = sorted(paired_keys)
        axis_names = (
            "appearance",
            "object_configuration",
            "clutter",
            "physics",
            "robot_controller",
            "camera",
        )

        def aggregate_simulation(
            outcomes: Sequence[EpisodeOutcome],
        ) -> dict[int, tuple[float, float]]:
            rows: dict[int, list[EpisodeOutcome]] = {}
            for outcome in outcomes:
                if (
                    outcome.source == "simulation"
                    and outcome.scenario_key in paired_keys
                ):
                    rows.setdefault(outcome.scenario_key, []).append(
                        outcome
                    )
            return {
                key: (
                    sum(value.success for value in values) / len(values),
                    sum(value.task_margin for value in values)
                    / len(values),
                )
                for key, values in rows.items()
            }

        def scenario_slices(
            paired: Sequence[EpisodeOutcome],
        ) -> dict[str, dict[str, float | int]]:
            buckets: dict[str, list[EpisodeOutcome]] = {}
            for outcome in paired:
                quantiles = outcome.scenario_quantiles
                if quantiles is None:
                    buckets.setdefault("unknown", []).append(outcome)
                    continue
                tails = [
                    index
                    for index, quantile in enumerate(quantiles)
                    if quantile < 0.05 or quantile > 0.95
                ]
                bucket = "tail_ood" if tails else "central_id"
                buckets.setdefault(bucket, []).append(outcome)
                for axis in sorted(
                    {schema.features[index].axis for index in tails}
                ):
                    name = (
                        axis_names[axis]
                        if 0 <= axis < len(axis_names)
                        else f"axis_{axis}"
                    )
                    buckets.setdefault(
                        f"tail_ood:{name}",
                        [],
                    ).append(outcome)
            return {
                name: {
                    "episodes": len(values),
                    "success_rate": (
                        sum(value.success for value in values)
                        / len(values)
                    ),
                    "mean_task_margin": (
                        sum(value.task_margin for value in values)
                        / len(values)
                    ),
                }
                for name, values in sorted(buckets.items())
                if values
            }

        summaries = []
        simulation_aggregates: dict[
            int,
            dict[int, tuple[float, float]],
        ] = {}
        for policy, outcomes in all_outcomes.items():
            paired = [
                outcome
                for outcome in outcomes
                if outcome.source == "simulation"
                and outcome.scenario_key in paired_keys
            ]
            hardware = [
                outcome
                for outcome in outcomes
                if outcome.source == "hardware"
            ]
            aggregates = aggregate_simulation(outcomes)
            simulation_aggregates[policy] = aggregates
            with self._connect() as database:
                feedback_row = database.execute(
                    """
                    SELECT metadata_json FROM feedback_programs
                    WHERE policy_fingerprint = ?
                      AND task_fingerprint = ?
                    ORDER BY rowid DESC
                    LIMIT 1
                    """,
                    (
                        _hex64(policy),
                        _hex64(task_fingerprint),
                    ),
                ).fetchone()
            feedback_metadata = (
                json.loads(feedback_row["metadata_json"])
                if feedback_row is not None
                else {}
            )
            predicted_hardware = feedback_metadata.get(
                "observed_scenario_hardware_success_prediction"
            )
            paired_success_rate = (
                sum(value[0] for value in aggregates.values())
                / len(aggregates)
                if aggregates
                else None
            )
            paired_task_margin = (
                sum(value[1] for value in aggregates.values())
                / len(aggregates)
                if aggregates
                else None
            )
            hardware_success_rate = (
                sum(outcome.success for outcome in hardware)
                / len(hardware)
                if hardware
                else None
            )
            failure_tags = sorted(
                {
                    tag
                    for outcome in outcomes
                    for tag in outcome.failure_tags
                }
            )
            simulation_failure_rates = {
                tag: (
                    sum(
                        tag in outcome.failure_tags
                        for outcome in paired
                    )
                    / len(paired)
                    if paired
                    else 0.0
                )
                for tag in failure_tags
            }
            hardware_failure_rates = {
                tag: (
                    sum(
                        tag in outcome.failure_tags
                        for outcome in hardware
                    )
                    / len(hardware)
                    if hardware
                    else 0.0
                )
                for tag in failure_tags
            }
            summaries.append(
                {
                    "policy_fingerprint": _hex64(policy),
                    "paired_simulation_episodes": len(paired),
                    "paired_simulation_success_rate": paired_success_rate,
                    "paired_simulation_mean_task_margin": (
                        paired_task_margin
                    ),
                    "hardware_episodes": len(hardware),
                    "hardware_success_rate": hardware_success_rate,
                    "predicted_hardware_success_rate": (
                        predicted_hardware
                    ),
                    "predictive_variance": feedback_metadata.get(
                        "observed_scenario_predictive_variance"
                    ),
                    "hardware_evidence_count": len(hardware),
                    "hardware_prediction": (
                        "available_with_uncertainty"
                        if predicted_hardware is not None
                        else "unavailable_sim_only"
                    ),
                    "simulation_hardware_calibration_error": (
                        abs(
                            paired_success_rate
                            - hardware_success_rate
                        )
                        if paired_success_rate is not None
                        and hardware_success_rate is not None
                        else None
                    ),
                    "model_hardware_calibration_error": (
                        abs(
                            predicted_hardware
                            - hardware_success_rate
                        )
                        if predicted_hardware is not None
                        and hardware_success_rate is not None
                        else None
                    ),
                    "simulation_failure_rates": (
                        simulation_failure_rates
                    ),
                    "hardware_failure_rates": hardware_failure_rates,
                    "scenario_slices": scenario_slices(paired),
                }
            )
        summary_by_policy = {
            _parse_hex64(summary["policy_fingerprint"]): summary
            for summary in summaries
        }
        pairwise = []
        ordered_policies = sorted(policy_fingerprints)
        for left_index, left in enumerate(ordered_policies):
            for right in ordered_policies[left_index + 1 :]:
                left_aggregates = simulation_aggregates[left]
                right_aggregates = simulation_aggregates[right]
                common = [
                    key
                    for key in ordered_paired_keys
                    if key in left_aggregates
                    and key in right_aggregates
                ]
                if not common:
                    continue
                success_deltas = [
                    left_aggregates[key][0]
                    - right_aggregates[key][0]
                    for key in common
                ]
                margin_deltas = [
                    left_aggregates[key][1]
                    - right_aggregates[key][1]
                    for key in common
                ]
                success_interval = _bootstrap_mean_interval(
                    success_deltas,
                    seed=left ^ ((right << 17) | (right >> 47)),
                )
                margin_interval = _bootstrap_mean_interval(
                    margin_deltas,
                    seed=right ^ ((left << 29) | (left >> 35)),
                )
                left_hardware = summary_by_policy[left][
                    "hardware_success_rate"
                ]
                right_hardware = summary_by_policy[right][
                    "hardware_success_rate"
                ]
                pairwise.append(
                    {
                        "left_policy_fingerprint": _hex64(left),
                        "right_policy_fingerprint": _hex64(right),
                        "paired_scenario_count": len(common),
                        "paired_success_delta": float(
                            np.mean(success_deltas)
                        ),
                        "paired_success_interval_95": list(
                            success_interval
                        ),
                        "paired_task_margin_delta": float(
                            np.mean(margin_deltas)
                        ),
                        "paired_task_margin_interval_95": list(
                            margin_interval
                        ),
                        "hardware_success_delta": (
                            left_hardware - right_hardware
                            if left_hardware is not None
                            and right_hardware is not None
                            else None
                        ),
                    }
                )

        hardware_rank_summaries = [
            summary
            for summary in summaries
            if summary["hardware_success_rate"] is not None
            and summary["paired_simulation_success_rate"] is not None
        ]
        real_values = [
            summary["hardware_success_rate"]
            for summary in hardware_rank_summaries
        ]
        simulation_values = [
            summary["paired_simulation_success_rate"]
            for summary in hardware_rank_summaries
        ]
        mmrv = (
            _mean_maximum_rank_violation(
                real_values,
                simulation_values,
            )
            if len(hardware_rank_summaries) >= 2
            else None
        )
        rank_correlation = _rank_correlation(
            real_values,
            simulation_values,
        )
        model_calibration_values = [
            summary["model_hardware_calibration_error"]
            for summary in summaries
            if summary["model_hardware_calibration_error"] is not None
        ]
        simulation_calibration_values = [
            summary["simulation_hardware_calibration_error"]
            for summary in summaries
            if summary[
                "simulation_hardware_calibration_error"
            ] is not None
        ]
        overlap_numerator = 0.0
        overlap_denominator = 0.0
        overlap_evidence = False
        for summary in summaries:
            if summary["hardware_episodes"] == 0:
                continue
            tags = set(summary["simulation_failure_rates"]) | set(
                summary["hardware_failure_rates"]
            )
            for tag in tags:
                simulation_rate = summary[
                    "simulation_failure_rates"
                ].get(tag, 0.0)
                hardware_rate = summary[
                    "hardware_failure_rates"
                ].get(tag, 0.0)
                overlap_numerator += min(
                    simulation_rate,
                    hardware_rate,
                )
                overlap_denominator += max(
                    simulation_rate,
                    hardware_rate,
                )
                overlap_evidence = True
        failure_region_overlap = (
            overlap_numerator / overlap_denominator
            if overlap_denominator > 0.0
            else 1.0
            if overlap_evidence
            else None
        )
        summaries.sort(
            key=lambda summary: (
                summary["predicted_hardware_success_rate"]
                if summary[
                    "predicted_hardware_success_rate"
                ] is not None
                else summary["paired_simulation_success_rate"]
                if summary["paired_simulation_success_rate"] is not None
                else -1.0
            ),
            reverse=True,
        )
        report = {
            "schema_version": 2,
            "kind": "PolicyEvaluationReport",
            "scenario_schema": schema.id,
            "schema_fingerprint": _hex64(schema.fingerprint),
            "task_fingerprint": _hex64(task_fingerprint),
            "paired_scenario_count": len(paired_keys),
            "hardware_evidence_count": sum(
                summary["hardware_episodes"]
                for summary in summaries
            ),
            "hardware_prediction_available": any(
                summary["predicted_hardware_success_rate"] is not None
                for summary in summaries
            ),
            "policies": summaries,
            "pairwise": pairwise,
            "mean_maximum_rank_violation": mmrv,
            "rank_correlation": rank_correlation,
            "model_calibration_error": (
                float(np.mean(model_calibration_values))
                if model_calibration_values
                else None
            ),
            "simulation_hardware_calibration_error": (
                float(np.mean(simulation_calibration_values))
                if simulation_calibration_values
                else None
            ),
            "failure_region_overlap": failure_region_overlap,
            "relative_ordering": [
                summary["policy_fingerprint"] for summary in summaries
            ],
        }
        report_ref = self.artifacts.put_json(
            report,
            kind="policy-evaluation-report",
        )
        self._register_artifact(report_ref)
        report["artifact_hash"] = report_ref.content_hash
        self.record_iteration(
            "evaluate",
            policy_hash=",".join(
                _hex64(policy) for policy in policy_fingerprints
            ),
            provenance={
                "report_artifact": report_ref.content_hash,
                "paired_scenario_count": len(paired_keys),
                "hardware_evidence_count": report[
                    "hardware_evidence_count"
                ],
                "mean_maximum_rank_violation": mmrv,
            },
        )
        return report


def make_affine_replay_evaluator(
    modes: Sequence[Mapping[str, Any]],
    *,
    feature_count: int,
) -> tuple[Callable[[mx.array], mx.array], int]:
    """Compile bounded replay-residual surrogates into an MLX evaluator.

    Each mode is a local replay residual field with ``matrix[residual,feature]``
    and ``bias[residual]``. Multiple modes preserve discontinuous contact
    explanations; each particle uses the lowest-residual mode.
    """

    if not modes:
        raise ValueError("at least one replay residual mode is required")
    matrices = np.asarray([mode["matrix"] for mode in modes], np.float32)
    biases = np.asarray([mode["bias"] for mode in modes], np.float32)
    if (
        matrices.ndim != 3
        or matrices.shape[2] != feature_count
        or biases.shape != matrices.shape[:2]
    ):
        raise ValueError(
            "replay mode matrices must be [mode,residual,feature] and "
            "biases [mode,residual]"
        )
    matrix = mx.array(matrices)
    bias = mx.array(biases)

    def evaluator(quantiles: mx.array) -> mx.array:
        residuals = mx.einsum("nf,mrf->mnr", quantiles, matrix)
        residuals += bias[:, None, :]
        losses = mx.sum(residuals * residuals, axis=-1)
        best = mx.argmin(losses, axis=0)
        return mx.take_along_axis(
            residuals,
            best[None, :, None],
            axis=0,
        )[0]

    return evaluator, int(matrices.shape[1])


def drain_completed_episode_records(
    records: CompactedEpisodeRecords,
    schema: ScenarioSchema,
    *,
    run_id: str,
    policy_fingerprint: int,
    task_fingerprint: int,
    embodiment_fingerprint: int,
    family_fingerprint: int,
    failure_tags: Sequence[str] = (),
) -> list[EpisodeOutcome]:
    """Materialize only dense completed prefixes at a rollout-chunk boundary."""

    if not run_id:
        raise ValueError("run_id cannot be empty")
    mx.eval(*records)
    words = np.asarray(records.outcome_words)
    task = np.asarray(records.task_values)
    interaction = np.asarray(records.interaction_values)
    headers = np.asarray(records.scenario_headers)
    values = np.asarray(records.scenario_values)
    identities = np.asarray(records.scenario_identities)
    counts = np.asarray(records.valid_count)
    if words.ndim == 2:
        words = words[None, ...]
        task = task[None, ...]
        interaction = interaction[None, ...]
        headers = headers[None, ...]
        values = values[None, ...]
        identities = identities[None, ...]
        counts = counts.reshape((1,))
    if (
        words.ndim != 3
        or words.shape[-1] != 12
        or values.shape[-2] != len(schema.features)
        or counts.shape != (words.shape[0],)
    ):
        raise ValueError("compacted episode buffers have invalid shapes")
    termination_names = (
        "success",
        "horizon",
        "policy",
        "physics",
        "safety",
        "external",
    )
    outcomes: list[EpisodeOutcome] = []
    for chunk in range(words.shape[0]):
        valid_count = int(counts[chunk])
        if not 0 <= valid_count <= words.shape[1]:
            raise ValueError("compacted episode count exceeds capacity")
        for row in range(valid_count):
            record = words[chunk, row]
            if int(record[4]) == 0 and int(record[7]) != 0:
                # Physics failures belong in runtime diagnostics, not in the
                # policy/outcome evidence used for alignment or evaluation.
                continue
            scenario_key = int(record[0]) | (int(record[1]) << 32)
            episode_counter = int(record[2]) | (
                int(record[3]) << 32
            )
            termination_index = int(record[5])
            if termination_index >= len(termination_names):
                raise ValueError("compacted termination code is invalid")
            failure_mask = int(record[8]) | (
                int(record[9]) << 32
            )
            provenance = headers[chunk, row, 1]
            alignment = int(provenance[0]) | (
                int(provenance[1]) << 32
            )
            feedback = int(provenance[2]) | (
                int(provenance[3]) << 32
            )
            raw_values = values[chunk, row, :, 0]
            quantiles = values[chunk, row, :, 1]
            flags = identities[chunk, row, :, 3]
            named_values = {
                feature.id: (
                    None
                    if int(flags[index]) & 4
                    else float(raw_values[index])
                )
                for index, feature in enumerate(schema.features)
            }
            outcome_identity = {
                "run_id": run_id,
                "scenario_key": _hex64(scenario_key),
                "episode_counter": _hex64(episode_counter),
                "policy_fingerprint": _hex64(policy_fingerprint),
            }
            outcomes.append(
                EpisodeOutcome(
                    id=_hash_bytes(
                        _canonical_json(outcome_identity)
                    ),
                    run_id=run_id,
                    source=(
                        "hardware"
                        if int(record[4]) == 1
                        else "simulation"
                    ),
                    policy_fingerprint=policy_fingerprint,
                    task_fingerprint=task_fingerprint,
                    embodiment_fingerprint=embodiment_fingerprint,
                    scenario_schema=schema.id,
                    scenario_key=scenario_key,
                    episode_counter=episode_counter,
                    family_fingerprint=family_fingerprint,
                    alignment_fingerprint=alignment,
                    feedback_fingerprint=feedback,
                    termination=termination_names[
                        termination_index
                    ],
                    success=bool(record[6]),
                    failure_tags=tuple(
                        tag
                        for index, tag in enumerate(failure_tags)
                        if failure_mask & (1 << index)
                    ),
                    physics_status=int(record[7]),
                    step_count=int(record[10]),
                    episode_return=float(task[chunk, row, 0]),
                    task_margin=float(task[chunk, row, 1]),
                    safety_margin=float(task[chunk, row, 2]),
                    duration_seconds=float(task[chunk, row, 3]),
                    minimum_visibility=float(
                        interaction[chunk, row, 0]
                    ),
                    integrated_contact_load=float(
                        interaction[chunk, row, 1]
                    ),
                    peak_contact_load=float(
                        interaction[chunk, row, 2]
                    ),
                    scenario_values=named_values,
                    missing_value_mask={
                        feature.id: bool(int(flags[index]) & 4)
                        for index, feature in enumerate(
                            schema.features
                        )
                    },
                    scenario_quantiles=[
                        float(value) for value in quantiles
                    ],
                )
            )
    return outcomes


def _normalize_scenario_values(
    schema: ScenarioSchema,
    values: Mapping[str, float | None],
) -> list[float]:
    quantiles = []
    normal = NormalDist()
    for feature in schema.features:
        raw = values.get(feature.id)
        if raw is None:
            quantiles.append(0.5)
            continue
        p0, p1, p2, p3 = feature.parameters
        if feature.distribution == 0:
            quantile = 0.5
        elif feature.distribution == 1:
            quantile = (float(raw) - p0) / max(p1 - p0, 1.0e-12)
        elif feature.distribution == 2:
            quantile = (
                math.log(max(float(raw), 1.0e-30)) - math.log(p0)
            ) / max(math.log(p1) - math.log(p0), 1.0e-12)
        elif feature.distribution == 3:
            clamped = min(max(float(raw), p2), p3)
            quantile = normal.cdf((clamped - p0) / p1)
        else:
            # Categorical alternatives require an explicit quantile from the
            # capture adapter. Missing/unknown categories remain centered.
            quantile = 0.5
        quantiles.append(float(min(max(quantile, 0.0), 1.0)))
    return quantiles


def _outcome_from_stored_manifest(value: Mapping[str, Any]) -> EpisodeOutcome:
    return EpisodeOutcome(
        id=str(value["id"]),
        run_id=str(value["run_id"]),
        source=str(value["source"]),
        policy_fingerprint=_parse_hex64(value["policy_fingerprint"]),
        task_fingerprint=_parse_hex64(value["task_fingerprint"]),
        embodiment_fingerprint=_parse_hex64(
            value["embodiment_fingerprint"]
        ),
        scenario_schema=str(value["scenario_schema"]),
        policy_id=str(value.get("policy_id", "")),
        robot_id=str(value.get("robot_id", "")),
        task_id=str(value.get("task_id", "")),
        scenario_key=_parse_hex64(value["scenario_key"]),
        episode_counter=_parse_hex64(value["episode_counter"]),
        family_fingerprint=_parse_hex64(value["family_fingerprint"]),
        alignment_fingerprint=_parse_hex64(
            value["alignment_fingerprint"]
        ),
        feedback_fingerprint=_parse_hex64(
            value["feedback_fingerprint"]
        ),
        termination=str(value["termination"]),
        success=bool(value["success"]),
        failure_tags=tuple(value["failure_tags"]),
        physics_status=int(value["physics_status"]),
        step_count=int(value["step_count"]),
        episode_return=float(value["episode_return"]),
        task_margin=float(value["task_margin"]),
        safety_margin=float(value["safety_margin"]),
        duration_seconds=float(value["duration_seconds"]),
        minimum_visibility=float(value["minimum_visibility"]),
        integrated_contact_load=float(
            value["integrated_contact_load"]
        ),
        peak_contact_load=float(value["peak_contact_load"]),
        scenario_values=dict(value["scenario_values"]),
        missing_value_mask={
            str(key): bool(missing)
            for key, missing in value.get(
                "missing_value_mask",
                {},
            ).items()
        },
        scenario_quantiles=(
            [float(component) for component in value["scenario_quantiles"]]
            if value.get("scenario_quantiles") is not None
            else None
        ),
        artifacts=tuple(value["artifacts"]),
    )


__all__ = [
    "AlignmentArtifact",
    "ArtifactRef",
    "ContentAddressedArtifactStore",
    "EpisodeOutcome",
    "FeedbackArtifact",
    "PolicyDescriptor",
    "R2S2RCoordinator",
    "drain_completed_episode_records",
    "make_affine_replay_evaluator",
]
