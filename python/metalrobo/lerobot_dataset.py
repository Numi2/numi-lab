"""Pinned, season-safe LeRobot ingestion for tactile imitation learning.

The loader intentionally has no dependency on PyTorch or the LeRobot Python
package. Numeric columns are streamed through Arrow, decoded into bounded
season-sized arrays, and handed to MLX by the trainer. Video decoding uses
PyAV only when callers explicitly request frames.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .tactile_stream import TactileStreamContract


ROBOT_DATASET_MANIFEST_SCHEMA = (
    "metalrobo.robot_dataset_manifest"
)
ROBOT_DATASET_MANIFEST_FORMAT = 1
SPLIT_ALGORITHM = "sha256-season-rank"
ORIGAMI_PINNED_REVISION = (
    "8194af6b9341dac7686c2f29704ff893e6f2f95e"
)
Split = Literal["train", "validation", "test"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_SPLITS: tuple[Split, ...] = (
    "train",
    "validation",
    "test",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_tuple(
    values: Any,
    *,
    dimensions: int,
    label: str,
) -> tuple[float, ...]:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size != dimensions or not np.isfinite(result).all():
        raise ValueError(
            f"{label} must contain {dimensions} finite values"
        )
    return tuple(float(value) for value in result)


@dataclass(frozen=True, slots=True)
class DatasetFeature:
    key: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def dimensions(self) -> int:
        return math.prod(self.shape)

    def validate(self) -> None:
        if (
            not self.key
            or self.dtype != "float32"
            or not self.shape
            or any(dimension <= 0 for dimension in self.shape)
        ):
            raise ValueError("robot dataset feature is invalid")


@dataclass(frozen=True, slots=True)
class FeatureNormalization:
    count: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    def validate(self, dimensions: int) -> None:
        if self.count <= 0:
            raise ValueError("normalization count must be positive")
        values = (
            self.mean,
            self.std,
            self.minimum,
            self.maximum,
        )
        if any(len(value) != dimensions for value in values):
            raise ValueError(
                "normalization dimensions do not match feature"
            )
        packed = np.asarray(values, dtype=np.float64)
        if not np.isfinite(packed).all():
            raise ValueError(
                "normalization contains non-finite values"
            )
        if np.any(np.asarray(self.std) < 0.0):
            raise ValueError(
                "normalization standard deviation is negative"
            )
        if np.any(
            np.asarray(self.minimum) > np.asarray(self.maximum)
        ):
            raise ValueError("normalization range is inverted")

    def normalize(
        self,
        value: npt.ArrayLike,
        *,
        epsilon: float = 1.0e-6,
    ) -> npt.NDArray[np.float32]:
        array = np.asarray(value, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.maximum(
            np.asarray(self.std, dtype=np.float32),
            np.float32(epsilon),
        )
        return np.ascontiguousarray((array - mean) / scale)

    def denormalize(
        self,
        value: npt.ArrayLike,
        *,
        epsilon: float = 1.0e-6,
    ) -> npt.NDArray[np.float32]:
        array = np.asarray(value, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.maximum(
            np.asarray(self.std, dtype=np.float32),
            np.float32(epsilon),
        )
        return np.ascontiguousarray(array * scale + mean)


@dataclass(frozen=True, slots=True)
class DatasetSeason:
    name: str
    relative_root: str
    split: Split
    episodes: int
    frames: int
    fps: float
    info_sha256: str
    stats_sha256: str | None

    def validate(self) -> None:
        relative = Path(self.relative_root)
        if (
            not self.name
            or not self.relative_root
            or relative.is_absolute()
            or ".." in relative.parts
            or self.split not in _SPLITS
            or self.episodes <= 0
            or self.frames <= 0
            or not math.isfinite(self.fps)
            or self.fps <= 0.0
            or _SHA256.fullmatch(self.info_sha256) is None
            or (
                self.stats_sha256 is not None
                and _SHA256.fullmatch(self.stats_sha256) is None
            )
        ):
            raise ValueError("robot dataset season is invalid")


@dataclass(frozen=True, slots=True)
class RobotDatasetManifest:
    source_repository: str
    source_revision: str
    stream_fingerprint: str
    validation_fraction: float
    test_fraction: float
    state: DatasetFeature
    action: DatasetFeature
    wrench: DatasetFeature
    state_normalization: FeatureNormalization
    action_normalization: FeatureNormalization
    wrench_normalization: FeatureNormalization
    seasons: tuple[DatasetSeason, ...]
    schema: str = ROBOT_DATASET_MANIFEST_SCHEMA
    format_version: int = ROBOT_DATASET_MANIFEST_FORMAT

    def validate(
        self,
        *,
        dataset_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if (
            self.schema != ROBOT_DATASET_MANIFEST_SCHEMA
            or self.format_version
            != ROBOT_DATASET_MANIFEST_FORMAT
            or not self.source_repository
            or _GIT_REVISION.fullmatch(self.source_revision) is None
            or _SHA256.fullmatch(self.stream_fingerprint) is None
            or not 0.0 <= self.validation_fraction < 1.0
            or not 0.0 <= self.test_fraction < 1.0
            or self.validation_fraction + self.test_fraction >= 1.0
            or not self.seasons
        ):
            raise ValueError("robot dataset manifest header is invalid")
        self.state.validate()
        self.action.validate()
        self.wrench.validate()
        self.state_normalization.validate(self.state.dimensions)
        self.action_normalization.validate(self.action.dimensions)
        self.wrench_normalization.validate(self.wrench.dimensions)
        names = [season.name for season in self.seasons]
        roots = [season.relative_root for season in self.seasons]
        if (
            len(set(names)) != len(names)
            or len(set(roots)) != len(roots)
        ):
            raise ValueError(
                "robot dataset manifest contains duplicate seasons"
            )
        root = (
            Path(dataset_root).expanduser().resolve()
            if dataset_root is not None
            else None
        )
        for season in self.seasons:
            season.validate()
            if abs(season.fps - 30.0) > 1.0e-6:
                raise ValueError(
                    "Origami seasons must retain the 30 Hz contract"
                )
            if root is not None:
                self._validate_local_season(root, season)

    @staticmethod
    def _validate_local_season(
        root: Path,
        season: DatasetSeason,
    ) -> None:
        season_root = (root / season.relative_root).resolve()
        if root not in season_root.parents and season_root != root:
            raise ValueError("dataset season escapes the dataset root")
        info = season_root / "meta" / "info.json"
        if (
            not info.is_file()
            or _sha256_file(info) != season.info_sha256
        ):
            raise ValueError(
                f"season {season.name!r} info metadata changed"
            )
        stats = season_root / "meta" / "stats.json"
        if season.stats_sha256 is not None and (
            not stats.is_file()
            or _sha256_file(stats) != season.stats_sha256
        ):
            raise ValueError(
                f"season {season.name!r} statistics changed"
            )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._payload(include_fingerprint=False))
        ).hexdigest()

    @property
    def total_episodes(self) -> int:
        return sum(season.episodes for season in self.seasons)

    @property
    def total_frames(self) -> int:
        return sum(season.frames for season in self.seasons)

    def seasons_for(self, split: Split) -> tuple[DatasetSeason, ...]:
        if split not in _SPLITS:
            raise ValueError(f"unknown dataset split {split!r}")
        return tuple(
            season
            for season in self.seasons
            if season.split == split
        )

    def _payload(self, *, include_fingerprint: bool) -> dict[str, Any]:
        def feature(value: DatasetFeature) -> dict[str, Any]:
            payload = asdict(value)
            payload["shape"] = list(value.shape)
            return payload

        def normalization(
            value: FeatureNormalization,
        ) -> dict[str, Any]:
            payload = asdict(value)
            for key in ("mean", "std", "minimum", "maximum"):
                payload[key] = list(payload[key])
            return payload

        result: dict[str, Any] = {
            "schema": self.schema,
            "format_version": self.format_version,
            "source": {
                "repository": self.source_repository,
                "revision": self.source_revision,
            },
            "stream_fingerprint": self.stream_fingerprint,
            "split_policy": {
                "algorithm": SPLIT_ALGORITHM,
                "validation_fraction": self.validation_fraction,
                "test_fraction": self.test_fraction,
            },
            "features": {
                "state": feature(self.state),
                "action": feature(self.action),
                "wrench": feature(self.wrench),
            },
            "normalization_split": "train",
            "normalization": {
                "state": normalization(self.state_normalization),
                "action": normalization(self.action_normalization),
                "wrench": normalization(
                    self.wrench_normalization
                ),
            },
            "seasons": [
                asdict(season) for season in self.seasons
            ],
            "totals": {
                "seasons": len(self.seasons),
                "episodes": self.total_episodes,
                "frames": self.total_frames,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return self._payload(include_fingerprint=True)

    def to_json(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(
            destination.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RobotDatasetManifest":
        try:
            source = payload["source"]
            policy = payload["split_policy"]
            features = payload["features"]
            normalization = payload["normalization"]
            seasons = payload["seasons"]
            totals = payload["totals"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "robot dataset manifest sections are missing"
            ) from error
        if (
            not isinstance(source, Mapping)
            or not isinstance(policy, Mapping)
            or not isinstance(features, Mapping)
            or not isinstance(normalization, Mapping)
            or not isinstance(seasons, list)
            or not isinstance(totals, Mapping)
            or policy.get("algorithm") != SPLIT_ALGORITHM
            or payload.get("normalization_split") != "train"
        ):
            raise ValueError("robot dataset manifest is malformed")

        def feature(name: str) -> DatasetFeature:
            value = features.get(name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} feature is missing")
            return DatasetFeature(
                key=str(value.get("key", "")),
                dtype=str(value.get("dtype", "")),
                shape=tuple(int(item) for item in value.get("shape", ())),
            )

        parsed_features = {
            name: feature(name)
            for name in ("state", "action", "wrench")
        }

        def norm(name: str) -> FeatureNormalization:
            value = normalization.get(name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} normalization is missing")
            dimensions = parsed_features[name].dimensions
            return FeatureNormalization(
                count=int(value.get("count", 0)),
                mean=_float_tuple(
                    value.get("mean", ()),
                    dimensions=dimensions,
                    label=f"{name} mean",
                ),
                std=_float_tuple(
                    value.get("std", ()),
                    dimensions=dimensions,
                    label=f"{name} std",
                ),
                minimum=_float_tuple(
                    value.get("minimum", ()),
                    dimensions=dimensions,
                    label=f"{name} minimum",
                ),
                maximum=_float_tuple(
                    value.get("maximum", ()),
                    dimensions=dimensions,
                    label=f"{name} maximum",
                ),
            )

        parsed_seasons: list[DatasetSeason] = []
        for value in seasons:
            if not isinstance(value, Mapping):
                raise ValueError("robot dataset season is malformed")
            split = str(value.get("split", ""))
            if split not in _SPLITS:
                raise ValueError("robot dataset split is malformed")
            parsed_seasons.append(
                DatasetSeason(
                    name=str(value.get("name", "")),
                    relative_root=str(
                        value.get("relative_root", "")
                    ),
                    split=split,  # type: ignore[arg-type]
                    episodes=int(value.get("episodes", 0)),
                    frames=int(value.get("frames", 0)),
                    fps=float(value.get("fps", 0.0)),
                    info_sha256=str(
                        value.get("info_sha256", "")
                    ),
                    stats_sha256=(
                        None
                        if value.get("stats_sha256") is None
                        else str(value.get("stats_sha256"))
                    ),
                )
            )
        result = cls(
            source_repository=str(source.get("repository", "")),
            source_revision=str(source.get("revision", "")),
            stream_fingerprint=str(
                payload.get("stream_fingerprint", "")
            ),
            validation_fraction=float(
                policy.get("validation_fraction", -1.0)
            ),
            test_fraction=float(policy.get("test_fraction", -1.0)),
            state=parsed_features["state"],
            action=parsed_features["action"],
            wrench=parsed_features["wrench"],
            state_normalization=norm("state"),
            action_normalization=norm("action"),
            wrench_normalization=norm("wrench"),
            seasons=tuple(parsed_seasons),
            schema=str(payload.get("schema", "")),
            format_version=int(payload.get("format_version", 0)),
        )
        result.validate()
        if (
            int(totals.get("seasons", -1)) != len(result.seasons)
            or int(totals.get("episodes", -1))
            != result.total_episodes
            or int(totals.get("frames", -1))
            != result.total_frames
        ):
            raise ValueError("robot dataset totals do not match")
        fingerprint = payload.get("fingerprint")
        if (
            not isinstance(fingerprint, str)
            or _SHA256.fullmatch(fingerprint) is None
            or fingerprint != result.fingerprint
        ):
            raise ValueError("robot dataset fingerprint mismatch")
        return result

    @classmethod
    def from_json(
        cls,
        path: str | os.PathLike[str],
    ) -> "RobotDatasetManifest":
        payload = json.loads(
            Path(path).expanduser().read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("robot dataset manifest must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class _SeasonSource:
    name: str
    root: Path
    info: Mapping[str, Any]
    stats: Mapping[str, Any]
    info_sha256: str
    stats_sha256: str


def _discover_seasons(dataset_root: Path) -> list[Path]:
    direct = dataset_root / "meta" / "info.json"
    if direct.is_file():
        return [dataset_root]
    candidates = sorted(
        path.parent.parent
        for path in dataset_root.glob(
            "season_*/lerobot3.0/meta/info.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no LeRobot 3.0 season metadata found under "
            f"{dataset_root}"
        )
    return candidates


def _load_source(root: Path) -> _SeasonSource:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"season statistics are missing: {stats_path}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict) or not isinstance(stats, dict):
        raise ValueError("LeRobot metadata must contain JSON objects")
    name = (
        root.parent.name
        if root.name == "lerobot3.0"
        else root.name
    )
    return _SeasonSource(
        name=name,
        root=root,
        info=info,
        stats=stats,
        info_sha256=_sha256_file(info_path),
        stats_sha256=_sha256_file(stats_path),
    )


def _feature_from_info(
    source: _SeasonSource,
    *,
    key: str,
    dimensions: int,
) -> DatasetFeature:
    features = source.info.get("features")
    value = (
        features.get(key)
        if isinstance(features, Mapping)
        else None
    )
    if not isinstance(value, Mapping):
        raise ValueError(
            f"season {source.name!r} does not contain {key!r}"
        )
    shape = tuple(int(item) for item in value.get("shape", ()))
    feature = DatasetFeature(
        key=key,
        dtype=str(value.get("dtype", "")),
        shape=shape,
    )
    feature.validate()
    if feature.dimensions != dimensions:
        raise ValueError(
            f"season {source.name!r} {key!r} has "
            f"{feature.dimensions} values; expected {dimensions}"
        )
    return feature


def _split_assignments(
    names: Sequence[str],
    *,
    revision: str,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, Split]:
    count = len(names)
    if count == 0:
        raise ValueError("cannot split an empty dataset")
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"{revision}\\0{name}".encode("utf-8")
        ).digest(),
    )
    test_count = (
        max(1, int(round(count * test_fraction)))
        if test_fraction > 0.0 and count >= 3
        else 0
    )
    validation_count = (
        max(1, int(round(count * validation_fraction)))
        if validation_fraction > 0.0 and count - test_count >= 2
        else 0
    )
    while test_count + validation_count >= count:
        if validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    result: dict[str, Split] = {}
    for index, name in enumerate(ranked):
        if index < test_count:
            result[name] = "test"
        elif index < test_count + validation_count:
            result[name] = "validation"
        else:
            result[name] = "train"
    return result


def _stats_vector(
    source: _SeasonSource,
    *,
    key: str,
    dimensions: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = source.stats.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"season {source.name!r} has no statistics for {key!r}"
        )
    mean = np.asarray(value.get("mean"), dtype=np.float64).reshape(-1)
    std = np.asarray(value.get("std"), dtype=np.float64).reshape(-1)
    minimum = np.asarray(
        value.get("min"),
        dtype=np.float64,
    ).reshape(-1)
    maximum = np.asarray(
        value.get("max"),
        dtype=np.float64,
    ).reshape(-1)
    count_values = np.asarray(
        value.get("count"),
        dtype=np.int64,
    ).reshape(-1)
    if (
        mean.size != dimensions
        or std.size != dimensions
        or minimum.size != dimensions
        or maximum.size != dimensions
        or count_values.size == 0
        or np.any(count_values <= 0)
        or not np.isfinite(
            np.concatenate((mean, std, minimum, maximum))
        ).all()
    ):
        raise ValueError(
            f"season {source.name!r} statistics for {key!r} "
            "do not match its feature"
        )
    if count_values.size == 1:
        count = int(count_values[0])
    elif np.all(count_values == count_values[0]):
        count = int(count_values[0])
    else:
        raise ValueError(
            f"season {source.name!r} has per-axis missing values "
            f"for {key!r}"
        )
    return count, mean, std, minimum, maximum


def _merge_normalization(
    sources: Sequence[_SeasonSource],
    feature: DatasetFeature,
) -> FeatureNormalization:
    count = 0
    weighted_mean = np.zeros(feature.dimensions, dtype=np.float64)
    weighted_second = np.zeros(
        feature.dimensions,
        dtype=np.float64,
    )
    minimum = np.full(feature.dimensions, np.inf)
    maximum = np.full(feature.dimensions, -np.inf)
    for source in sources:
        (
            source_count,
            source_mean,
            source_std,
            source_minimum,
            source_maximum,
        ) = _stats_vector(
            source,
            key=feature.key,
            dimensions=feature.dimensions,
        )
        count += source_count
        weighted_mean += source_count * source_mean
        weighted_second += source_count * (
            np.square(source_std) + np.square(source_mean)
        )
        minimum = np.minimum(minimum, source_minimum)
        maximum = np.maximum(maximum, source_maximum)
    mean = weighted_mean / count
    variance = np.maximum(
        weighted_second / count - np.square(mean),
        0.0,
    )
    result = FeatureNormalization(
        count=count,
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in np.sqrt(variance)),
        minimum=tuple(float(value) for value in minimum),
        maximum=tuple(float(value) for value in maximum),
    )
    result.validate(feature.dimensions)
    return result


def build_origami_manifest(
    dataset_root: str | os.PathLike[str],
    *,
    stream_contract: TactileStreamContract,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> RobotDatasetManifest:
    """Inspect pinned local metadata and build a season-level manifest."""

    stream_contract.validate()
    if (
        not 0.0 <= validation_fraction < 1.0
        or not 0.0 <= test_fraction < 1.0
        or validation_fraction + test_fraction >= 1.0
    ):
        raise ValueError("dataset split fractions are invalid")
    root = Path(dataset_root).expanduser().resolve()
    sources = [
        _load_source(path)
        for path in _discover_seasons(root)
    ]
    required = {
        "state": (
            stream_contract.state.key,
            len(stream_contract.state.names),
        ),
        "action": (
            stream_contract.action.key,
            len(stream_contract.action.names),
        ),
        "wrench": (
            stream_contract.wrench_key,
            len(stream_contract.sensors)
            * len(stream_contract.wrench_axes),
        ),
    }
    first = sources[0]
    features = {
        name: _feature_from_info(
            first,
            key=key,
            dimensions=dimensions,
        )
        for name, (key, dimensions) in required.items()
    }
    for source in sources[1:]:
        for name, (key, dimensions) in required.items():
            if _feature_from_info(
                source,
                key=key,
                dimensions=dimensions,
            ) != features[name]:
                raise ValueError(
                    f"season {source.name!r} changes the {name} schema"
                )
    assignments = _split_assignments(
        [source.name for source in sources],
        revision=stream_contract.source_revision,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    training_sources = [
        source
        for source in sources
        if assignments[source.name] == "train"
    ]
    if not training_sources:
        raise ValueError("season split does not contain training data")
    seasons: list[DatasetSeason] = []
    for source in sources:
        if str(source.info.get("codebase_version", "")) != "v3.0":
            raise ValueError(
                f"season {source.name!r} is not LeRobot v3.0"
            )
        fps = float(source.info.get("fps", 0.0))
        if abs(fps - stream_contract.rate_hz) > 1.0e-6:
            raise ValueError(
                f"season {source.name!r} rate does not match "
                "the stream contract"
            )
        relative_root = source.root.relative_to(root).as_posix()
        seasons.append(
            DatasetSeason(
                name=source.name,
                relative_root=relative_root,
                split=assignments[source.name],
                episodes=int(source.info.get("total_episodes", 0)),
                frames=int(source.info.get("total_frames", 0)),
                fps=fps,
                info_sha256=source.info_sha256,
                stats_sha256=source.stats_sha256,
            )
        )
    result = RobotDatasetManifest(
        source_repository=stream_contract.source_repository,
        source_revision=stream_contract.source_revision,
        stream_fingerprint=stream_contract.fingerprint,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        state=features["state"],
        action=features["action"],
        wrench=features["wrench"],
        state_normalization=_merge_normalization(
            training_sources,
            features["state"],
        ),
        action_normalization=_merge_normalization(
            training_sources,
            features["action"],
        ),
        wrench_normalization=_merge_normalization(
            training_sources,
            features["wrench"],
        ),
        seasons=tuple(sorted(seasons, key=lambda value: value.name)),
    )
    result.validate(dataset_root=root)
    return result


def prepare_origami_metadata(
    destination: str | os.PathLike[str],
    *,
    revision: str,
    repository: str = "SharpaIT/Robotic_Origami_Challenge",
    token: str | None = None,
) -> Path:
    """Download only pinned LeRobot 3.0 metadata from the gated dataset."""

    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("dataset revision must be a full Git commit")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "metadata preparation requires huggingface_hub"
        ) from error
    destination_path = Path(destination).expanduser().resolve()
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=str(destination_path),
        allow_patterns=[
            "README.md",
            "season_*/lerobot3.0/meta/*",
        ],
    )
    return destination_path


def download_origami_seasons(
    destination: str | os.PathLike[str],
    *,
    revision: str,
    seasons: Sequence[str],
    video_keys: Sequence[str] = (),
    repository: str = "SharpaIT/Robotic_Origami_Challenge",
    token: str | None = None,
) -> Path:
    """Fetch pinned numeric data and explicitly selected video streams."""

    if _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("dataset revision must be a full Git commit")
    if not seasons:
        raise ValueError("at least one collection season is required")
    if any(
        not re.fullmatch(r"season_[A-Za-z0-9_.-]+", season)
        for season in seasons
    ):
        raise ValueError("collection season name is invalid")
    if any(
        not re.fullmatch(r"observation\\.images\\.[A-Za-z0-9_.-]+", key)
        for key in video_keys
    ):
        raise ValueError("LeRobot video key is invalid")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "dataset download requires huggingface_hub"
        ) from error
    patterns = ["README.md"]
    for season in seasons:
        prefix = f"{season}/lerobot3.0"
        patterns.extend(
            (
                f"{prefix}/meta/*",
                f"{prefix}/data/**/*.parquet",
            )
        )
        patterns.extend(
            f"{prefix}/videos/{key}/**/*.mp4"
            for key in video_keys
        )
    destination_path = Path(destination).expanduser().resolve()
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        token=token,
        local_dir=str(destination_path),
        allow_patterns=patterns,
    )
    return destination_path


@dataclass(frozen=True, slots=True)
class EpisodeArrays:
    season: str
    episode_index: int
    fps: float
    state: npt.NDArray[np.float32]
    action: npt.NDArray[np.float32]
    wrench: npt.NDArray[np.float32]
    timestamp: npt.NDArray[np.float32]
    frame_index: npt.NDArray[np.int64]

    @property
    def length(self) -> int:
        return int(self.state.shape[0])

    def validate(
        self,
        *,
        state_dimensions: int,
        action_dimensions: int,
        wrench_dimensions: int,
    ) -> None:
        length = self.length
        if (
            length <= 0
            or self.state.shape != (length, state_dimensions)
            or self.action.shape != (length, action_dimensions)
            or self.wrench.shape != (length, wrench_dimensions)
            or self.timestamp.shape != (length,)
            or self.frame_index.shape != (length,)
            or not np.isfinite(self.state).all()
            or not np.isfinite(self.action).all()
            or not np.isfinite(self.wrench).all()
            or not np.isfinite(self.timestamp).all()
            or np.any(np.diff(self.frame_index) != 1)
            or np.any(np.diff(self.timestamp) < -1.0e-6)
        ):
            raise ValueError("LeRobot episode arrays are not aligned")


def _arrow_column(
    table: Any,
    key: str,
    dimensions: int,
    dtype: npt.DTypeLike,
) -> np.ndarray:
    if key not in table.column_names:
        raise ValueError(f"Parquet data does not contain {key!r}")
    column = table[key].combine_chunks()
    values_type = getattr(column.type, "value_type", None)
    list_size = getattr(column.type, "list_size", None)
    if values_type is not None and list_size is not None:
        array = np.asarray(column.values.to_numpy(zero_copy_only=False))
        array = array.reshape((len(column), int(list_size)))
    else:
        array = np.asarray(column.to_pylist())
        if dimensions > 1:
            array = array.reshape((len(column), dimensions))
    array = np.asarray(array, dtype=dtype)
    if dimensions == 1:
        array = array.reshape(-1)
    elif array.shape != (len(column), dimensions):
        raise ValueError(
            f"Parquet column {key!r} has shape {array.shape}; "
            f"expected {(len(column), dimensions)}"
        )
    return np.ascontiguousarray(array)


class LeRobotNumericReader:
    """Read one season at a time and preserve episode boundaries."""

    def __init__(
        self,
        dataset_root: str | os.PathLike[str],
        manifest: RobotDatasetManifest,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        manifest.validate(dataset_root=self.dataset_root)
        self.manifest = manifest

    def episodes(
        self,
        season: DatasetSeason,
    ) -> tuple[EpisodeArrays, ...]:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "numeric LeRobot ingestion requires pyarrow"
            ) from error
        season_root = self.dataset_root / season.relative_root
        paths = sorted((season_root / "data").rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"no Parquet data found for season {season.name!r}"
            )
        columns = [
            self.manifest.state.key,
            self.manifest.action.key,
            self.manifest.wrench.key,
            "episode_index",
            "frame_index",
            "timestamp",
        ]
        tables = [
            pq.read_table(path, columns=columns)
            for path in paths
        ]
        table = pa.concat_tables(tables, promote_options="default")
        state = _arrow_column(
            table,
            self.manifest.state.key,
            self.manifest.state.dimensions,
            np.float32,
        )
        action = _arrow_column(
            table,
            self.manifest.action.key,
            self.manifest.action.dimensions,
            np.float32,
        )
        wrench = _arrow_column(
            table,
            self.manifest.wrench.key,
            self.manifest.wrench.dimensions,
            np.float32,
        )
        episode_index = _arrow_column(
            table,
            "episode_index",
            1,
            np.int64,
        )
        frame_index = _arrow_column(
            table,
            "frame_index",
            1,
            np.int64,
        )
        timestamp = _arrow_column(
            table,
            "timestamp",
            1,
            np.float32,
        )
        order = np.lexsort((frame_index, episode_index))
        arrays = (
            state[order],
            action[order],
            wrench[order],
            episode_index[order],
            frame_index[order],
            timestamp[order],
        )
        boundaries = np.flatnonzero(
            np.diff(arrays[3], prepend=arrays[3][0] - 1)
        )
        ends = np.append(boundaries[1:], len(arrays[3]))
        episodes: list[EpisodeArrays] = []
        for start, end in zip(boundaries, ends, strict=True):
            episode = EpisodeArrays(
                season=season.name,
                episode_index=int(arrays[3][start]),
                fps=season.fps,
                state=np.ascontiguousarray(arrays[0][start:end]),
                action=np.ascontiguousarray(arrays[1][start:end]),
                wrench=np.ascontiguousarray(arrays[2][start:end]),
                timestamp=np.ascontiguousarray(arrays[5][start:end]),
                frame_index=np.ascontiguousarray(
                    arrays[4][start:end]
                ),
            )
            episode.validate(
                state_dimensions=self.manifest.state.dimensions,
                action_dimensions=self.manifest.action.dimensions,
                wrench_dimensions=self.manifest.wrench.dimensions,
            )
            episodes.append(episode)
        if (
            len(episodes) != season.episodes
            or sum(episode.length for episode in episodes)
            != season.frames
        ):
            raise ValueError(
                f"season {season.name!r} data totals do not match "
                "its pinned metadata"
            )
        return tuple(episodes)


@dataclass(frozen=True, slots=True)
class OrigamiBatch:
    state_history: npt.NDArray[np.float32]
    wrench_history: npt.NDArray[np.float32]
    action_chunk: npt.NDArray[np.float32]
    action_mask: npt.NDArray[np.bool_]
    stream_state_history: npt.NDArray[np.float32]
    stream_wrench_history: npt.NDArray[np.float32]
    stream_action_chunk: npt.NDArray[np.float32]
    stream_action_derivative: npt.NDArray[np.float32]
    stream_action_mask: npt.NDArray[np.bool_]
    stream_time: npt.NDArray[np.float32]
    next_wrench: npt.NDArray[np.float32]
    references: tuple["WindowReference", ...]


@dataclass(frozen=True, slots=True)
class WindowReference:
    season: str
    episode_index: int
    frame_index: int
    stream_frame_index: int


def _history(
    value: np.ndarray,
    end: int,
    length: int,
) -> np.ndarray:
    start = max(0, end - length + 1)
    window = value[start : end + 1]
    if len(window) < length:
        window = np.concatenate(
            (
                np.repeat(
                    window[:1],
                    length - len(window),
                    axis=0,
                ),
                window,
            ),
            axis=0,
        )
    return window


def _padded_chunk(
    value: np.ndarray,
    start: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    end = min(len(value), start + length)
    window = value[start:end]
    mask = np.zeros(length, dtype=np.bool_)
    mask[: len(window)] = True
    if len(window) < length:
        window = np.concatenate(
            (
                window,
                np.repeat(
                    window[-1:],
                    length - len(window),
                    axis=0,
                ),
            ),
            axis=0,
        )
    return window, mask


class OrigamiWindowSampler:
    """Bounded-memory random windows with whole-season split isolation."""

    def __init__(
        self,
        reader: LeRobotNumericReader,
        *,
        split: Split,
        history: int,
        horizon: int,
        seed: int,
    ) -> None:
        if history <= 0 or horizon <= 1:
            raise ValueError("history and horizon must be positive")
        self.reader = reader
        self.split = split
        self.history = history
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.seasons = reader.manifest.seasons_for(split)
        if not self.seasons:
            raise ValueError(
                f"dataset split {split!r} contains no seasons"
            )
        self._loaded_season: DatasetSeason | None = None
        self._episodes: tuple[EpisodeArrays, ...] = ()

    def _load_random_season(self) -> None:
        season = self.seasons[
            int(self.rng.integers(0, len(self.seasons)))
        ]
        if self._loaded_season != season:
            self._episodes = self.reader.episodes(season)
            self._loaded_season = season

    def sample(self, batch_size: int) -> OrigamiBatch:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        values: dict[str, list[np.ndarray]] = {
            "state_history": [],
            "wrench_history": [],
            "action_chunk": [],
            "action_mask": [],
            "stream_state_history": [],
            "stream_wrench_history": [],
            "stream_action_chunk": [],
            "stream_action_derivative": [],
            "stream_action_mask": [],
            "stream_time": [],
            "next_wrench": [],
        }
        references: list[WindowReference] = []
        while len(values["action_chunk"]) < batch_size:
            self._load_random_season()
            episode = self._episodes[
                int(self.rng.integers(0, len(self._episodes)))
            ]
            if episode.length < 2:
                continue
            start = int(self.rng.integers(0, episode.length - 1))
            maximum_offset = min(
                self.horizon - 1,
                episode.length - start - 1,
            )
            stream_offset = int(
                self.rng.integers(0, maximum_offset + 1)
            )
            stream_start = start + stream_offset
            action_chunk, action_mask = _padded_chunk(
                episode.action,
                start,
                self.horizon,
            )
            stream_chunk, stream_mask = _padded_chunk(
                episode.action,
                stream_start,
                self.horizon,
            )
            derivative_source = np.concatenate(
                (episode.action[1:], episode.action[-1:]),
                axis=0,
            )
            derivative = episode.fps * (
                derivative_source - episode.action
            )
            derivative_chunk, _ = _padded_chunk(
                derivative,
                stream_start,
                self.horizon,
            )
            next_wrench_index = min(
                stream_start + 1,
                episode.length - 1,
            )
            values["state_history"].append(
                _history(episode.state, start, self.history)
            )
            values["wrench_history"].append(
                _history(episode.wrench, start, self.history)
            )
            values["action_chunk"].append(action_chunk)
            values["action_mask"].append(action_mask)
            values["stream_state_history"].append(
                _history(
                    episode.state,
                    stream_start,
                    self.history,
                )
            )
            values["stream_wrench_history"].append(
                _history(
                    episode.wrench,
                    stream_start,
                    self.history,
                )
            )
            values["stream_action_chunk"].append(stream_chunk)
            values["stream_action_derivative"].append(
                derivative_chunk
            )
            values["stream_action_mask"].append(stream_mask)
            values["stream_time"].append(
                np.asarray(
                    stream_offset / episode.fps,
                    dtype=np.float32,
                )
            )
            values["next_wrench"].append(
                episode.wrench[next_wrench_index]
            )
            references.append(
                WindowReference(
                    season=episode.season,
                    episode_index=episode.episode_index,
                    frame_index=int(episode.frame_index[start]),
                    stream_frame_index=int(
                        episode.frame_index[stream_start]
                    ),
                )
            )
        return OrigamiBatch(
            **{
                key: np.ascontiguousarray(np.stack(items))
                for key, items in values.items()
            },
            references=tuple(references),
        )


@dataclass(frozen=True, slots=True)
class EpisodeVideoSegment:
    path: Path
    from_timestamp: float
    to_timestamp: float


class LeRobotVideoReader:
    """Resolve v3 episode video metadata and decode aligned histories."""

    def __init__(
        self,
        dataset_root: str | os.PathLike[str],
        manifest: RobotDatasetManifest,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        manifest.validate(dataset_root=self.dataset_root)
        self.manifest = manifest
        self._season_by_name = {
            season.name: season for season in manifest.seasons
        }
        self._metadata: dict[
            str,
            tuple[Mapping[str, Any], dict[int, Mapping[str, Any]]],
        ] = {}

    def _season_metadata(
        self,
        season_name: str,
    ) -> tuple[Mapping[str, Any], dict[int, Mapping[str, Any]]]:
        cached = self._metadata.get(season_name)
        if cached is not None:
            return cached
        season = self._season_by_name.get(season_name)
        if season is None:
            raise ValueError(f"unknown collection season {season_name!r}")
        season_root = self.dataset_root / season.relative_root
        info = json.loads(
            (season_root / "meta" / "info.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(info, dict):
            raise ValueError("LeRobot info metadata is malformed")
        episode_root = season_root / "meta" / "episodes"
        rows: list[Mapping[str, Any]] = []
        parquet_paths = (
            sorted(episode_root.rglob("*.parquet"))
            if episode_root.is_dir()
            else []
        )
        if parquet_paths:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as error:
                raise RuntimeError(
                    "video metadata ingestion requires pyarrow"
                ) from error
            tables = [pq.read_table(path) for path in parquet_paths]
            rows = pa.concat_tables(
                tables,
                promote_options="default",
            ).to_pylist()
        else:
            jsonl = season_root / "meta" / "episodes.jsonl"
            if not jsonl.is_file():
                raise FileNotFoundError(
                    f"episode video metadata is missing for "
                    f"{season_name!r}"
                )
            for line in jsonl.read_text(
                encoding="utf-8"
            ).splitlines():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        "episode metadata row is malformed"
                    )
                rows.append(value)
        by_episode: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            episode_index = int(row.get("episode_index", -1))
            if episode_index < 0 or episode_index in by_episode:
                raise ValueError(
                    "episode video metadata indices are invalid"
                )
            by_episode[episode_index] = row
        result = (info, by_episode)
        self._metadata[season_name] = result
        return result

    @staticmethod
    def _video_value(
        row: Mapping[str, Any],
        key: str,
        field: str,
    ) -> Any:
        flattened = row.get(f"videos/{key}/{field}")
        if flattened is not None:
            return flattened
        videos = row.get("videos")
        if isinstance(videos, Mapping):
            video = videos.get(key)
            if isinstance(video, Mapping):
                return video.get(field)
        return None

    def segment(
        self,
        reference: WindowReference,
        key: str,
    ) -> EpisodeVideoSegment:
        info, episodes = self._season_metadata(reference.season)
        features = info.get("features")
        feature = (
            features.get(key)
            if isinstance(features, Mapping)
            else None
        )
        if (
            not isinstance(feature, Mapping)
            or feature.get("dtype") != "video"
        ):
            raise ValueError(
                f"LeRobot season does not contain video {key!r}"
            )
        row = episodes.get(reference.episode_index)
        if row is None:
            raise ValueError(
                "episode has no corresponding video metadata"
            )
        chunk_index = self._video_value(
            row,
            key,
            "chunk_index",
        )
        file_index = self._video_value(
            row,
            key,
            "file_index",
        )
        from_timestamp = self._video_value(
            row,
            key,
            "from_timestamp",
        )
        to_timestamp = self._video_value(
            row,
            key,
            "to_timestamp",
        )
        if any(
            value is None
            for value in (
                chunk_index,
                file_index,
                from_timestamp,
                to_timestamp,
            )
        ):
            raise ValueError(
                f"episode video metadata for {key!r} is incomplete"
            )
        template = str(
            info.get(
                "video_path",
                (
                    "videos/{video_key}/chunk-{chunk_index:03d}/"
                    "file-{file_index:03d}.mp4"
                ),
            )
        )
        relative = template.format(
            video_key=key,
            chunk_index=int(chunk_index),
            file_index=int(file_index),
            episode_chunk=int(chunk_index),
            episode_file=int(file_index),
        )
        season = self._season_by_name[reference.season]
        path = (
            self.dataset_root / season.relative_root / relative
        ).resolve()
        season_root = (
            self.dataset_root / season.relative_root
        ).resolve()
        if season_root not in path.parents or not path.is_file():
            raise FileNotFoundError(
                f"pinned episode video is missing: {path}"
            )
        start = float(from_timestamp)
        end = float(to_timestamp)
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0.0
            or end <= start
        ):
            raise ValueError("episode video time range is invalid")
        return EpisodeVideoSegment(path, start, end)

    @staticmethod
    def _decode_targets(
        path: Path,
        targets: Sequence[float],
        *,
        fps: float,
        width: int,
        height: int,
    ) -> list[np.ndarray]:
        try:
            import av
        except ImportError as error:
            raise RuntimeError(
                "video ingestion requires PyAV"
            ) from error
        order = np.argsort(np.asarray(targets))
        sorted_targets = [targets[int(index)] for index in order]
        decoded: list[np.ndarray | None] = [
            None for _ in targets
        ]
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            seek_time = max(sorted_targets[0] - 1.0, 0.0)
            container.seek(
                int(seek_time / time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            target_cursor = 0
            previous_time: float | None = None
            previous_image: np.ndarray | None = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_time = float(frame.pts * stream.time_base)
                image: np.ndarray | None = None
                while (
                    target_cursor < len(sorted_targets)
                    and sorted_targets[target_cursor] <= frame_time
                ):
                    target = sorted_targets[target_cursor]
                    use_previous = (
                        previous_time is not None
                        and previous_image is not None
                        and abs(previous_time - target)
                        <= abs(frame_time - target)
                    )
                    if use_previous:
                        selected_time = previous_time
                        selected = previous_image
                    else:
                        selected_time = frame_time
                        if image is None:
                            image = frame.reformat(
                                width=width,
                                height=height,
                                format="rgb24",
                            ).to_ndarray()
                        selected = image
                    if abs(selected_time - target) > 1.5 / fps:
                        raise ValueError(
                            "video timestamp is not frame-aligned"
                        )
                    original_index = int(order[target_cursor])
                    decoded[original_index] = np.ascontiguousarray(
                        selected,
                        dtype=np.uint8,
                    )
                    target_cursor += 1
                if target_cursor >= len(sorted_targets):
                    break
                previous_time = frame_time
                previous_image = frame.reformat(
                    width=width,
                    height=height,
                    format="rgb24",
                ).to_ndarray()
        if any(value is None for value in decoded):
            raise ValueError(
                "video ended before all aligned frames were decoded"
            )
        return [
            value
            for value in decoded
            if value is not None
        ]

    def batch_histories(
        self,
        references: Sequence[WindowReference],
        *,
        keys: Sequence[str],
        history: int,
        width: int,
        height: int,
        streaming: bool = False,
    ) -> npt.NDArray[np.uint8]:
        if (
            not references
            or not keys
            or history <= 0
            or width <= 0
            or height <= 0
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("video history request is invalid")
        output = np.empty(
            (
                len(references),
                history,
                len(keys),
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )
        requests: dict[
            Path,
            list[tuple[float, int, int, int, float]],
        ] = {}
        for batch_index, reference in enumerate(references):
            season = self._season_by_name[reference.season]
            current = (
                reference.stream_frame_index
                if streaming
                else reference.frame_index
            )
            for view_index, key in enumerate(keys):
                segment = self.segment(reference, key)
                for history_index in range(history):
                    frame_index = max(
                        0,
                        current - history + history_index + 1,
                    )
                    timestamp = (
                        segment.from_timestamp
                        + frame_index / season.fps
                    )
                    if timestamp > (
                        segment.to_timestamp + 1.0 / season.fps
                    ):
                        raise ValueError(
                            "requested frame exceeds episode video range"
                        )
                    requests.setdefault(segment.path, []).append(
                        (
                            timestamp,
                            batch_index,
                            history_index,
                            view_index,
                            season.fps,
                        )
                    )
        for path, values in requests.items():
            frame_rates = {value[4] for value in values}
            if len(frame_rates) != 1:
                raise ValueError(
                    "one video file cannot change frame rate"
                )
            images = self._decode_targets(
                path,
                [value[0] for value in values],
                fps=next(iter(frame_rates)),
                width=width,
                height=height,
            )
            for value, image in zip(values, images, strict=True):
                _, batch_index, history_index, view_index, _ = value
                output[
                    batch_index,
                    history_index,
                    view_index,
                ] = image
        return np.ascontiguousarray(output)


def decode_video_frames(
    path: str | os.PathLike[str],
    frame_indices: Sequence[int],
) -> npt.NDArray[np.uint8]:
    """Decode exact presentation-order frames with PyAV."""

    if not frame_indices:
        raise ValueError("at least one video frame is required")
    if any(index < 0 for index in frame_indices):
        raise ValueError("video frame indices must be non-negative")
    try:
        import av
    except ImportError as error:
        raise RuntimeError("video ingestion requires PyAV") from error
    requested = list(frame_indices)
    selected: dict[int, np.ndarray] = {}
    maximum = max(requested)
    with av.open(str(Path(path).expanduser())) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                selected[index] = frame.to_ndarray(format="rgb24")
            if index >= maximum:
                break
    if any(index not in selected for index in requested):
        raise ValueError("video ended before a requested frame")
    shapes = {selected[index].shape for index in requested}
    if len(shapes) != 1:
        raise ValueError("decoded video changes frame dimensions")
    return np.ascontiguousarray(
        np.stack([selected[index] for index in requested]),
        dtype=np.uint8,
    )


__all__ = [
    "DatasetFeature",
    "DatasetSeason",
    "EpisodeArrays",
    "EpisodeVideoSegment",
    "FeatureNormalization",
    "LeRobotNumericReader",
    "LeRobotVideoReader",
    "ORIGAMI_PINNED_REVISION",
    "OrigamiBatch",
    "OrigamiWindowSampler",
    "ROBOT_DATASET_MANIFEST_FORMAT",
    "ROBOT_DATASET_MANIFEST_SCHEMA",
    "RobotDatasetManifest",
    "SPLIT_ALGORITHM",
    "WindowReference",
    "build_origami_manifest",
    "decode_video_frames",
    "download_origami_seasons",
    "prepare_origami_metadata",
]
