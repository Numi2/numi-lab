"""MLX latent world model, CEM planner, and native Crow policy distillation.

The module consumes canonical ``numi.crow-replay.v1`` accepted-state replays
from the v10 navigation TaskPack. Planning never replaces Metal physics:
planned actions become demonstrations, the student becomes a PolicyPack, and
promotion still requires autonomous native held-out rollouts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ModuleNotFoundError:  # Data-contract tests do not require MLX.
    mx = None
    nn = None
    optim = None


REPLAY_SCHEMA = "numi.crow-replay.v1"
MODEL_SCHEMA = "numi.crow-latent-world-model.v1"
DEMONSTRATION_SCHEMA = "numi.crow-planner-demonstrations.v1"
STUDENT_SCHEMA = "numi.crow-world-model-student.v1"
V10_TASK = "birdflow_american_crow_navigation_v10_world_model"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_envelope(path: Path, schema: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema": schema,
        "payload_sha256": _sha256(payload),
        "payload": payload,
    }
    path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_envelope(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path} is not {schema}")
    payload = value.get("payload")
    if not isinstance(payload, dict) or value.get("payload_sha256") != _sha256(payload):
        raise ValueError(f"{path} payload hash is invalid")
    return payload


def _finite_matrix(value: Any, width: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [samples, {width}]")
    return result


@dataclass(frozen=True, slots=True)
class CrowTransitionDataset:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    fingerprints: dict[str, str]
    courses: tuple[str, ...]
    scheduled_resets: tuple[bool, ...]
    maximum_root_height: float
    airborne_frame_fraction: float

    @property
    def observation_count(self) -> int:
        return int(self.observations.shape[1])

    @property
    def action_count(self) -> int:
        return int(self.actions.shape[1])


def load_replays(paths: Sequence[Path]) -> CrowTransitionDataset:
    if not paths:
        raise ValueError("at least one CrowReplayPack is required")
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[float] = []
    identity: dict[str, str] | None = None
    courses: list[str] = []
    scheduled_resets: list[bool] = []
    root_heights: list[float] = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("schema") != REPLAY_SCHEMA:
            raise ValueError(f"{path} is not a CrowReplayPack")
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("task") != V10_TASK:
            raise ValueError(f"{path} is not a v10 navigation replay")
        observation_count = int(payload.get("actor_observation_count", 0))
        action_count = int(payload.get("action_count", 0))
        if observation_count <= 0 or action_count <= 0:
            raise ValueError(f"{path} omits actor/action dimensions")
        current_identity = {
            key: str(payload.get(key, ""))
            for key in (
                "world_fingerprint",
                "task_fingerprint",
                "observation_fingerprint",
                "action_fingerprint",
            )
        }
        if not all(current_identity.values()):
            raise ValueError(f"{path} omits deployment fingerprints")
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise ValueError("Crow replays disagree on deployment fingerprints")
        courses.append(str(payload.get("navigation_course", "unknown")))
        if not isinstance(payload.get("scheduled_resets"), bool):
            raise ValueError(f"{path} omits scheduled-reset provenance")
        scheduled_resets.append(bool(payload["scheduled_resets"]))
        frames = payload.get("frames")
        if not isinstance(frames, list) or len(frames) < 2:
            raise ValueError(f"{path} has fewer than two frames")
        for current, following in zip(frames[:-1], frames[1:], strict=True):
            if not isinstance(current, dict) or not isinstance(following, dict):
                raise ValueError(f"{path} contains a malformed frame")
            # A completed frame is followed by an explicit device reset. That
            # discontinuity is not dynamics and must never become a learned
            # transition target.
            if bool(current.get("done", False)) or bool(current.get("timeout", False)):
                continue
            observation = np.asarray(current.get("actor_observation"), dtype=np.float32)
            action = np.asarray(current.get("accepted_actions"), dtype=np.float32)
            next_observation = np.asarray(
                following.get("actor_observation"), dtype=np.float32
            )
            if (
                observation.shape != (observation_count,)
                or next_observation.shape != (observation_count,)
                or action.shape != (action_count,)
                or not np.isfinite(observation).all()
                or not np.isfinite(next_observation).all()
                or not np.isfinite(action).all()
            ):
                raise ValueError(f"{path} frame dimensions are invalid")
            observations.append(observation)
            actions.append(action)
            next_observations.append(next_observation)
            rewards.append(float(following.get("reward", 0.0)))
            dones.append(float(bool(following.get("done", False))))
        for frame in frames:
            height = float(frame.get("root_height", math.nan))
            if not math.isfinite(height):
                raise ValueError(f"{path} frame omits finite root height")
            root_heights.append(height)
    assert identity is not None
    if not observations:
        raise ValueError("Crow replays contain no contiguous transitions")
    return CrowTransitionDataset(
        observations=np.stack(observations),
        actions=np.stack(actions),
        next_observations=np.stack(next_observations),
        rewards=np.asarray(rewards, dtype=np.float32)[:, None],
        dones=np.asarray(dones, dtype=np.float32)[:, None],
        fingerprints=identity,
        courses=tuple(courses),
        scheduled_resets=tuple(scheduled_resets),
        maximum_root_height=max(root_heights),
        airborne_frame_fraction=sum(height >= 0.35 for height in root_heights)
        / len(root_heights),
    )


def require_navigation_flight_data(
    dataset: CrowTransitionDataset,
    *,
    minimum_maximum_root_height: float,
    minimum_airborne_fraction: float,
) -> None:
    """Reject reset-dominated or ground-only data before expensive training."""
    if any(dataset.scheduled_resets):
        raise ValueError(
            "navigation replays used scheduled resets; recollect with "
            "`numi crow navigation collect` or --no-scheduled-resets"
        )
    if dataset.maximum_root_height < minimum_maximum_root_height:
        raise ValueError(
            "navigation dataset is ground-only: maximum root height "
            f"{dataset.maximum_root_height:.3f} m is below "
            f"{minimum_maximum_root_height:.3f} m"
        )
    if dataset.airborne_frame_fraction < minimum_airborne_fraction:
        raise ValueError(
            "navigation dataset has insufficient flight coverage: airborne "
            f"fraction {dataset.airborne_frame_fraction:.4f} is below "
            f"{minimum_airborne_fraction:.4f}"
        )


def _require_mlx() -> None:
    if mx is None or nn is None or optim is None:
        raise RuntimeError("MLX is required; run with the Numi Lab MLX Python")


if nn is not None:
    class CrowLatentWorldModel(nn.Module):
        def __init__(
            self, observation_count: int, action_count: int,
            hidden_count: int, latent_count: int,
        ) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(observation_count, hidden_count), nn.SiLU(),
                nn.Linear(hidden_count, latent_count),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_count, hidden_count), nn.SiLU(),
                nn.Linear(hidden_count, observation_count),
            )
            self.transition = nn.Sequential(
                nn.Linear(latent_count + action_count, hidden_count), nn.SiLU(),
                nn.Linear(hidden_count, latent_count),
            )
            self.reward_head = nn.Sequential(
                nn.Linear(latent_count + action_count, hidden_count), nn.SiLU(),
                nn.Linear(hidden_count, 1),
            )
            self.done_head = nn.Linear(latent_count, 1)

        def predict(self, observation: Any, action: Any) -> tuple[Any, Any, Any, Any]:
            latent = self.encoder(observation)
            joined = mx.concatenate((latent, action), axis=-1)
            next_latent = latent + self.transition(joined)
            return (
                self.decoder(latent),
                self.decoder(next_latent),
                self.reward_head(mx.concatenate((next_latent, action), axis=-1)),
                self.done_head(next_latent),
            )


    class CrowStudent(nn.Module):
        def __init__(self, widths: Sequence[int], activations: Sequence[int]) -> None:
            super().__init__()
            if len(widths) < 2 or len(activations) != len(widths) - 1:
                raise ValueError("student topology is invalid")
            self.layers = [
                nn.Linear(input_count, output_count)
                for input_count, output_count in zip(widths[:-1], widths[1:])
            ]
            self.activations = tuple(int(value) for value in activations)

        def __call__(self, observation: Any) -> Any:
            value = observation
            for layer, activation in zip(self.layers, self.activations, strict=True):
                value = layer(value)
                if activation == 1:
                    value = mx.maximum(value, 0.0)
                elif activation == 2:
                    value = mx.tanh(value)
                elif activation == 3:
                    value = nn.elu(value)
                elif activation == 4:
                    value = nn.silu(value)
                elif activation != 0:
                    raise ValueError(f"unsupported student activation {activation}")
            return value


def _normalization(dataset: CrowTransitionDataset) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate((dataset.observations, dataset.next_observations))
    mean = joined.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = joined.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1.0e-3))


def train_world_model(arguments: argparse.Namespace) -> dict[str, Any]:
    _require_mlx()
    dataset = load_replays([Path(value) for value in arguments.replay])
    require_navigation_flight_data(
        dataset,
        minimum_maximum_root_height=arguments.minimum_maximum_root_height,
        minimum_airborne_fraction=arguments.minimum_airborne_fraction,
    )
    if len(dataset.observations) < 2:
        raise ValueError("world-model training needs at least two transitions")
    mean, std = _normalization(dataset)
    observation = (dataset.observations - mean) / std
    next_observation = (dataset.next_observations - mean) / std
    generator = np.random.default_rng(arguments.seed)
    order = generator.permutation(len(observation))
    validation_count = max(1, int(round(arguments.validation_fraction * len(order))))
    validation = order[:validation_count]
    training = order[validation_count:]
    if not len(training):
        training = order
    model = CrowLatentWorldModel(
        dataset.observation_count, dataset.action_count,
        arguments.hidden, arguments.latent,
    )
    optimizer = optim.Adam(learning_rate=arguments.learning_rate)
    optimizer.init(model.trainable_parameters())

    def loss_function(obs: Any, action: Any, nxt: Any, reward: Any, done: Any):
        reconstruction, predicted_next, predicted_reward, done_logit = model.predict(
            obs, action
        )
        latent_target = mx.stop_gradient(model.encoder(nxt))
        latent = model.encoder(obs)
        predicted_latent = latent + model.transition(
            mx.concatenate((latent, action), axis=-1)
        )
        reconstruction_loss = mx.mean(mx.square(reconstruction - obs))
        next_loss = mx.mean(mx.square(predicted_next - nxt))
        latent_loss = mx.mean(mx.square(predicted_latent - latent_target))
        reward_loss = mx.mean(mx.square(predicted_reward - reward))
        done_loss = mx.mean(
            mx.maximum(done_logit, 0.0) - done * done_logit
            + mx.log1p(mx.exp(-mx.abs(done_logit)))
        )
        total = reconstruction_loss + next_loss + latent_loss + reward_loss + 0.1 * done_loss
        return total, {
            "loss": total, "reconstruction_mse": reconstruction_loss,
            "next_observation_mse": next_loss, "latent_mse": latent_loss,
            "reward_mse": reward_loss, "done_bce": done_loss,
        }

    loss_and_gradient = nn.value_and_grad(model, loss_function)
    history: list[dict[str, float]] = []
    for epoch in range(arguments.epochs):
        shuffled = generator.permutation(training)
        totals: dict[str, float] = {}
        batches = 0
        for start in range(0, len(shuffled), arguments.batch_size):
            selected = shuffled[start:start + arguments.batch_size]
            (_, metrics), gradients = loss_and_gradient(
                mx.array(observation[selected]), mx.array(dataset.actions[selected]),
                mx.array(next_observation[selected]), mx.array(dataset.rewards[selected]),
                mx.array(dataset.dones[selected]),
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, metrics)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
            batches += 1
        history.append({key: value / batches for key, value in totals.items()})
    validation_observation = mx.array(observation[validation])
    _, validation_metrics = loss_function(
        validation_observation, mx.array(dataset.actions[validation]),
        mx.array(next_observation[validation]), mx.array(dataset.rewards[validation]),
        mx.array(dataset.dones[validation]),
    )
    mx.eval(validation_metrics)
    validation_record = {
        key: float(value.item()) for key, value in validation_metrics.items()
    }
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.safetensors"
    model.save_weights(str(model_path))
    payload = {
        "classification": "simulated latent dynamics model; not physical qualification",
        "task": V10_TASK,
        "observation_count": dataset.observation_count,
        "action_count": dataset.action_count,
        "hidden_count": arguments.hidden,
        "latent_count": arguments.latent,
        "transition_count": len(dataset.observations),
        "replay_count": len(arguments.replay),
        "courses": sorted(set(dataset.courses)),
        "data_quality": {
            "scheduled_resets": sorted(set(dataset.scheduled_resets)),
            "maximum_root_height_m": dataset.maximum_root_height,
            "airborne_frame_fraction": dataset.airborne_frame_fraction,
            "airborne_height_threshold_m": 0.35,
        },
        "fingerprints": dataset.fingerprints,
        "observation_mean": mean.tolist(),
        "observation_standard_deviation": std.tolist(),
        "training": {
            "seed": arguments.seed, "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "final": history[-1], "validation": validation_record,
        },
        "weights": model_path.name,
        "weights_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    _write_envelope(output / "manifest.json", MODEL_SCHEMA, payload)
    return payload


def _load_model(directory: Path) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray]:
    _require_mlx()
    payload = _read_envelope(directory / "manifest.json", MODEL_SCHEMA)
    model = CrowLatentWorldModel(
        int(payload["observation_count"]), int(payload["action_count"]),
        int(payload["hidden_count"]), int(payload["latent_count"]),
    )
    weights = directory / str(payload["weights"])
    if hashlib.sha256(weights.read_bytes()).hexdigest() != payload["weights_sha256"]:
        raise ValueError("world-model weights hash is invalid")
    model.load_weights(str(weights))
    mx.eval(model.parameters())
    return (
        model, payload,
        np.asarray(payload["observation_mean"], dtype=np.float32),
        np.asarray(payload["observation_standard_deviation"], dtype=np.float32),
    )


def _predicted_return(model: Any, starts: np.ndarray, actions: np.ndarray) -> np.ndarray:
    batch, horizon, _ = actions.shape
    latent = model.encoder(mx.array(starts))
    survival = mx.ones((batch,), dtype=mx.float32)
    score = mx.zeros((batch,), dtype=mx.float32)
    previous = mx.zeros_like(mx.array(actions[:, 0]))
    for step in range(horizon):
        action = mx.array(actions[:, step])
        joined = mx.concatenate((latent, action), axis=-1)
        latent = latent + model.transition(joined)
        reward = model.reward_head(mx.concatenate((latent, action), axis=-1))[:, 0]
        done = mx.sigmoid(model.done_head(latent)[:, 0])
        smoothness = mx.mean(mx.square(action - previous), axis=-1)
        score = score + survival * (reward - 0.002 * smoothness - 2.0 * done)
        survival = survival * (1.0 - done)
        previous = action
    mx.eval(score)
    return np.asarray(score, dtype=np.float32)


def plan_demonstrations(arguments: argparse.Namespace) -> dict[str, Any]:
    model, manifest, mean, std = _load_model(Path(arguments.model).resolve())
    dataset = load_replays([Path(value) for value in arguments.replay])
    if dataset.fingerprints != manifest["fingerprints"]:
        raise ValueError("planner replay fingerprints disagree with the model")
    normalized = (dataset.observations - mean) / std
    generator = np.random.default_rng(arguments.seed)
    selected = np.linspace(
        0, len(normalized) - 1,
        num=min(arguments.starts, len(normalized)), dtype=np.int64,
    )
    demonstrations: list[dict[str, Any]] = []
    for sample in selected:
        baseline = np.repeat(dataset.actions[sample][None], arguments.horizon, axis=0)
        average = baseline.copy()
        deviation = np.ones_like(average) * np.float32(arguments.initial_std)
        start = np.repeat(normalized[sample][None], arguments.candidates, axis=0)
        for _ in range(arguments.iterations):
            candidates = np.clip(
                average[None] + deviation[None] * generator.standard_normal(
                    (arguments.candidates, arguments.horizon, dataset.action_count)
                ),
                baseline[None] - arguments.maximum_action_delta,
                baseline[None] + arguments.maximum_action_delta,
            ).astype(np.float32)
            candidates = np.clip(candidates, -1.0, 1.0)
            scores = _predicted_return(model, start, candidates)
            elite = candidates[np.argpartition(scores, -arguments.elites)[-arguments.elites:]]
            average = elite.mean(axis=0)
            deviation = np.maximum(elite.std(axis=0), 0.05)
        planned = np.clip(
            average,
            baseline - arguments.maximum_action_delta,
            baseline + arguments.maximum_action_delta,
        )
        planned = np.clip(planned, -1.0, 1.0).astype(np.float32)
        planned_return = float(_predicted_return(model, normalized[sample][None], planned[None])[0])
        baseline_return = float(_predicted_return(model, normalized[sample][None], baseline[None])[0])
        if planned_return - baseline_return < arguments.minimum_predicted_improvement:
            planned = baseline.copy()
            planned_return = baseline_return
        demonstrations.append({
            "source_transition": int(sample),
            "observation": dataset.observations[sample].tolist(),
            "planner_actions": planned.tolist(),
            "baseline_actions": baseline.tolist(),
            "planner_predicted_return": planned_return,
            "baseline_predicted_return": baseline_return,
            "predicted_improvement": planned_return - baseline_return,
        })
    payload = {
        "classification": "model-predicted demonstrations; requires native execution",
        "task": V10_TASK,
        "fingerprints": dataset.fingerprints,
        "model_manifest_sha256": _sha256(manifest),
        "planner": {
            "algorithm": "cross_entropy_method", "seed": arguments.seed,
            "horizon": arguments.horizon, "candidates": arguments.candidates,
            "elites": arguments.elites, "iterations": arguments.iterations,
            "maximum_action_delta": arguments.maximum_action_delta,
            "minimum_predicted_improvement": arguments.minimum_predicted_improvement,
        },
        "demonstrations": demonstrations,
    }
    _write_envelope(Path(arguments.output), DEMONSTRATION_SCHEMA, payload)
    return payload


def distill_student(arguments: argparse.Namespace) -> dict[str, Any]:
    _require_mlx()
    demonstrations = _read_envelope(Path(arguments.demonstrations), DEMONSTRATION_SCHEMA)
    model_manifest = _read_envelope(
        Path(arguments.model) / "manifest.json", MODEL_SCHEMA
    )
    records = demonstrations.get("demonstrations", [])
    if not records:
        raise ValueError("planner artifact contains no demonstrations")
    observation_count = int(model_manifest["observation_count"])
    action_count = int(model_manifest["action_count"])
    planner_observations = _finite_matrix(
        [record["observation"] for record in records], observation_count,
        "demonstration observations",
    )
    planner_actions = _finite_matrix(
        [record["planner_actions"][0] for record in records], action_count,
        "planner actions",
    )
    baseline_actions = _finite_matrix(
        [record["baseline_actions"][0] for record in records], action_count,
        "baseline actions",
    )
    dataset = load_replays([Path(value) for value in arguments.replay])
    if dataset.fingerprints != demonstrations["fingerprints"]:
        raise ValueError("distillation replay fingerprints disagree with planner")
    from .mlx_policy_learning import read_policy_pack

    base_path = Path(arguments.base_policy_pack).resolve()
    base = read_policy_pack(base_path, library_path=arguments.library)
    if (
        base.actor_observation_count != observation_count
        or base.action_count != action_count
        or {
            "world_fingerprint": str(base.world_fingerprint),
            "task_fingerprint": str(base.task_fingerprint),
            "observation_fingerprint": str(base.observation_fingerprint),
            "action_fingerprint": str(base.action_fingerprint),
        } != demonstrations["fingerprints"]
    ):
        raise ValueError("base PolicyPack disagrees with the v10 planner contract")
    widths = [observation_count] + [layer.output_count for layer in base.layers]
    activations = [layer.activation for layer in base.layers]
    student = CrowStudent(widths, activations)
    for destination, source in zip(student.layers, base.layers, strict=True):
        destination.weight = mx.array(source.weights)
        destination.bias = mx.array(source.bias)
    mean = base.effective_observation_mean.astype(np.float32)
    inverse_std = base.effective_observation_inverse_standard_deviation.astype(
        np.float32
    )
    normalized_replay = np.clip(
        (dataset.observations - mean) * inverse_std,
        -base.observation_clip,
        base.observation_clip,
    )
    normalized_planner = np.clip(
        (planner_observations - mean) * inverse_std,
        -base.observation_clip,
        base.observation_clip,
    )
    base_replay_targets = base.actor_mean(dataset.observations)
    correction_targets = np.clip(
        baseline_actions
        + arguments.planner_blend * (planner_actions - baseline_actions),
        -1.0,
        1.0,
    )
    observations = np.concatenate(
        [normalized_replay]
        + [normalized_planner] * arguments.planner_repeats
    )
    targets = np.concatenate(
        [base_replay_targets]
        + [correction_targets] * arguments.planner_repeats
    )
    optimizer = optim.Adam(learning_rate=arguments.learning_rate)
    optimizer.init(student.trainable_parameters())

    def loss_function(source: Any, target: Any):
        prediction = student(source)
        loss = mx.mean(mx.square(prediction - target))
        return loss, {"action_mse": loss}

    loss_and_gradient = nn.value_and_grad(student, loss_function)
    generator = np.random.default_rng(arguments.seed)
    final_loss = math.inf
    for _ in range(arguments.epochs):
        order = generator.permutation(len(observations))
        for start in range(0, len(observations), arguments.batch_size):
            selected = order[start:start + arguments.batch_size]
            (_, metrics), gradients = loss_and_gradient(
                mx.array(observations[selected]), mx.array(targets[selected])
            )
            optimizer.update(student, gradients)
            mx.eval(student.parameters(), optimizer.state, metrics)
            final_loss = float(metrics["action_mse"].item())
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "student.safetensors"
    student.save_weights(str(weights_path))
    policy_path = ""
    replay_prediction = np.asarray(student(mx.array(normalized_replay)))
    planner_prediction = np.asarray(student(mx.array(normalized_planner)))
    baseline_imitation_mse = float(
        np.mean(np.square(replay_prediction - base_replay_targets))
    )
    planner_target_mse = float(
        np.mean(np.square(planner_prediction - correction_targets))
    )
    maximum_action_delta = float(
        np.max(np.abs(planner_prediction - baseline_actions))
    )
    if arguments.policy_pack:
        from .native import PolicyDenseLayerArtifact, write_policy_pack

        def layer(source: Any, activation: int) -> PolicyDenseLayerArtifact:
            return PolicyDenseLayerArtifact(
                weights=np.asarray(source.weight, dtype=np.float32),
                bias=np.asarray(source.bias, dtype=np.float32),
                activation=activation,
            )

        policy_path = str(Path(arguments.policy_pack).resolve())
        write_policy_pack(
            policy_path,
            policy_id=arguments.policy_id,
            revision=arguments.revision,
            contract_version=1,
            world_fingerprint=int(demonstrations["fingerprints"]["world_fingerprint"]),
            task_fingerprint=int(demonstrations["fingerprints"]["task_fingerprint"]),
            observation_fingerprint=int(
                demonstrations["fingerprints"]["observation_fingerprint"]
            ),
            action_fingerprint=int(demonstrations["fingerprints"]["action_fingerprint"]),
            layers=tuple(
                layer(source, activation)
                for source, activation in zip(
                    student.layers, activations, strict=True
                )
            ),
            observation_mean=mean,
            observation_inverse_standard_deviation=inverse_std,
            observation_clip=base.observation_clip,
            action_clip=1.0,
            library_path=arguments.library,
        )
    payload = {
        "classification": "distilled neural controller; not promoted",
        "task": V10_TASK,
        "fingerprints": demonstrations["fingerprints"],
        "demonstration_count": len(records),
        "topology": widths,
        "base_policy_pack": str(base_path),
        "base_policy_pack_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "training": {
            "epochs": arguments.epochs,
            "seed": arguments.seed,
            "planner_blend": arguments.planner_blend,
            "planner_repeats": arguments.planner_repeats,
            "final_action_mse": final_loss,
            "baseline_imitation_mse": baseline_imitation_mse,
            "planner_target_mse": planner_target_mse,
            "maximum_action_delta_from_baseline": maximum_action_delta,
        },
        "weights": weights_path.name,
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "policy_pack": policy_path,
    }
    _write_envelope(output / "manifest.json", STUDENT_SCHEMA, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--replay", action="append", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--hidden", type=int, default=256)
    train.add_argument("--latent", type=int, default=64)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument("--minimum-maximum-root-height", type=float, default=0.5)
    train.add_argument("--minimum-airborne-fraction", type=float, default=0.05)
    train.add_argument("--seed", type=int, default=20260828)
    plan = commands.add_parser("plan")
    plan.add_argument("--model", required=True)
    plan.add_argument("--replay", action="append", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--starts", type=int, default=64)
    plan.add_argument("--horizon", type=int, default=20)
    plan.add_argument("--candidates", type=int, default=256)
    plan.add_argument("--elites", type=int, default=32)
    plan.add_argument("--iterations", type=int, default=5)
    plan.add_argument("--initial-std", type=float, default=0.55)
    plan.add_argument("--maximum-action-delta", type=float, default=0.15)
    plan.add_argument("--minimum-predicted-improvement", type=float, default=0.0)
    plan.add_argument("--seed", type=int, default=20260829)
    distill = commands.add_parser("distill")
    distill.add_argument("--model", required=True)
    distill.add_argument("--demonstrations", required=True)
    distill.add_argument("--replay", action="append", required=True)
    distill.add_argument("--base-policy-pack", required=True)
    distill.add_argument("--output", required=True)
    distill.add_argument("--epochs", type=int, default=100)
    distill.add_argument("--batch-size", type=int, default=64)
    distill.add_argument("--learning-rate", type=float, default=3.0e-4)
    distill.add_argument("--planner-blend", type=float, default=0.1)
    distill.add_argument("--planner-repeats", type=int, default=8)
    distill.add_argument("--seed", type=int, default=20260830)
    distill.add_argument("--policy-pack")
    distill.add_argument("--policy-id", default="crow_navigation_v10_world_model_student")
    distill.add_argument("--revision", type=int, default=1)
    distill.add_argument("--library")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    started = time.perf_counter()
    if arguments.command == "train":
        if (
            arguments.minimum_maximum_root_height < 0.0
            or not 0.0 <= arguments.minimum_airborne_fraction <= 1.0
        ):
            raise ValueError("training data-quality thresholds are invalid")
        payload = train_world_model(arguments)
    elif arguments.command == "plan":
        if arguments.elites <= 0 or arguments.elites > arguments.candidates:
            raise ValueError("planner elites must be in [1, candidates]")
        if not 0.0 < arguments.maximum_action_delta <= 2.0:
            raise ValueError("maximum action delta must be in (0, 2]")
        payload = plan_demonstrations(arguments)
    else:
        if not 0.0 <= arguments.planner_blend <= 1.0:
            raise ValueError("planner blend must be in [0, 1]")
        if arguments.planner_repeats <= 0:
            raise ValueError("planner repeats must be positive")
        payload = distill_student(arguments)
    print(json.dumps({
        "status": "ok", "command": arguments.command,
        "task": payload["task"], "elapsed_seconds": time.perf_counter() - started,
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
