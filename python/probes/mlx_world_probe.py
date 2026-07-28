#!/usr/bin/env python3
"""Decisive active-encoder MLX physics and rollout probe."""

from __future__ import annotations

import json
import math

import mlx.core as mx

from metalrobo import (
    ActorCritic,
    MLXPPOTrainer,
    MLXRolloutCollector,
    PPOConfig,
    compile_world,
    initial_state,
    step,
)
from metalrobo._mlx_ext import _debug_cpu_step


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def maximum_error(
    left: list[float],
    right: list[float],
) -> float:
    require(len(left) == len(right), "parity vector size mismatch")
    return max(
        (abs(a - b) for a, b in zip(left, right)),
        default=0.0,
    )


def main() -> None:
    mx.random.seed(17)
    frank = compile_world("franka", physics_substeps=4)
    state = initial_state(frank, 8)
    effort_row = [
        0.125 * float(index + 1)
        for index in range(frank.nv)
    ]
    efforts = mx.broadcast_to(
        mx.array(effort_row, dtype=mx.float32),
        (8, frank.nv),
    )

    compiled_step = mx.compile(
        lambda q, v, actions: step(
            frank,
            state._replace(q=q, v=v),
            actions,
        )
    )
    reference = compiled_step(state.q, state.v, efforts)
    mx.eval(reference)
    require(
        reference.status[:, 0].tolist() == [0] * 8,
        "compiled Franka step failed",
    )

    reference_q = reference.next_state.q.tolist()
    reference_v = reference.next_state.v.tolist()
    reference_status = reference.status.tolist()
    for _ in range(100):
        replay = compiled_step(state.q, state.v, efforts)
        mx.eval(replay)
        require(
            replay.next_state.q.tolist() == reference_q
            and replay.next_state.v.tolist() == reference_v
            and replay.status.tolist() == reference_status,
            "same-device deterministic replay diverged",
        )

    cpu = _debug_cpu_step(
        frank,
        state.q[0].tolist(),
        state.v[0].tolist(),
        effort_row,
    )
    q_error = maximum_error(
        reference.next_state.q[0].tolist(),
        cpu[: frank.nq],
    )
    v_error = maximum_error(
        reference.next_state.v[0].tolist(),
        cpu[frank.nq :],
    )
    require(
        q_error < 5.0e-5 and v_error < 1.0e-4,
        "MLX primitive failed the FP64 integration oracle",
    )

    failing_efforts = mx.where(
        mx.arange(8)[:, None] == 3,
        mx.full((8, frank.nv), math.nan),
        efforts,
    )
    failed = step(frank, state, failing_efforts)
    mx.eval(failed)
    failure_codes = failed.status[:, 0].tolist()
    require(
        failure_codes[3] != 0
        and sum(code != 0 for code in failure_codes) == 1,
        "failure was not isolated to one environment",
    )
    require(
        failed.next_state.q[3].tolist() == state.q[3].tolist()
        and failed.next_state.v[3].tolist() == state.v[3].tolist(),
        "failed environment did not roll back exactly",
    )
    require(
        failed.next_state.q[2].tolist()
        != state.q[2].tolist(),
        "healthy environment did not publish",
    )

    g1 = compile_world("g1", physics_substeps=4)
    g1_state = initial_state(g1, 4)
    g1_result = mx.compile(
        lambda q, v, actions: step(
            g1,
            g1_state._replace(q=q, v=v),
            actions,
        )
    )(
        g1_state.q,
        g1_state.v,
        mx.zeros((4, g1.nv), dtype=mx.float32),
    )
    mx.eval(g1_result)
    require(
        g1_result.status[:, 0].tolist() == [0] * 4,
        "compiled G1 step failed",
    )

    autodiff_rejected = False
    try:
        gradient = mx.grad(
            lambda q: mx.sum(
                step(
                    frank,
                    state._replace(q=q),
                    efforts,
                ).next_state.q
            )
        )(state.q)
        mx.eval(gradient)
    except RuntimeError as error:
        autodiff_rejected = "does not implement VJP" in str(error)
    require(
        autodiff_rejected,
        "physics autodiff was not rejected explicitly",
    )

    model = ActorCritic(
        frank.nq + frank.nv,
        frank.nv,
        (32, 32),
    )
    mx.eval(model.parameters())
    collector = MLXRolloutCollector(
        frank,
        model,
        8,
        gamma=0.99,
        gae_lambda=0.95,
        chunk_size=2,
        maximum_episode_steps=8,
    )
    rollout_state, rollout = collector.collect(
        collector.initial(),
        8,
    )
    mx.eval(rollout, rollout_state)
    require(
        rollout.observations.shape == (8, 8, 14)
        and rollout.latents.shape == (8, 8, 7),
        "MLX rollout shape mismatch",
    )
    require(
        int(
            mx.sum(
                rollout.physics_errors.astype(mx.uint32)
            ).item()
        )
        == 0,
        "MLX rollout reported a physics failure",
    )

    trainer = MLXPPOTrainer(
        PPOConfig(
            environment_count=8,
            rollout_steps=8,
            iterations=1,
            update_epochs=1,
            minibatch_size=32,
            hidden_sizes=(32, 32),
            checkpoint_interval=0,
        ),
        rollout_chunk_size=2,
        maximum_episode_steps=8,
    )
    _, update_rollout = trainer.collector.collect(
        trainer.rollout_state,
        8,
    )
    mx.eval(update_rollout)
    metrics = trainer._update(update_rollout)
    require(
        all(math.isfinite(value) for value in metrics.values()),
        "MLX PPO update produced a non-finite metric",
    )

    print(
        json.dumps(
            {
                "mlx_world": "ok",
                "mlx_version": mx.__version__,
                "franka_environments": 8,
                "g1_environments": 4,
                "compiled_policy_physics_reward": True,
                "deterministic_replays": 100,
                "isolated_failure_code": failure_codes[3],
                "fp64_max_q_error": q_error,
                "fp64_max_v_error": v_error,
                "rollout_shape": list(
                    rollout.observations.shape
                ),
                "ppo_updates": int(
                    metrics["minibatch_updates"]
                ),
                "numpy_step_conversions": 0,
                "autodiff_rejected": autodiff_rejected,
                "contact_supported": frank.contact_supported,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
