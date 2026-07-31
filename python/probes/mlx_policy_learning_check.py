#!/usr/bin/env python3
"""Focused MLX learner check with no simulator or rollout scheduler."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from metalrobo.mlx_policy_learning import (
    MLXPPOConfiguration,
    MLXPolicyBatch,
    MLXPolicyLearner,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path)
    arguments = parser.parse_args()

    actor_count = 480
    critic_count = 696
    action_count = 29
    sample_count = 64
    learner = MLXPolicyLearner(
        actor_count,
        critic_count,
        action_count,
        MLXPPOConfiguration(
            update_epochs=1,
            minibatch_size=32,
            hidden_sizes=(32,),
            learning_rate=1.0e-4,
            target_kl=None,
            seed=17,
        ),
    )
    generator = np.random.default_rng(17)
    actor = mx.array(
        generator.normal(
            0.0,
            0.2,
            (sample_count, actor_count),
        ).astype(np.float32)
    )
    critic = mx.array(
        generator.normal(
            0.0,
            0.2,
            (sample_count, critic_count),
        ).astype(np.float32)
    )
    latents = mx.array(
        generator.normal(
            0.0,
            0.3,
            (sample_count, action_count),
        ).astype(np.float32)
    )
    old_log_probabilities, _, old_values = (
        learner.model.evaluate(actor, critic, latents)
    )
    advantages = mx.array(
        generator.normal(0.0, 1.0, sample_count).astype(
            np.float32
        )
    )
    returns = old_values + 0.1 * advantages
    mx.eval(
        old_log_probabilities,
        old_values,
        advantages,
        returns,
    )
    batch = MLXPolicyBatch(
        actor_observations=actor,
        critic_observations=critic,
        latents=latents,
        old_log_probabilities=old_log_probabilities,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
    )
    metrics = learner.update(batch)

    if arguments.output is None:
        temporary = tempfile.TemporaryDirectory()
        output = Path(temporary.name) / "learner.policypack"
    else:
        temporary = None
        output = arguments.output
    try:
        artifact = learner.write_policy_pack(
            output,
            policy_id="mlx_policy_learning_check",
            library_path=arguments.library,
        )
        if not artifact.is_file() or artifact.stat().st_size <= 32:
            raise RuntimeError("PolicyPack artifact was not published")
        print(
            json.dumps(
                {
                    "check": "mlx_policy_learning",
                    "policy_revision": learner.revision,
                    "minibatch_updates": int(
                        metrics["minibatch_updates"]
                    ),
                    "loss": metrics["loss"],
                    "artifact": str(artifact),
                    "artifact_bytes": artifact.stat().st_size,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
