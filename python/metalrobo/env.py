"""A dependency-free vector-environment facade for the native Franka task."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt

from .native import FloatArray, NativeRuntime


class FrankaEnv:
    """Vectorized Franka environment backed by one Metal command stream.

    The five arrays returned by :meth:`step` follow Gymnasium's vector API
    ordering. Observations, rewards, and terminations alias native memory;
    truncations are always false because the native task reports its horizon
    through ``terminated`` and performs device-side automatic reset.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int = 1024,
        *,
        seed: int = 1,
        library_path: str | os.PathLike[str] | None = None,
        metallib_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.runtime = NativeRuntime(
            num_envs,
            seed=seed,
            library_path=library_path,
            metallib_path=metallib_path,
        )
        self.num_envs = self.runtime.environment_count
        self.single_observation_shape = (self.runtime.observation_count,)
        self.single_action_shape = (self.runtime.action_count,)
        self.observation_shape = (self.num_envs, *self.single_observation_shape)
        self.action_shape = (self.num_envs, *self.single_action_shape)
        self._seed = int(seed)
        self._truncated = np.zeros(self.num_envs, dtype=np.bool_)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    @property
    def observations(self) -> FloatArray:
        return self.runtime.observations

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        if seed is not None:
            self._seed = int(seed)
        self._episode_returns.fill(0)
        self._episode_lengths.fill(0)
        observations = self.runtime.reset(self._seed)
        return observations, {
            "device": self.runtime.device_name,
            "native_version": self.runtime.version,
        }

    def step(
        self, actions: npt.ArrayLike
    ) -> tuple[
        FloatArray,
        FloatArray,
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        dict[str, Any],
    ]:
        observations = self.runtime.step(actions)
        rewards = self.runtime.rewards
        terminated = self.runtime.terminated

        self._episode_returns += rewards
        self._episode_lengths += 1
        completed = np.flatnonzero(terminated)
        info: dict[str, Any] = {"runtime_stats": self.runtime.stats}
        if completed.size:
            info["final_info"] = {
                "indices": completed.copy(),
                "returns": self._episode_returns[completed].copy(),
                "lengths": self._episode_lengths[completed].copy(),
            }
            self._episode_returns[completed] = 0
            self._episode_lengths[completed] = 0

        return observations, rewards, terminated, self._truncated, info

    def close(self) -> None:
        self.runtime.close()

    def __enter__(self) -> FrankaEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
