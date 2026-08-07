"""Exact lifting of Numi PolicyPack actors into low-rank hyper-policy bases."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from .base import HyperBaseLayer, HyperBasePolicy
from .mlx_model import PhaseVaryingLowRankActor
from ..mlx_policy_learning import NativePolicyPack, read_policy_pack


def initialize_hyper_base_from_policy_pack(
    policy_pack: NativePolicyPack | str | Path,
    *,
    ranks: Sequence[int],
    coefficient_limit: float | Sequence[float] = 1.0,
    seed: int = 1,
) -> HyperBasePolicy:
    """Lift an existing physical actor into the canonical low-rank basis.

    Base weights are copied exactly.  Adapter bases are orthogonal/small and the
    generated coefficients start at zero, so the lifted actor is behaviorally
    identical to the source PolicyPack before meta-training.
    """

    pack = (
        read_policy_pack(policy_pack)
        if isinstance(policy_pack, (str, Path))
        else policy_pack
    )
    if len(ranks) != len(pack.layers):
        raise ValueError("one adapter rank is required for every actor layer")
    generator = np.random.default_rng(seed)
    layers: list[HyperBaseLayer] = []
    for index, (source, rank) in enumerate(zip(pack.layers, ranks, strict=True)):
        rank = int(rank)
        if rank <= 0 or rank > min(source.input_count, source.output_count):
            raise ValueError(f"adapter rank for actor layer {index} is invalid")
        down = _orthogonal_rows(
            generator,
            rows=rank,
            columns=source.input_count,
        )
        up = _orthogonal_columns(
            generator,
            rows=source.output_count,
            columns=rank,
        )
        bias_basis = np.zeros((source.output_count, rank), dtype=np.float32)
        layers.append(
            HyperBaseLayer(
                input_count=source.input_count,
                output_count=source.output_count,
                rank=rank,
                activation=source.activation,
                weight=np.asarray(source.weights, dtype=np.float32).copy(),
                bias=np.asarray(source.bias, dtype=np.float32).copy(),
                adapter_down=down,
                adapter_up=up,
                adapter_bias_basis=bias_basis,
            )
        )
    coefficient_count = sum(int(value) for value in ranks)
    limits = np.asarray(coefficient_limit, dtype=np.float32)
    if limits.ndim == 0:
        limits = np.full((coefficient_count,), float(limits), dtype=np.float32)
    if limits.shape != (coefficient_count,):
        raise ValueError("coefficient limit width is invalid")
    base = HyperBasePolicy(
        id=f"{pack.id}-hyper-base",
        revision=1,
        world_fingerprint=pack.world_fingerprint,
        task_fingerprint=pack.task_fingerprint,
        observation_fingerprint=pack.observation_fingerprint,
        action_fingerprint=pack.action_fingerprint,
        observation_mean=pack.effective_observation_mean.astype(np.float32, copy=True),
        observation_inverse_standard_deviation=(
            pack.effective_observation_inverse_standard_deviation.astype(
                np.float32, copy=True
            )
        ),
        layers=tuple(layers),
        coefficient_limits=limits,
        action_bias=pack.effective_action_bias.astype(np.float32, copy=True),
        action_scale=pack.effective_action_scale.astype(np.float32, copy=True),
        observation_clip=pack.observation_clip,
        action_clip=pack.action_clip,
    ).with_fingerprint()
    base.validate(require_fingerprint=True)
    return base


def export_hyper_base(
    actor: PhaseVaryingLowRankActor,
    template: HyperBasePolicy,
) -> HyperBasePolicy:
    """Publish trained adapter bases while preserving the immutable base actor."""

    template.validate(require_fingerprint=True)
    mx.eval(actor.parameters())
    layers: list[HyperBaseLayer] = []
    for runtime, source in zip(actor.layers, template.layers, strict=True):
        layers.append(
            HyperBaseLayer(
                input_count=source.input_count,
                output_count=source.output_count,
                rank=source.rank,
                activation=source.activation,
                weight=source.weight.copy(),
                bias=source.bias.copy(),
                adapter_down=np.asarray(runtime.adapter_down, dtype=np.float32).copy(),
                adapter_up=np.asarray(runtime.adapter_up, dtype=np.float32).copy(),
                adapter_bias_basis=np.asarray(
                    runtime.adapter_bias_basis, dtype=np.float32
                ).copy(),
            )
        )
    return HyperBasePolicy(
        id=template.id,
        revision=template.revision + 1,
        world_fingerprint=template.world_fingerprint,
        task_fingerprint=template.task_fingerprint,
        observation_fingerprint=template.observation_fingerprint,
        action_fingerprint=template.action_fingerprint,
        observation_mean=template.observation_mean.copy(),
        observation_inverse_standard_deviation=(
            template.observation_inverse_standard_deviation.copy()
        ),
        layers=tuple(layers),
        coefficient_limits=template.coefficient_limits.copy(),
        action_bias=template.action_bias.copy(),
        action_scale=template.action_scale.copy(),
        observation_clip=template.observation_clip,
        action_clip=template.action_clip,
    ).with_fingerprint()


def _orthogonal_rows(
    generator: np.random.Generator,
    *,
    rows: int,
    columns: int,
) -> np.ndarray:
    matrix = generator.normal(size=(columns, rows))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return q.T.astype(np.float32)


def _orthogonal_columns(
    generator: np.random.Generator,
    *,
    rows: int,
    columns: int,
) -> np.ndarray:
    matrix = generator.normal(size=(rows, columns))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return q.astype(np.float32)
