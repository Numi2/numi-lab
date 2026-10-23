"""Fail-closed transactional selection for projected skin-source ancestry.

This small policy is used by the offline conforming joint-surface refinement
builder. Geometry and its source face/barycentric provenance must be committed
together: a rejected geometric move cannot replace a previously valid C map.
"""
from __future__ import annotations
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def select_source_embedding(
    proposal: T,
    step_accepted: bool,
    prior: T | None,
    is_valid: Callable[[T | None], bool],
    resolve_anchor: Callable[[], T],
) -> tuple[T, str]:
    """Select C ancestry transactionally for a projected source point.

    The proposed source face/barycentric tuple is accepted only when its paired
    geometric displacement passed its construction gate and the ancestry is
    valid. Otherwise a valid prior tuple is retained. If no valid prior exists,
    the caller must resolve ancestry for the actual retained anchor point.
    Invalid or unavailable fallback ancestry raises instead of silently
    emitting a face index with stale or incompatible geometry.
    """
    if step_accepted and is_valid(proposal):
        return proposal, "accepted_proposal"
    if prior is not None and is_valid(prior):
        return prior, "retained_prior"
    resolved = resolve_anchor()
    if not is_valid(resolved):
        raise ValueError("anchor resolver returned invalid source ancestry")
    return resolved, "resolved_anchor"
