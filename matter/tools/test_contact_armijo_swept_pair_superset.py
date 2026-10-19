#!/usr/bin/env python3
"""Bounded proof for the FEM Armijo broadphase reuse envelope.

This checks the geometry invariant only; it does not simulate Matter or claim
an integrated/native qualification. Every per-alpha sweep validator and
narrowphase remains in the production path.
"""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THICKNESS = 0.015


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def scale(a, s):
    return tuple(x * s for x in a)


def interpolate(base, delta, alpha):
    return add(base, scale(delta, alpha))


def face_bounds(face, alpha, envelope_alpha=None):
    """Mirror each face's swept endpoint AABB, optionally first-alpha envelope."""
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    samples = []
    for base, start_delta, finish_delta in face:
        samples.extend((base, interpolate(base, start_delta, alpha),
                        interpolate(base, finish_delta, alpha)))
        if envelope_alpha is not None:
            samples.extend((interpolate(base, start_delta, envelope_alpha),
                            interpolate(base, finish_delta, envelope_alpha)))
    for point in samples:
        for axis in range(3):
            low[axis] = min(low[axis], point[axis] - THICKNESS)
            high[axis] = max(high[axis], point[axis] + THICKNESS)
    return tuple(low), tuple(high)


def overlaps(a, b):
    alo, ahi = a
    blo, bhi = b
    return all(alo[i] <= bhi[i] and blo[i] <= ahi[i] for i in range(3))


def cache_contains(face, alpha, envelope_alpha):
    """Require the current swept bounds to stay within the cached raw envelope."""
    cached = face_bounds(face, alpha=envelope_alpha)
    current = face_bounds(face, alpha=alpha)
    return all(cached[0][i] <= current[0][i] and
               current[1][i] <= cached[1][i] for i in range(3))


def make_face(rng):
    face = []
    center = tuple(rng.uniform(-1.0, 1.0) for _ in range(3))
    for corner in range(3):
        base = tuple(center[i] + rng.uniform(-0.08, 0.08) for i in range(3))
        start_delta = tuple(rng.uniform(-0.15, 0.15) for _ in range(3))
        finish_delta = tuple(rng.uniform(-0.25, 0.25) for _ in range(3))
        face.append((base, start_delta, finish_delta))
    return face


def main():
    rng = random.Random(0x4E554D49)
    pair_checks = 0
    face_checks = 0
    for _scene in range(40):
        alpha0 = rng.uniform(0.05, 1.0)
        faces = [make_face(rng) for _ in range(24)]
        cached = [face_bounds(face, alpha0, alpha0) for face in faces]
        cached_pairs = {(i, j) for i in range(len(faces))
                        for j in range(i + 1, len(faces))
                        if overlaps(cached[i], cached[j])}
        for trial in range(1, 8):
            alpha = alpha0 * (2.0 ** -trial)
            current = [face_bounds(face, alpha) for face in faces]
            for i, face in enumerate(faces):
                assert cache_contains(face, alpha, alpha0)
                face_checks += 1
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    if overlaps(current[i], current[j]):
                        assert (i, j) in cached_pairs, (
                            f"trial pair escaped envelope at alpha={alpha}")
                    pair_checks += 1

    escaped = make_face(rng)
    alpha0 = 0.75
    assert cache_contains(escaped, alpha0 / 2.0, alpha0)
    cached_escape_bounds = face_bounds(escaped, alpha0, alpha0)
    # A late endpoint outside the first-trial envelope must select the existing
    # full-rebuild path instead of trusting the cached pair set.
    base, start_delta, finish_delta = escaped[0]
    escaped[0] = (base, start_delta, add(finish_delta, (2.0, 0.0, 0.0)))
    escaped_bounds = face_bounds(escaped, alpha0 / 2.0)
    assert not all(cached_escape_bounds[0][axis] <= escaped_bounds[0][axis] and
                   escaped_bounds[1][axis] <= cached_escape_bounds[1][axis]
                   for axis in range(3))

    fem = (ROOT / "matter/src/metal/fem.metalinc").read_text()
    contact = (ROOT / "matter/src/metal/contact.metalinc").read_text()
    assert "initialBound * exp2(-float(trialIndex))" in fem
    assert "nm_contact_validate_armijo_deformable_sweep" in contact
    assert "kernel void nm_contact_narrowphase_deformable(" in contact
    assert "nm_contact_validate_armijo_surface_cache" in contact

    print("armijo_surface_pair_superset_checks=pass "
          f"faces={face_checks} pair_trials={pair_checks} "
          "escape_fallback=pass physical_steps=0")


if __name__ == "__main__":
    main()
