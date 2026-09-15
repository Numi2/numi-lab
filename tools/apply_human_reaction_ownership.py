"""Remove permanent static constraint loads from the current Human runner.

The exact initial reference remains diagnostic. Runtime equality/limit loads
must be reactions of the current constrained solve, not constant actuation.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

BASE_BLOB = "bf64acfb36a04eae889546e9e82713d8d8266109"
MARKER = "Runtime constraint reactions are solved, never permanently preloaded."


def transform(source: str) -> str:
    raw = source.encode("utf-8")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != BASE_BLOB:
        raise ValueError(f"Human runner changed: {blob}; expected {BASE_BLOB}")
    old = """        preloadedGeneralizedForce[dof] = static_cast<float>(
            compiledActivation.generalizedJointEqualityForce[dof] +
            compiledActivation.generalizedPositionLimitForce[dof] +
            runtimePassiveForce
        );"""
    new = """        // Runtime constraint reactions are solved, never permanently preloaded.
        // A static stop or equality force is not a time-invariant actuator;
        // retaining it prevents complete unloading as the body moves.
        // The initial passive-tissue preload is retained as the existing
        // bounded-release approximation, NOT a calibrated dynamic tissue law.
        preloadedGeneralizedForce[dof] = static_cast<float>(runtimePassiveForce);"""
    if source.count(old) != 1:
        raise ValueError("expected exactly one static constraint preload owner")
    source = source.replace(old, new, 1)
    # This record combines measured initial muscle force and static reference
    # reactions. It must not masquerade as the runtime impulse/force solution.
    for old, new in (
        ("persistent_dynamic_force_audit=", "persistent_initial_force_reference="),
        ("numi.human.persistent-dynamic-force-audit.v1",
         "numi.human.persistent-initial-force-reference.v1"),
    ):
        if source.count(old) != 1:
            raise ValueError(f"expected exactly one diagnostic token: {old}")
        source = source.replace(old, new, 1)
    return source


def verify(source: str) -> None:
    if source.count(MARKER) != 1:
        raise ValueError("reaction ownership correction missing")
    expression = source.split("preloadedGeneralizedForce[dof] =", 1)[1].split(";", 1)[0]
    if "generalizedJointEqualityForce" in expression or "generalizedPositionLimitForce" in expression:
        raise ValueError("static constraint reaction reintroduced as a permanent force")
    if expression.strip() != "static_cast<float>(runtimePassiveForce)":
        raise ValueError("unexpected permanent preload owner")
    if "persistent_dynamic_force_audit=" in source:
        raise ValueError("static reference is incorrectly labelled dynamic evidence")
    if source.count("persistent_initial_force_reference=") != 1:
        raise ValueError("initial force reference is missing or duplicated")


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "apps/numilab_human_myosim_visual_probe.mm"
    source = path.read_text(encoding="utf-8")
    revised = source if MARKER in source else transform(source)
    verify(revised)
    if revised != source:
        temporary = path.with_suffix(".mm.tmp")
        temporary.write_text(revised, encoding="utf-8")
        temporary.replace(path)
    print("Human runtime constraint reactions are not permanent source forces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
