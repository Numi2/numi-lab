#!/usr/bin/env python3
"""Acquire and fingerprint the pinned PX4 X500 source model without altering it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as etree
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "robots" / "px4_x500" / "UPSTREAM.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(element: etree.Element | None, name: str) -> str:
    if element is None:
        raise ValueError(f"missing XML element: {name}")
    value = element.text
    if value is None or not value.strip():
        raise ValueError(f"empty XML element: {name}")
    return value.strip()


def scalar(element: etree.Element | None, name: str) -> float:
    return float(text(element, name))


def read_source_contract(source: Path) -> dict[str, object]:
    base = etree.parse(source / "models/x500_base/model.sdf").getroot()
    vehicle = etree.parse(source / "models/x500/model.sdf").getroot()
    base_link = base.find("./model/link[@name='base_link']")
    if base_link is None:
        raise ValueError("PX4 X500 source has no base_link")
    inertia = base_link.find("./inertial/inertia")
    if inertia is None:
        raise ValueError("PX4 X500 base_link has no inertia")
    rotors: list[dict[str, object]] = []
    for link in base.findall("./model/link"):
        if not link.attrib.get("name", "").startswith("rotor_"):
            continue
        joint = base.find(f"./model/joint[child='{link.attrib['name']}']")
        if joint is None:
            raise ValueError(f"missing joint for {link.attrib['name']}")
        rotor_inertia = link.find("./inertial/inertia")
        if rotor_inertia is None:
            raise ValueError(f"missing inertia for {link.attrib['name']}")
        rotors.append({
            "link": link.attrib["name"],
            "joint": joint.attrib["name"],
            "pose": [float(value) for value in text(link.find("pose"), "rotor pose").split()],
            "mass_kg": scalar(link.find("./inertial/mass"), "rotor mass"),
            "inertia_kg_m2": {
                axis: (
                    scalar(rotor_inertia.find(axis), axis)
                    if rotor_inertia.find(axis) is not None
                    else 0.0
                )
                for axis in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
            },
        })
    motors: list[dict[str, object]] = []
    for plugin in vehicle.findall("./model/plugin"):
        if plugin.attrib.get("filename") != "gz-sim-multicopter-motor-model-system":
            continue
        motors.append({
            "joint": text(plugin.find("jointName"), "jointName"),
            "link": text(plugin.find("linkName"), "linkName"),
            "turning_direction": text(plugin.find("turningDirection"), "turningDirection"),
            "time_constant_up_s": scalar(plugin.find("timeConstantUp"), "timeConstantUp"),
            "time_constant_down_s": scalar(plugin.find("timeConstantDown"), "timeConstantDown"),
            "maximum_rotor_velocity_rad_s": scalar(plugin.find("maxRotVelocity"), "maxRotVelocity"),
            "thrust_coefficient": scalar(plugin.find("motorConstant"), "motorConstant"),
            "moment_coefficient": scalar(plugin.find("momentConstant"), "momentConstant"),
            "rotor_drag_coefficient": scalar(plugin.find("rotorDragCoefficient"), "rotorDragCoefficient"),
            "rolling_moment_coefficient": scalar(plugin.find("rollingMomentCoefficient"), "rollingMomentCoefficient"),
            "motor_number": int(text(plugin.find("motorNumber"), "motorNumber")),
        })
    if len(rotors) != 4 or len(motors) != 4:
        raise ValueError(f"expected four X500 rotors and motors, got {len(rotors)} and {len(motors)}")
    sensors = [
        {"name": sensor.attrib["name"], "type": sensor.attrib["type"], "update_rate_hz": scalar(sensor.find("update_rate"), "sensor update_rate")}
        for sensor in base.findall("./model/link[@name='base_link']/sensor")
    ]
    return {
        "base_link": {
            "mass_kg": scalar(base_link.find("./inertial/mass"), "base mass"),
            "inertia_kg_m2": {axis: scalar(inertia.find(axis), axis) for axis in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")},
        },
        "rotors": sorted(rotors, key=lambda rotor: rotor["joint"]),
        "motors": sorted(motors, key=lambda motor: motor["motor_number"]),
        "sensors": sensors,
    }


def checkout(manifest: dict[str, object]) -> tempfile.TemporaryDirectory[str]:
    upstream = manifest["upstream"]
    assert isinstance(upstream, dict)
    temporary = tempfile.TemporaryDirectory(prefix="numi-px4-x500-")
    destination = Path(temporary.name) / "PX4-gazebo-models"
    subprocess.run(["git", "clone", "--no-checkout", str(upstream["repository"]), str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", str(upstream["commit"])], check=True)
    return temporary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, help="existing PX4-gazebo-models checkout; its HEAD must match the pin")
    parser.add_argument("--output", type=Path, required=True, help="destination for the immutable source snapshot and contract")
    options = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text())
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if options.source_directory:
        source = options.source_directory.resolve()
    else:
        temporary = checkout(manifest)
        source = Path(temporary.name) / "PX4-gazebo-models"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    expected_revision = manifest["upstream"]["commit"]
    if revision != expected_revision:
        raise RuntimeError(f"PX4 source revision mismatch: expected {expected_revision}, got {revision}")
    files = manifest["required_files"]
    assert isinstance(files, dict)
    for relative, expected_hash in files.items():
        observed = sha256(source / relative)
        if observed != expected_hash:
            raise RuntimeError(f"PX4 source hash mismatch for {relative}: expected {expected_hash}, got {observed}")
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    for relative in [*files, manifest["upstream"]["license_file"]]:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    contract = {
        "schema": "numi.px4-x500-source-contract.v1",
        "provenance": manifest,
        "resolved_revision": revision,
        "source_hashes": {relative: sha256(snapshot / relative) for relative in files},
        "mechanics": read_source_contract(snapshot),
        "boundary": "Source extraction only. No SDF coordinate, aerodynamic, controller, sensor, collision, or hardware-fidelity term has been changed or qualified by this artifact.",
    }
    (output / "x500.source.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(f"source=PX4 X500 revision={revision} output={output}")
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
