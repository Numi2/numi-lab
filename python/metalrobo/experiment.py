"""Compile and run portable Numi robot projects using the native Mac app."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    stats: dict
    state_trace: Path | None
    contact_trace: Path | None


class Experiment:
    """A project and its native tools. Each run begins from the authored reset."""

    def __init__(self, project: Path, *, rollout: Path, metallib: Path):
        self.project = Path(project).resolve()
        self.rollout = Path(rollout).resolve()
        self.metallib = Path(metallib).resolve()
        candidates = list(self.project.glob("*.numi-window.json"))
        if len(candidates) != 1:
            raise ValueError("Expected one compiled scene in the project directory")
        self.scene = candidates[0]

    @classmethod
    def compile(
        cls, urdf: Path, project: Path, *, app: Path, srdf: Path | None = None,
        gravity: float | None = None, friction: float = 0.8, effort: float = 0.1,
    ) -> Experiment:
        """Cook a fixed-base URDF or supported MJCF XML into an empty project."""
        contents = Path(app).resolve() / "Contents"
        metallib = contents / "Resources" / "MetalRobo.metallib"
        source = Path(urdf).resolve()
        arguments = [
            contents / "Helpers" / "metalrobo_numi_window_catalog_compile",
            "--mjcf" if source.suffix.lower() == ".xml" else "--urdf", str(source),
            "--output", str(Path(project).resolve()), "--metallib", str(metallib),
            "--friction", str(friction), "--effort", str(effort),
        ]
        if gravity is not None:
            arguments += ["--gravity", str(gravity)]
        if srdf is not None:
            arguments += ["--srdf", str(Path(srdf).resolve())]
        subprocess.run(arguments, check=True, capture_output=True, text=True)
        return cls(project, rollout=contents / "MacOS" / "metalrobo_task_rollout", metallib=metallib)

    def run(
        self, *, steps: int = 200, environments: int = 1, chunk: int = 1,
        physics_substeps: int | None = None, seed: int = 0,
        actions: Iterable[Sequence[float]] | None = None,
        state_trace: Path | None = None, contact_trace: Path | None = None,
    ) -> ExperimentResult:
        """Run the stored controller or broadcast explicit normalized actions.

        Explicit rows contain one value per project actuator. The native runner
        holds the final action if the rollout outlasts the provided sequence.
        """
        arguments = [
            self.rollout, "--scene-file", str(self.scene), "--headless",
            "--metallib", str(self.metallib), "--steps", str(steps),
            "--envs", str(environments), "--chunk", str(chunk),
            "--seed", str(seed),
        ]
        if physics_substeps is not None:
            arguments += ["--physics-substeps", str(physics_substeps)]
        if state_trace is not None:
            arguments += ["--state-trace", str(state_trace)]
        if contact_trace is not None:
            arguments += ["--contact-trace", str(contact_trace)]
        with tempfile.TemporaryDirectory(prefix="numi-actions-") as temporary:
            if actions is not None:
                clip = Path(temporary) / "actions.f32"
                width = None
                with clip.open("wb") as target:
                    for row in actions:
                        if len(row) == 0 or any(not math.isfinite(x) or abs(x) > 1 for x in row):
                            raise ValueError("Action rows require finite values in [-1, 1]")
                        if width is not None and len(row) != width:
                            raise ValueError("Action rows must have the same width")
                        width = len(row)
                        target.write(struct.pack(f"<{width}f", *row))
                if width is None:
                    raise ValueError("Provide at least one action row")
                expected = json.loads((self.project / "project.json").read_text()).get("action_count")
                if expected is not None and expected != width:
                    raise ValueError("Action rows must contain one value per project actuator")
                arguments += ["--action-clip", str(clip)]
            result = subprocess.run(arguments, check=True, capture_output=True, text=True)
        return ExperimentResult(json.loads(result.stdout), state_trace, contact_trace)
