#!/usr/bin/env python3
"""Render native accepted coupon geometry without interpolation or deformation scaling.

Failed runs require --allow-incomplete and are labeled as incomplete diagnostics.
The animation is a visualization of geometry, never a qualification certificate.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coupon_frames(root, environment, allow_incomplete):
    manifest = json.loads((root / "manifest.json").read_text())
    solver = manifest["solver"]
    expected = sum(int(solver[key]) for key in
                   ("steps", "hold_steps", "unload_steps", "relax_steps", "release_steps"))
    selected = []
    seen = set()
    rejected = []
    with (root / "observations.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["environment"]) != environment:
                continue
            step = int(row["step"])
            if step in seen:
                raise ValueError("duplicate native observation step")
            seen.add(step)
            finite = all(math.isfinite(float(row[key])) for key in
                         ("time_s", "certificate_residual", "certificate_volume",
                          "certificate_pressure", "certificate_raw_accepted_flag"))
            accepted = finite and int(row["status_code"]) == 0 and int(row["step_accepted"]) == 1
            accepted = accepted and float(row["certificate_raw_accepted_flag"]) > 0.5
            for field, tolerance in (("certificate_residual", "relative_residual_tolerance"),
                                     ("certificate_volume", "volume_tolerance"),
                                     ("certificate_pressure", "pressure_tolerance")):
                accepted = accepted and float(row[field]) <= float(solver[tolerance])
            if not accepted:
                rejected.append(step)
                continue
            path = root / (f"accepted_step_{step+1:06d}_{row['arm']}.obj")
            if not path.is_file():
                raise ValueError(f"accepted geometry is missing: {path.name}")
            selected.append(dict(step=step, time_s=float(row["time_s"]),
                                 stage=row["phase"], path=path, sha256=digest(path)))
    selected.sort(key=lambda row: row["step"])
    complete = not rejected and [row["step"] for row in selected] == list(range(expected))
    summary_path = root / "result.json"
    # Probe revisions used both result.json and summary.json. The accepted full
    # CSV coverage is required even when a summary declares success.
    if not summary_path.exists():
        summary_path = root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        complete = complete and summary.get("status") in ("passed", "success", "completed")
    else:
        complete = False
    if not complete and not allow_incomplete:
        raise ValueError("native history is incomplete or rejected; --allow-incomplete is required")
    if not selected:
        raise ValueError("no certificate-accepted geometry to render")
    return selected, dict(complete=complete, expected_steps=expected,
                          accepted_steps=len(selected), rejected_steps=rejected,
                          manifest_sha256=digest(root / "manifest.json"),
                          observations_sha256=digest(root / "observations.csv"))


def read_obj(path, maximum_faces):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v":
            if len(parts) != 4:
                raise ValueError("unsupported OBJ vertex")
            point = [float(value) for value in parts[1:]]
            if not all(math.isfinite(value) for value in point):
                raise ValueError("nonfinite OBJ vertex")
            vertices.append(point)
        elif parts[0] == "f":
            if len(parts) != 4 or any("/" in value for value in parts[1:]):
                raise ValueError("only native triangular OBJ faces are supported")
            faces.append([int(value)-1 for value in parts[1:]])
            if len(faces) > maximum_faces:
                raise ValueError("surface face budget exceeded; geometry will not be silently decimated")
        else:
            raise ValueError("unsupported OBJ record")
    if not vertices or not faces or any(index < 0 or index >= len(vertices)
                                         for face in faces for index in face):
        raise ValueError("empty or invalid OBJ topology")
    return vertices, faces


def render(root, destination, environment=0, allow_incomplete=False, stride=1,
           fps=12, maximum_faces=100000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    if destination.exists():
        raise ValueError("animation output already exists")
    frames, evidence = coupon_frames(root, environment, allow_incomplete)
    shown = frames[::stride]
    if shown[-1] is not frames[-1]:
        shown.append(frames[-1])
    bounds_min, bounds_max, topology = None, None, None
    for frame in shown:
        vertices, faces = read_obj(frame["path"], maximum_faces)
        vertices = np.asarray(vertices)
        if topology is None:
            topology = faces
        elif faces != topology:
            raise ValueError("changing topology requires a native remeshing-aware frame index")
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        bounds_min = low if bounds_min is None else np.minimum(bounds_min, low)
        bounds_max = high if bounds_max is None else np.maximum(bounds_max, high)
    center = (bounds_min+bounds_max)/2
    span = max(float((bounds_max-bounds_min).max()), 1e-9)*1.1
    images = []
    for frame in shown:
        vertices, faces = read_obj(frame["path"], maximum_faces)
        triangles = np.asarray(vertices)[np.asarray(faces)]*1000
        figure = plt.figure(figsize=(7.2, 5.4), dpi=100)
        axes = figure.add_subplot(111, projection="3d")
        axes.add_collection3d(Poly3DCollection(triangles, facecolor="#bd9868",
                              edgecolor="#735336", linewidth=0.12, alpha=1))
        for axis, index in ((axes.set_xlim, 0), (axes.set_ylim, 1), (axes.set_zlim, 2)):
            axis((center[index]-span/2)*1000, (center[index]+span/2)*1000)
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elev=22, azim=-58)
        axes.set_xlabel("x (mm)"); axes.set_ylabel("y (mm)"); axes.set_zlabel("z (mm)")
        label = "Complete native history" if evidence["complete"] else "INCOMPLETE NATIVE DIAGNOSTIC"
        axes.set_title(f"{label}\n{frame['stage']} · t={frame['time_s']:.6g} s · step {frame['step']+1}")
        figure.tight_layout()
        figure.canvas.draw()
        images.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert("RGB"))
        plt.close(figure)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination, save_all=True, append_images=images[1:],
                   duration=round(1000/fps), loop=0)
    provenance = dict(schema="numi.cardboard.accepted_geometry_animation.v1", **evidence,
                      animation_sha256=digest(destination), environment=environment,
                      displayed_frames=len(shown), frame_stride=stride, playback_fps=fps,
                      geometric_interpolation=False, deformation_scale=1,
                      fixed_camera=True, topology_decimation=False,
                      physical_validation=False,
                      frames=[{key: str(value) if key == "path" else value
                               for key, value in frame.items()} for frame in shown])
    destination.with_suffix(destination.suffix+".json").write_text(json.dumps(provenance, indent=2)+"\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--maximum-faces", type=int, default=100000)
    args = parser.parse_args()
    if min(args.stride, args.fps, args.maximum_faces) <= 0 or args.environment < 0:
        parser.error("positive render budgets and a nonnegative environment are required")
    result = render(args.run, args.output, args.environment, args.allow_incomplete,
                    args.stride, args.fps, args.maximum_faces)
    print(json.dumps({key: result[key] for key in ("complete", "accepted_steps", "displayed_frames")}))


if __name__ == "__main__":
    main()
