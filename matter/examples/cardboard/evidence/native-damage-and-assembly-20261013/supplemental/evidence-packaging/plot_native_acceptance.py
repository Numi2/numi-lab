#!/usr/bin/env python3
"""Plot accepted native steps for retained cardboard solver trials.

This is an outcome-coverage figure only. It does not compare endpoint error,
physical accuracy, or mesh convergence. Every input path and SHA-256 is recorded
in the generated JSON sidecar.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("/Users/home/cardboard-followup-20261013")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_plan(argv: list[str]) -> int | None:
    names = ("--steps", "--hold-steps", "--unload-steps", "--relax-steps")
    values: dict[str, int] = {}
    for i, arg in enumerate(argv[:-1]):
        if arg in names:
            try:
                values[arg] = int(argv[i + 1])
            except ValueError:
                pass
    return sum(values.values()) if values else None


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def accepted_from_csv(path: Path, expected_arm: str) -> int | None:
    if not path.is_file():
        return None
    accepted: set[int] = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("arm") != expected_arm or row.get("environment") != "0":
                continue
            if row.get("step_accepted") == "1":
                try:
                    accepted.add(int(row["step"]))
                except (KeyError, ValueError):
                    continue
    return len(accepted)


def case_record(root: Path, relative: str, label: str, arm: str,
                *, scope: str, nx: int | None = None) -> dict:
    case = root / relative
    prediction_path = case / "prediction.json"
    receipt_path = case / "receipt.json"
    result_path = case / "output" / "result.json"
    csv_path = case / "output" / "observations.csv"
    prediction = read_json(prediction_path)
    receipt = read_json(receipt_path)
    result = read_json(result_path)
    argv = prediction.get("argv", []) if prediction else []
    planned = parse_plan(argv)
    accepted_csv = accepted_from_csv(csv_path, arm)
    accepted = result.get("accepted_steps") if result else accepted_csv
    if accepted is None:
        accepted = accepted_csv
    if receipt and receipt.get("timed_out"):
        state = "TIMEOUT"
    elif not prediction or not receipt:
        state = "UNATTEMPTED"
    elif (result and result.get("status") in {"complete", "completed"}
          and receipt.get("exit_code") == 0 and planned is not None
          and accepted == planned):
        state = "COMPLETE"
    elif result or accepted is not None:
        state = "FAILED"
    else:
        state = "INCOMPLETE"
    bindings = {}
    for name, path in (("prediction", prediction_path), ("receipt", receipt_path),
                       ("result", result_path), ("observations", csv_path)):
        if path.is_file():
            bindings[name] = {"path": str(path), "sha256": sha256(path)}
    return {
        "case": relative,
        "label": label,
        "nx": nx,
        "arm": arm,
        "scope": scope,
        "state": state,
        "accepted_steps": accepted,
        "accepted_steps_from_csv": accepted_csv,
        "planned_steps": planned,
        "receipt_exit_code": receipt.get("exit_code") if receipt else None,
        "timed_out": receipt.get("timed_out") if receipt else None,
        "elapsed_seconds": receipt.get("wall_seconds") if receipt else None,
        "failure": result.get("failure") if result else None,
        "compiled_world_fingerprint": result.get("compiled_world_fingerprint") if result else None,
        "bindings": bindings,
    }


def collect(root: Path) -> dict:
    cases: list[dict] = []
    for protocol in ("mesh-armijo-003", "mesh-armijo-004"):
        for nx in (8, 16, 32):
            rel = f"solver/{protocol}/nx{nx}"
            cases.append(case_record(root, rel, f"{protocol[-3:]} nx={nx}", "bent",
                                     scope="numerical path coverage only", nx=nx))
    for version in ("001", "002", "003"):
        rel = f"scoring/armijo-{version}"
        if version == "001":
            scope = "SCOPE_NOT_QUALIFIED: frozen027 selector bypass"
        else:
            scope = "frozen028 repaired endpoint gate; native diagnostics only"
        cases.append(case_record(root, rel, version, "indented", scope=scope))
    return {
        "schema": "numi.cardboard.native-acceptance-coverage.v1",
        "interpretation": "Accepted-step counts only. No endpoint convergence or physical qualification is implied.",
        "cases": cases,
    }


def bar_style(record: dict) -> tuple[str, str]:
    if record["state"] == "UNATTEMPTED":
        return "#c9c9c9", ".."
    if record["state"] == "TIMEOUT":
        return "#d2a14d", "//"
    if record["case"].startswith("solver/mesh-armijo-003"):
        return "#527aa3", ""
    if record["case"].startswith("solver/mesh-armijo-004"):
        return "#5b9077", ""
    if record["case"].endswith("armijo-001"):
        return "#777777", ""
    if record["case"].endswith("armijo-002"):
        return "#5b9077", ""
    if record["case"].endswith("armijo-003"):
        return "#527aa3", ""
    return "#999999", ""


def draw(data: dict, png: Path) -> None:
    fig, (mesh_ax, score_ax) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.2, 1]})
    mesh_cases = [c for c in data["cases"] if c["case"].startswith("solver/mesh-")]
    x = np.arange(3)
    width = 0.34
    for offset, protocol in ((-width / 2, "mesh-armijo-003"), (width / 2, "mesh-armijo-004")):
        records = {c["nx"]: c for c in mesh_cases if protocol in c["case"]}
        for i, nx in enumerate((8, 16, 32)):
            rec = records.get(nx)
            if not rec:
                continue
            planned = rec["planned_steps"] or 96
            accepted = rec["accepted_steps"] or 0
            color, hatch = bar_style(rec)
            mesh_ax.bar(i + offset, accepted / planned, width, color=color,
                        edgecolor="#333333", linewidth=.65, hatch=hatch,
                        label=protocol if i == 0 else None)
            if rec["state"] == "UNATTEMPTED":
                annotation = "unattempted"
            elif rec["state"] == "TIMEOUT":
                annotation = f"{accepted}/{planned}\ntimeout"
            else:
                annotation = f"{accepted}/{planned}\n{rec['state'].lower()}"
            mesh_ax.text(i + offset, max(accepted / planned, 0.015) + .035, annotation,
                         ha="center", va="bottom", fontsize=7.5, rotation=0)
    mesh_ax.set_xticks(x, ["nx=8", "nx=16", "nx=32"])
    mesh_ax.set_ylim(0, 1.25)
    mesh_ax.set_ylabel("Accepted steps / requested steps")
    mesh_ax.set_title("Mesh trial coverage")
    mesh_ax.legend(title="Path", frameon=False, fontsize=8)
    mesh_ax.grid(axis="y", alpha=.2)

    score_cases = [c for c in data["cases"] if c["case"].startswith("scoring/")]
    xscore = np.arange(len(score_cases))
    for i, rec in enumerate(score_cases):
        planned = rec["planned_steps"] or 96
        accepted = rec["accepted_steps"] or 0
        color, hatch = bar_style(rec)
        score_ax.bar(i, accepted / planned, .64, color=color, edgecolor="#333333",
                     linewidth=.65, hatch=hatch)
        status = rec["state"].lower()
        if rec["case"].endswith("armijo-001"):
            status += "\nSCOPE_NOT_QUALIFIED"
        elif rec["state"] == "UNATTEMPTED":
            status = "unattempted"
        score_ax.text(i, max(accepted / planned, .015) + .035,
                      f"{accepted}/{planned}\n{status}", ha="center", va="bottom", fontsize=7.5)
    score_ax.set_xticks(xscore, [c["label"] for c in score_cases])
    score_ax.set_ylim(0, 1.25)
    score_ax.set_title("Punch-and-backing path coverage")
    score_ax.grid(axis="y", alpha=.2)
    fig.suptitle("Native accepted-step coverage (not endpoint convergence or physical qualification)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/home/cardboard-followup-20261013/evidence-packaging"))
    args = parser.parse_args()
    data = collect(args.root)
    json_path = args.output_dir / "native-acceptance-coverage.json"
    png_path = args.output_dir / "native-acceptance-coverage.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    draw(data, png_path)
    print(json.dumps({"json": str(json_path), "json_sha256": sha256(json_path),
                      "png": str(png_path), "png_sha256": sha256(png_path)}, indent=2))


if __name__ == "__main__":
    main()
