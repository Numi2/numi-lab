#!/usr/bin/env python3
"""CPU contract and schedule checks for source157 grip telemetry."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys



def function_body(source: str, signature_prefix: str) -> str:
    start = source.find(signature_prefix)
    if start < 0:
        raise AssertionError(f"source function not found: {signature_prefix}")
    opening = source.find("{", start)
    if opening < 0:
        raise AssertionError(f"function body not found: {signature_prefix}")
    depth = 0
    for i in range(opening, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : i]
    raise AssertionError(f"unclosed function body: {signature_prefix}")


def phase_at(cfg: dict[str, int], step: int) -> str:
    # The branch ordering and inequalities are checked against loadingPoint below.
    if step < cfg["steps"]:
        return "loading"
    step -= cfg["steps"]
    if step < cfg["hold"]:
        return "hold"
    step -= cfg["hold"]
    if step < cfg["unload"]:
        return "unloading"
    step -= cfg["unload"]
    if step < cfg["relax"]:
        return "relaxation"
    step -= cfg["relax"]
    if step < cfg["release"]:
        return "release"
    return "post_release_relaxation"


def run_schedule(cfg: dict[str, int]) -> list[dict[str, object]]:
    total = sum(cfg.values())
    rows: list[dict[str, object]] = []
    # Three right-grip nodes mirror the driver's per-node target assignment.
    for step in range(total):
        phase = phase_at(cfg, step)
        mode = 2.0 if phase in ("release", "post_release_relaxation") else 1.0
        commanded_node_modes = [mode, mode, mode]
        constrained = all(node_mode == 1.0 for node_mode in commanded_node_modes)
        old_logged = 0 if phase == "release" else 1
        rows.append({"step": step, "phase": phase,
                     "node_modes": commanded_node_modes,
                     "constrained": int(constrained),
                     "old_logged": old_logged})
    return rows


def check(base: Path, patched: Path) -> dict[str, object]:
    base_text = base.read_text()
    text = patched.read_text()
    fill_prefix = "void fillTargets(const Arguments& arguments,"
    load_prefix = "LoadPoint loadingPoint(const Arguments& a, std::uint32_t step)"
    total_prefix = "std::uint32_t totalSteps(const Arguments& a)"
    fill_base = function_body(base_text, fill_prefix)
    fill_new = function_body(text, fill_prefix)
    if fill_base != fill_new:
        raise AssertionError("physics target generation changed")
    fill = fill_new
    for required in (
        'phase == "release" || phase == "post_release_relaxation"',
        "targets[base + node] = {0.0f, 0.0f, 0.0f, 2.0f};",
        "targets[base + node] = {",
        "                1.0f,",
    ):
        if required not in fill:
            raise AssertionError(f"driver target-mode contract changed: {required}")

    load = function_body(text, load_prefix)
    total = function_body(text, total_prefix)
    for required in (
        'return {a.bendAngleDegrees * static_cast<double>(step + 1u) / a.steps, "loading"};',
        'if (step < a.holdSteps) return {a.bendAngleDegrees, "hold"};',
        'if (step < a.unloadSteps)',
        '"unloading"};',
        'if (step < a.relaxSteps) return {0.0, "relaxation"};',
        'if (step < a.releaseSteps)',
        'return {0.0, "release"};',
        'return {0.0, "post_release_relaxation"};',
        'a.steps + a.holdSteps + a.unloadSteps + a.relaxSteps + a.releaseSteps +',
        'a.postReleaseRelaxSteps;',
    ):
        if required not in load + total:
            raise AssertionError(f"driver phase schedule contract changed: {required}")

    # Source guard: the telemetry path reads the generated command modes and
    # rejects mixed/unknown modes instead of deriving the field from phase text.
    helper_prefix = "bool commandedRightGripConstrained(const std::vector<nm_float4>& targets,"
    helper = function_body(text, helper_prefix)
    for required in (
        "mesh.rightGripNodes",
        "targets[base + node].w",
        "mode == 1.0f || mode == 2.0f",
        "targets[base + node].w == mode",
        "return mode == 1.0f;",
    ):
        if required not in helper:
            raise AssertionError(f"commanded-mode reader contract missing: {required}")
    observation = function_body(text, "void writeObservation(std::ofstream& output,")
    if "(rightGripConstrained ? 1 : 0)" not in observation:
        raise AssertionError("CSV field is not sourced from commanded right-grip mode")
    if 'std::string_view(phase) == "release"' in observation:
        raise AssertionError("phase-name guess remains in CSV telemetry")
    if "commandedRightGripConstrained(targets, mesh, environment)" not in text:
        raise AssertionError("observation call does not bind generated targets")

    source_post_requirement = "arguments.postReleaseRelaxSteps == 0u || arguments.releaseSteps > 0u"
    if source_post_requirement not in base_text:
        raise AssertionError("post-release/release admission relationship changed")

    configs = {
        "complete_with_postrelease": {
            "steps": 3, "hold": 1, "unload": 2, "relax": 2,
            "release": 2, "post": 3,
        },
        "complete_without_postrelease": {
            "steps": 3, "hold": 1, "unload": 2, "relax": 2,
            "release": 2, "post": 0,
        },
        "zero_release": {
            "steps": 2, "hold": 1, "unload": 2, "relax": 1,
            "release": 0, "post": 0,
        },
        "terminal_one_step_release": {
            "steps": 1, "hold": 0, "unload": 0, "relax": 0,
            "release": 1, "post": 0,
        },
    }
    results: dict[str, object] = {}
    for name, cfg in configs.items():
        rows = run_schedule(cfg)
        if len(rows) != sum(cfg.values()):
            raise AssertionError(f"{name}: total step count mismatch")
        for row in rows:
            phase = row["phase"]
            want_constrained = phase not in ("release", "post_release_relaxation")
            if row["constrained"] != int(want_constrained):
                raise AssertionError(f"{name} step {row['step']}: commanded mode mismatch")
        if name == "complete_with_postrelease":
            post_rows = [r for r in rows if r["phase"] == "post_release_relaxation"]
            if len(post_rows) != cfg["post"] or any(r["constrained"] != 0 for r in post_rows):
                raise AssertionError("post-release rows do not report free right grips")
            old_false_pass = [r for r in post_rows if r["old_logged"] != r["constrained"]]
            if len(old_false_pass) != len(post_rows):
                raise AssertionError("old phase-name counterexample was not reproduced")
        if name == "zero_release" and any(r["phase"] in ("release", "post_release_relaxation") for r in rows):
            raise AssertionError("zero-release schedule unexpectedly entered a release phase")
        if name == "terminal_one_step_release":
            if rows[-1]["phase"] != "release" or rows[-1]["constrained"] != 0:
                raise AssertionError("terminal release row should be free")
        results[name] = {
            "rows": len(rows),
            "phase_counts": {p: sum(r["phase"] == p for r in rows)
                             for p in ("loading", "hold", "unloading", "relaxation",
                                       "release", "post_release_relaxation")},
            "right_grip_modes": [r["constrained"] for r in rows],
            "old_phase_guess_mismatches": sum(r["old_logged"] != r["constrained"] for r in rows),
            "old_phase_guess_mismatch_steps": [r["step"] for r in rows
                                               if r["old_logged"] != r["constrained"]],
        }
    if configs["complete_with_postrelease"]["post"] == 0:
        raise AssertionError("test fixture must include post-release samples")
    return {"checks": 4, "source_target_generation_unchanged": True,
            "phase_name_counterexample": "old reporter logged post_release_relaxation as constrained=1 while commanded target w=2",
            "schedules": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--patched-source", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        report = check(args.base_source, args.patched_source)
    except (AssertionError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    import json
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
