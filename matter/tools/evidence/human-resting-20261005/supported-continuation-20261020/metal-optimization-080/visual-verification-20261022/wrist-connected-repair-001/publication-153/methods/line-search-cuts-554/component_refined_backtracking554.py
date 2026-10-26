#!/usr/bin/env python3
"""Bounded exact-gated alpha refinement over frozen component owner 549.

Every interpolated Float32 pose is passed through the owner's unchanged gate,
including exact self-pair gate 315 when --exact-self is supplied. This records
a finite local search schedule only; it makes no monotonicity or maximum-alpha
claim. The frozen 549 source is imported and patched in memory, never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT = Path("/Users/numi/Work/numi-human-connected-surface-069")
OWNER549 = PROJECT / "D/component-area-backtracking-549/component_area_backtracking549.py"
OWNER549_SHA = "523b1aef89d92eef09bf63e9de2d4f3fa6bd938ef53a45915f9ea138f5b8f249"
SEARCH_CLAIM = "bounded_actual_gate_probes_only_no_monotonicity_or_global_maximum_claim"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def emit_component(base: np.ndarray, base_mm: np.ndarray, delta_mm: np.ndarray,
                   nodes: np.ndarray, alpha: float) -> np.ndarray:
    out = base.copy()
    g = np.asarray(nodes, dtype=np.int64)
    if alpha:
        out[g] = ((base_mm[g] + alpha * delta_mm[g]) / 1000.0).astype(np.float32)
    return out


def component_line_search(base: np.ndarray, target: np.ndarray, free: np.ndarray,
                          groups: list[np.ndarray], gate: Callable[[np.ndarray], tuple[bool, dict]],
                          max_halvings: int = 32, refine_steps: int = 6) -> tuple[np.ndarray, list[dict]]:
    """Frozen 549 power-of-two pass, then bounded exact-gate bracket probes."""
    if base.dtype != np.float32 or target.dtype != np.float32 or target.shape != base.shape:
        raise ValueError("points must be matching Float32 arrays")
    if base.ndim != 2 or base.shape[1] != 3 or not np.isfinite(base).all() or not np.isfinite(target).all():
        raise ValueError("points must be finite Float32 (N,3)")
    if not isinstance(refine_steps, int) or not 0 <= refine_steps <= 32:
        raise ValueError("refine_steps must be an integer in [0,32]")
    if not gate(base)[0]:
        raise ValueError("baseline does not pass gates")
    if sorted(np.concatenate(groups).tolist()) != sorted(np.asarray(free).tolist()):
        raise ValueError("component partition mismatch")

    x = base.copy()
    base_mm = base.astype(np.float64) * 1000.0
    delta_mm = (target.astype(np.float64) - base.astype(np.float64)) * 1000.0
    records = []

    for ci, group in enumerate(groups):
        g = np.asarray(group, dtype=np.int64)
        attempts = []
        refinement = {
            "requested_steps": refine_steps,
            "attempted": False,
            "initial_bracket": None,
            "trials": [],
            "search_claim": SEARCH_CLAIM,
        }
        selected = None
        previous_failed_alpha = None

        # Preserve 549's original descending power-of-two search and emission.
        for k in range(max_halvings + 1):
            alpha = 2.0 ** (-k)
            trial = emit_component(x, base_mm, delta_mm, g, alpha)
            if same(trial[g], x[g]):
                attempts.append({
                    "alpha": alpha,
                    "pass": False,
                    "phase": "power2",
                    "skipped": "Float32 emission identical to current component",
                })
                break
            ok, details = gate(trial)
            attempts.append({"alpha": alpha, "pass": bool(ok), "phase": "power2", **details})
            if ok:
                x = trial
                selected = alpha
                # Refine only if an immediately preceding larger recorded
                # power-of-two trial actually failed.
                if previous_failed_alpha is not None and refine_steps:
                    lo = alpha
                    hi = previous_failed_alpha
                    refinement["attempted"] = True
                    refinement["initial_bracket"] = {
                        "passing_alpha": lo,
                        "failed_alpha": hi,
                        "bracket_width": hi - lo,
                    }
                    for ri in range(refine_steps):
                        mid = (lo + hi) * 0.5
                        candidate = emit_component(x, base_mm, delta_mm, g, mid)
                        trial_record = {
                            "index": ri,
                            "bracket_before": {"passing_alpha": lo, "failed_alpha": hi},
                            "alpha": mid,
                        }
                        if same(candidate[g], x[g]):
                            trial_record.update({
                                "pass": False,
                                "skipped": "Float32 emission identical to current accepted component",
                            })
                            refinement["trials"].append(trial_record)
                            break
                        refined_ok, refined_details = gate(candidate)
                        trial_record.update({"pass": bool(refined_ok), **refined_details})
                        refinement["trials"].append(trial_record)
                        # A failed test never changes x/cache. A pass is retained
                        # immediately; the next probe is then based on this actual
                        # accepted Float32 point and gate-315 cache state.
                        if refined_ok:
                            x = candidate
                            selected = mid
                            lo = mid
                        else:
                            hi = mid
                break
            previous_failed_alpha = alpha
        records.append({
            "component": ci,
            "nodes": g.tolist(),
            "selected_alpha": selected,
            "trials": attempts,
            "alpha_refinement_554": refinement,
        })

    outside = np.setdiff1d(np.arange(len(base)), np.asarray(free))
    if not same(x[outside], base[outside]):
        raise AssertionError("outside rows changed")
    return x, records


def import_owner(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_component549_alpha_refinement554", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen 549 owner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine-steps", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args, owner_argv = parser.parse_known_args()
    if not 0 <= args.refine_steps <= 32:
        raise ValueError("--refine-steps must be in [0,32]")
    if "--exact-self" not in owner_argv:
        raise ValueError("this wrapper requires --exact-self so every refined trial uses gate 315")
    if not OWNER549.is_file() or sha(OWNER549) != OWNER549_SHA:
        raise ValueError("frozen 549 owner pin mismatch")

    owner = import_owner(OWNER549)
    owner.line_search = lambda base, target, free, groups, gate, max_halvings=32: component_line_search(
        base, target, free, groups, gate, max_halvings=max_halvings,
        refine_steps=args.refine_steps,
    )
    sys.argv = [str(OWNER549), *owner_argv, "--output", str(args.output)]
    owner.main()

    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text())
    report["source_pins"][str(Path(__file__).resolve())] = sha(Path(__file__))
    report["alpha_refinement_554"] = {
        "owner": str(OWNER549),
        "owner_sha256": OWNER549_SHA,
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": sha(Path(__file__)),
        "refine_steps_requested": args.refine_steps,
        "gate": "unchanged 549 gate; exact 315 required by wrapper CLI",
        "search_claim": SEARCH_CLAIM,
        "per_component": [
            {
                "component": row["component"],
                "selected_alpha": row["selected_alpha"],
                **row["alpha_refinement_554"],
            }
            for row in report["components"]
        ],
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(report_path)
    print(json.dumps({
        "candidate": report.get("candidate_path"),
        "components": len(report.get("components", [])),
        "selected_alphas": [r.get("selected_alpha") for r in report.get("components", [])],
        "refine_steps": args.refine_steps,
        "search_claim": SEARCH_CLAIM,
        "native_admission": False,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

