#!/usr/bin/env python3
"""Full captured-center 9-basis FP64 FD audit for the source173 DD reference.

No native or physical execution is performed. Inputs are hash-pinned and all
per-row endpoint/gate/branch results are retained in a compressed NPZ.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np


STRICT_RESIDUAL = 2.0e-12
STRICT_DYNAMIC_CORRECTION = 2.0e-11
DERIVATIVE_RELATIVE_L2 = 1.0e-6
EPSILON_POWERS = (8, 10, 12, 14, 16, 18, 20, 22, 24, 26)
SOURCE173_SHA256 = "a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995"
CAPTURE_SHA256 = "4361b6aa9f0d269a9733b0bd8f508dbc1204624451611816512dad02c303d27c"
MP_ARRAYS_SHA256 = "107e942cc2e131eed8f8db7759f8deb3e86fcdc42d8da249996daf51d6ed11ad"
SCALE = np.array([1.0e4] * 6 + [1.0] * 6 + [1.0e4] * 3, dtype=np.float64)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import equation base: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def projection(base, ref, p, f, old):
    initial = base.project_centers(ref, p, f, old, p[:, 26])
    return base.continue_projection_tight(ref, p, f, old, p[:, 26], initial)


def signature(v2_module, p, old, z):
    _, parts, meta = v2_module.fb_row(p, old, z)
    dl, y = parts[2], parts[3]
    return np.stack((np.sign(dl.rounded()), np.sign(y.rounded()),
                      meta["fb_quotient_branch"].astype(np.float64)), axis=1)


def tangent_matrix(ref, base, p, f, old, z):
    out = np.empty((len(p), 9, 9), dtype=np.float64)
    for col in range(9):
        h = np.zeros_like(f)
        h.reshape(len(p), 9)[:, col] = 1.0
        out[:, :, col] = ref.tangent(base, p, f, old, z, h).reshape(len(p), 9)
    return out


def relative_l2(observed, expected):
    diff = np.linalg.norm(observed - expected, axis=-1)
    refnorm = np.linalg.norm(expected, axis=-1)
    rel = np.full(refnorm.shape, np.nan, dtype=np.float64)
    np.divide(diff, refnorm, out=rel, where=np.isfinite(diff) & (refnorm > 0.0))
    return rel, diff, refnorm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-reference", required=True, type=pathlib.Path)
    ap.add_argument("--capture", required=True, type=pathlib.Path)
    ap.add_argument("--mp-arrays", required=True, type=pathlib.Path)
    ap.add_argument("--portable-tools", required=True, type=pathlib.Path,
                    help="r-ready/matter/tools folder containing the three portable modules")
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--epsilon-powers", default=",".join(map(str, EPSILON_POWERS)))
    args = ap.parse_args()
    powers = tuple(int(x) for x in args.epsilon_powers.split(","))
    if not powers or any(x < 1 or x > 50 for x in powers) or tuple(sorted(set(powers))) != powers:
        raise SystemExit("epsilon powers must be unique, increasing integers in [1,50]")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if sha256(args.source_reference) != SOURCE173_SHA256:
        raise SystemExit("source173 equation-base hash mismatch")
    if sha256(args.capture) != CAPTURE_SHA256:
        raise SystemExit("native469 capture hash mismatch")
    if sha256(args.mp_arrays) != MP_ARRAYS_SHA256:
        raise SystemExit("independent MP arrays hash mismatch")

    portable = args.portable_tools.resolve()
    sys.path.insert(0, str(portable))
    base_v1 = load_module("cardboard_compensated_fp64_base", portable / "cardboard_compensated_fp64_base.py")
    dd = load_module("cardboard_dd_fp64", portable / "cardboard_dd_fp64.py")
    ref_v2 = load_module("cardboard_compensated_fp64_reference", portable / "cardboard_compensated_fp64_reference.py")
    if not hasattr(ref_v2, "Reference") or not hasattr(ref_v2, "tangent"):
        raise SystemExit("portable v2 API mismatch")
    # Ensure imported portable modules are the files whose contents the caller supplied.
    portable_hashes = {p.name: sha256(p) for p in sorted(portable.glob("cardboard_*.py"))}
    if set(portable_hashes) != {"cardboard_compensated_fp64_base.py", "cardboard_compensated_fp64_reference.py", "cardboard_dd_fp64.py"}:
        raise SystemExit(f"unexpected portable module set: {sorted(portable_hashes)}")
    base = base_v1.import_reference(args.source_reference)
    ref = ref_v2.Reference(base)

    args.output.mkdir(parents=True)
    started = time.monotonic()
    progress_path = args.output / "progress.jsonl"

    def progress(stage, **kw):
        record = {"stage": stage, "elapsed_seconds": time.monotonic() - started, **kw}
        with progress_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()

    progress("input_bound", source_sha256=sha256(args.source_reference),
             capture_sha256=sha256(args.capture), mp_arrays_sha256=sha256(args.mp_arrays),
             epsilon_powers=list(powers), portable_module_hashes=portable_hashes)
    cap = np.load(args.capture)
    mpa = np.load(args.mp_arrays)
    paper_tets = np.flatnonzero(cap["material"] != 2)
    if len(paper_tets) != 1728 or len(np.unique(paper_tets)) != 1728:
        raise SystemExit("captured paper tet set changed")
    if not np.array_equal(mpa["paper_material_index"], cap["material"][paper_tets]):
        raise SystemExit("per-tet material order differs from independent MP binding")
    if not np.array_equal(cap["F_center"], cap["s0_F"]):
        raise SystemExit("captured center differs from s0 input")
    p = mpa["paper_parameters_f32"].astype(np.float64)
    old = cap["accepted_paper_history"].astype(np.float64)
    F0 = cap["F_center"][paper_tets].astype(np.float64)
    if p.shape[0] != 1728 or p.shape[1] < 27 or old.shape != (1728, 15) or F0.shape != (1728, 3, 3):
        raise SystemExit("captured paper input shape mismatch")
    if not np.isfinite(p).all() or not np.isfinite(old).all() or not np.isfinite(F0).all():
        raise SystemExit("nonfinite captured inputs")

    center = projection(base, ref, p, F0, old)
    z0, it0, r0, c0, ok0, why0, refinements0 = center
    center_failure = np.flatnonzero(~ok0)
    if center_failure.size:
        # Preserve the exact failed centers rather than stopping without evidence.
        np.savez_compressed(args.output / "center-failures.npz", rows=center_failure,
                            tets=paper_tets[center_failure], state=z0[center_failure],
                            iterations=it0[center_failure], residual=r0[center_failure],
                            correction=c0[center_failure], reasons=np.asarray(why0, dtype="U256"))
    np.savez_compressed(args.output / "center-projection.npz", paper_tets=paper_tets,
                        F=F0, parameters=p, old=old, state=z0, iterations=it0,
                        residual=r0, correction=c0, valid=ok0,
                        reasons=np.asarray(why0, dtype="U256"))
    progress("center_projection", strict_passes=int(ok0.sum()), strict_failures=int((~ok0).sum()),
             max_residual=float(np.nanmax(r0)), max_correction=float(np.nanmax(c0)))
    if not ok0.all():
        raise SystemExit("center strict projection has failures; endpoints were not evaluated")

    center_stress = base.stress(p, F0, z0).reshape(1728, 9)
    center_sig = signature(ref_v2, p, old, z0)
    C = tangent_matrix(ref_v2, base, p, F0, old, z0)
    if not np.isfinite(center_stress).all() or not np.isfinite(C).all():
        raise SystemExit("nonfinite center stress or implicit tangent")
    np.savez_compressed(args.output / "center-implicit-tangent.npz", paper_tets=paper_tets,
                        stress=center_stress, tangent=C, branch=center_sig,
                        state=z0, residual=r0, correction=c0, iterations=it0)
    progress("center_tangent", rows=len(paper_tets), directions=9)

    ndir, neps, nrow = 9, len(powers), len(paper_tets)
    shape = (ndir, neps, nrow)
    valid_plus = np.zeros(shape, dtype=bool)
    valid_minus = np.zeros(shape, dtype=bool)
    same_plus = np.zeros(shape, dtype=bool)
    same_minus = np.zeros(shape, dtype=bool)
    centered_rel = np.full(shape, np.nan)
    centered_abs = np.full(shape, np.nan)
    ref_norm = np.full(shape, np.nan)
    plus_rel = np.full(shape, np.nan)
    minus_rel = np.full(shape, np.nan)
    delta_plus = np.zeros(shape)
    delta_minus = np.zeros(shape)
    plus_residual = np.full(shape, np.nan)
    minus_residual = np.full(shape, np.nan)
    plus_correction = np.full(shape, np.nan)
    minus_correction = np.full(shape, np.nan)
    plus_iterations = np.zeros(shape, dtype=np.int32)
    minus_iterations = np.zeros(shape, dtype=np.int32)
    plus_reason = np.full(shape, "", dtype="U256")
    minus_reason = np.full(shape, "", dtype="U256")
    plus_branch = np.full(shape + (3,), 99.0)
    minus_branch = np.full(shape + (3,), 99.0)

    for direction in range(9):
        i, j = divmod(direction, 3)
        for ep_index, power in enumerate(powers):
            eps = math.ldexp(1.0, -power)
            fp = F0.copy(); fm = F0.copy()
            fp[:, i, j] += eps
            fm[:, i, j] -= eps
            realized_p = fp[:, i, j] - F0[:, i, j]
            realized_m = F0[:, i, j] - fm[:, i, j]
            delta_plus[direction, ep_index] = realized_p
            delta_minus[direction, ep_index] = realized_m
            Fs = np.concatenate((fp, fm), axis=0)
            ps = np.concatenate((p, p), axis=0)
            olds = np.concatenate((old, old), axis=0)
            states = projection(base, ref, ps, Fs, olds)
            zs, its, rs, cs, oks, whys, _ = states
            zp, zm = zs[:nrow], zs[nrow:]
            ip, im = its[:nrow], its[nrow:]
            rp, rm = rs[:nrow], rs[nrow:]
            cp, cm = cs[:nrow], cs[nrow:]
            op, om = oks[:nrow], oks[nrow:]
            plus_reason[direction, ep_index] = np.asarray(whys[:nrow], dtype="U256")
            minus_reason[direction, ep_index] = np.asarray(whys[nrow:], dtype="U256")
            plus_iterations[direction, ep_index] = ip
            minus_iterations[direction, ep_index] = im
            plus_residual[direction, ep_index] = rp
            minus_residual[direction, ep_index] = rm
            plus_correction[direction, ep_index] = cp
            minus_correction[direction, ep_index] = cm
            valid_plus[direction, ep_index] = op & np.isfinite(zp).all(1)
            valid_minus[direction, ep_index] = om & np.isfinite(zm).all(1)
            if valid_plus[direction, ep_index].any():
                pids = np.flatnonzero(valid_plus[direction, ep_index])
                plus_branch[direction, ep_index, pids] = signature(ref_v2, p[pids], old[pids], zp[pids])
            if valid_minus[direction, ep_index].any():
                mids = np.flatnonzero(valid_minus[direction, ep_index])
                minus_branch[direction, ep_index, mids] = signature(ref_v2, p[mids], old[mids], zm[mids])
            same_plus[direction, ep_index] = valid_plus[direction, ep_index] & np.all(
                plus_branch[direction, ep_index] == center_sig, axis=1)
            same_minus[direction, ep_index] = valid_minus[direction, ep_index] & np.all(
                minus_branch[direction, ep_index] == center_sig, axis=1)
            if valid_plus[direction, ep_index].any():
                ids = np.flatnonzero(valid_plus[direction, ep_index])
                sp = base.stress(p[ids], fp[ids], zp[ids]).reshape(len(ids), 9)
                # Native 9x9 tangent uses stress packed into nine components.
                plus_stress = np.full((nrow, 9), np.nan); plus_stress[ids] = sp
            else:
                plus_stress = np.full((nrow, 9), np.nan)
            if valid_minus[direction, ep_index].any():
                ids = np.flatnonzero(valid_minus[direction, ep_index])
                sm = base.stress(p[ids], fm[ids], zm[ids]).reshape(len(ids), 9)
                minus_stress = np.full((nrow, 9), np.nan); minus_stress[ids] = sm
            else:
                minus_stress = np.full((nrow, 9), np.nan)

            reference_action = C[:, :, direction]
            expected_p = reference_action.copy()
            expected_m = reference_action.copy()
            goodp = same_plus[direction, ep_index] & (realized_p != 0.0)
            goodm = same_minus[direction, ep_index] & (realized_m != 0.0)
            observed_p = np.full((nrow, 9), np.nan)
            observed_m = np.full((nrow, 9), np.nan)
            observed_p[goodp] = (plus_stress[goodp] - center_stress[goodp]) / realized_p[goodp, None]
            observed_m[goodm] = (center_stress[goodm] - minus_stress[goodm]) / realized_m[goodm, None]
            rp_rel, _, _ = relative_l2(observed_p, expected_p)
            rm_rel, _, _ = relative_l2(observed_m, expected_m)
            plus_rel[direction, ep_index] = rp_rel
            minus_rel[direction, ep_index] = rm_rel

            centered_good = (same_plus[direction, ep_index] & same_minus[direction, ep_index]
                             & (realized_p != 0.0) & (realized_m != 0.0))
            # Only the selected F entry changes, so the exact realized center
            # secant denominator is the sum of the two realized signed steps.
            denom = realized_p + realized_m
            centered_observed = np.full((nrow, 9), np.nan)
            centered_observed[centered_good] = (
                plus_stress[centered_good] - minus_stress[centered_good]
            ) / denom[centered_good, None]
            rr, ee, nn = relative_l2(centered_observed, reference_action)
            centered_rel[direction, ep_index] = rr
            centered_abs[direction, ep_index] = ee
            ref_norm[direction, ep_index] = nn

            progress("basis_epsilon", direction=direction, matrix_entry=[i, j],
                     power=power,
                     plus_strict=int(valid_plus[direction, ep_index].sum()),
                     minus_strict=int(valid_minus[direction, ep_index].sum()),
                     branch_preserving_both=int((same_plus[direction, ep_index] & same_minus[direction, ep_index]).sum()),
                     centered_le_1e6=int(np.sum(centered_good & np.isfinite(rr) & (rr <= DERIVATIVE_RELATIVE_L2))),
                     centered_inconclusive=int(np.sum(~centered_good | ~np.isfinite(rr))))

    # A row/direction qualifies only when at least two adjacent entries in the
    # declared epsilon sweep are strict, branch-preserving centered passes.
    centered_eligible = (valid_plus & valid_minus & same_plus & same_minus
                         & (delta_plus != 0.0) & (delta_minus != 0.0)
                         & np.isfinite(centered_rel) & (ref_norm > 0.0))
    centered_pass = centered_eligible & (centered_rel <= DERIVATIVE_RELATIVE_L2)
    adjacent_pair_pass = np.zeros((ndir, nrow), dtype=bool)
    for k in range(neps - 1):
        adjacent_pair_pass |= centered_pass[:, k, :] & centered_pass[:, k + 1, :]
    directional_summary = []
    for d in range(ndir):
        perrow_passes = centered_pass[d].sum(axis=0)
        perrow_eligible = centered_eligible[d].sum(axis=0)
        directional_summary.append({
            "direction_index": d,
            "F_component": list(divmod(d, 3)),
            "rows": nrow,
            "rows_with_any_resolved_centered_comparison": int(np.any(centered_eligible[d], axis=0).sum()),
            "rows_with_any_centered_pass": int(np.any(centered_pass[d], axis=0).sum()),
            "rows_with_adjacent_centered_passes": int(adjacent_pair_pass[d].sum()),
            "row_direction_inconclusive_no_resolved_comparison": int((~np.any(centered_eligible[d], axis=0)).sum()),
            "zero_reference_action_rows": int(np.sum(np.all(ref_norm[d] == 0.0, axis=0))),
            "maximum_centered_relative_error_over_eligible": float(np.nanmax(np.where(centered_eligible[d], centered_rel[d], np.nan)))
                if np.any(centered_eligible[d]) else None,
            "maximum_centered_absolute_error": float(np.nanmax(centered_abs[d]))
                if np.isfinite(centered_abs[d]).any() else None,
            "one_sided_plus_passes_at_any_grid_point": int(np.any(np.isfinite(plus_rel[d]) & (plus_rel[d] <= DERIVATIVE_RELATIVE_L2), axis=0).sum()),
            "one_sided_minus_passes_at_any_grid_point": int(np.any(np.isfinite(minus_rel[d]) & (minus_rel[d] <= DERIVATIVE_RELATIVE_L2), axis=0).sum()),
        })

    np.savez_compressed(args.output / "nine-basis-fd-arrays.npz",
        epsilon_powers=np.asarray(powers, dtype=np.int16), paper_tets=paper_tets,
        direction_component=np.asarray([divmod(d, 3) for d in range(9)], dtype=np.int8),
        tangent=C, center_branch=center_sig,
        valid_plus=valid_plus, valid_minus=valid_minus, same_plus=same_plus, same_minus=same_minus,
        plus_branch=plus_branch, minus_branch=minus_branch,
        plus_delta=delta_plus, minus_delta=delta_minus,
        plus_residual=plus_residual, minus_residual=minus_residual,
        plus_correction=plus_correction, minus_correction=minus_correction,
        plus_reason=plus_reason, minus_reason=minus_reason,
        plus_iterations=plus_iterations, minus_iterations=minus_iterations,
        centered_relative_l2=centered_rel, centered_absolute_l2=centered_abs,
        reference_action_l2=ref_norm, centered_eligible=centered_eligible,
        centered_pass=centered_pass, adjacent_pair_pass=adjacent_pair_pass,
        plus_one_sided_relative_l2=plus_rel, minus_one_sided_relative_l2=minus_rel)
    report = {
        "schema": "numi.cardboard.bilinear15.nine-basis-fp64-fd-audit.v1",
        "classification": "CPU-only independent compensated FP64 return-map and tangent audit; not native, physical, calibration, or whole-box qualification",
        "equation_source_sha256": sha256(args.source_reference),
        "capture_sha256": sha256(args.capture),
        "mp_parameter_binding_sha256": sha256(args.mp_arrays),
        "portable_module_hashes": portable_hashes,
        "paper_rows": nrow,
        "all_nine_basis_components": True,
        "epsilon_powers": list(powers),
        "strict_gates_unchanged": {"scaled_residual": STRICT_RESIDUAL,
                                   "dynamic_scaled_correction": STRICT_DYNAMIC_CORRECTION,
                                   "centered_derivative_relative_l2": DERIVATIVE_RELATIVE_L2},
        "perturbation": "Each endpoint changes exactly one captured F component by plus/minus 2^-power; all denominators use realized FP64 component differences. The authored material is rate-independent; no Fdot averaging is used.",
        "center_projection": {"strict_pass": int(ok0.sum()), "strict_fail": int((~ok0).sum()),
                             "max_scaled_residual": float(np.nanmax(r0)),
                             "max_dynamic_scaled_correction": float(np.nanmax(c0)),
                             "iterations_min_max": [int(it0.min()), int(it0.max())]},
        "per_basis_direction": directional_summary,
        "epsilon_summaries": [
            {"direction_index": d, "power": powers[k],
             "strict_plus": int(valid_plus[d,k].sum()), "strict_minus": int(valid_minus[d,k].sum()),
             "branch_preserving_plus": int(same_plus[d,k].sum()),
             "branch_preserving_minus": int(same_minus[d,k].sum()),
             "branch_preserving_both": int((same_plus[d,k] & same_minus[d,k]).sum()),
             "centered_eligible": int(centered_eligible[d,k].sum()),
             "centered_pass_le_1e_6": int(centered_pass[d,k].sum()),
             "centered_max_rel_eligible": float(np.nanmax(np.where(centered_eligible[d,k], centered_rel[d,k], np.nan)))
                if centered_eligible[d,k].any() else None,
             "plus_one_sided_eligible": int(np.isfinite(plus_rel[d,k]).sum()),
             "minus_one_sided_eligible": int(np.isfinite(minus_rel[d,k]).sum())}
            for d in range(9) for k in range(neps)
        ],
        "interpretation_limits": [
            "A row-direction is a sampled-grid pass only if two adjacent epsilon entries both pass the unchanged 1e-6 relative-L2 gate with strict endpoints and both endpoint branches matching the center.",
            "Branch-crossing endpoints are retained and excluded from centered smooth-branch gates; one-sided branch-preserving comparisons are reported separately.",
            "Zero realized endpoint deltas and zero reference action norms are inconclusive. FP64 repeatability is not a measured native/noise floor; this audit does not claim an absolute floor.",
            "This verifies the independent FP64 reference’s local derivatives on this captured center set; it does not qualify native Metal finite differences or the full coupon trajectory."
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    progress("complete", elapsed_seconds=report["elapsed_seconds"],
             output_sha256=sha256(args.output / "nine-basis-fd-arrays.npz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
