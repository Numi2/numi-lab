#!/usr/bin/env python3
"""Summarize opt-in NM_FEM_TANGENT_CENSUS records; never infer timing."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

PREFIX = "matter_fem_tangent_census="
SUMMARY_PREFIX = "matter_fem_tangent_census_summary="


def parse_log(path):
    summary, rows = None, []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if line.startswith(SUMMARY_PREFIX):
            if summary is not None:
                raise ValueError(f"{path}:{number}: duplicate summary")
            summary = json.loads(line[len(SUMMARY_PREFIX):])
        elif line.startswith(PREFIX):
            rows.append(json.loads(line[len(PREFIX):]))
    if summary is None or not rows:
        raise ValueError(f"{path}: census summary or material records missing")
    if summary.get("schema") != 1 or not summary.get("source_work_only"):
        raise ValueError(f"{path}: unsupported census schema/claim level")
    if len(rows) != summary.get("material_count"):
        raise ValueError("material record count disagrees with summary")
    if len({r["material_index"] for r in rows}) != len(rows):
        raise ValueError("duplicate material indices")
    if not all(r.get("program_ranges_valid") for r in rows):
        raise ValueError("one or more scalar program ranges are invalid")
    if sum(r["active_fem_tetrahedra"] for r in rows) != summary["active_fem_tetrahedra"]:
        raise ValueError("per-material active tetrahedron sum mismatch")
    return summary, sorted(rows, key=lambda r: r["material_index"])


def fmt(n):
    return f"{n:,}"


def report(log_path, invocation_path):
    summary, rows = parse_log(log_path)
    invocation = json.loads(invocation_path.read_text()) if invocation_path.exists() else {}
    evidence_root = Path(__file__).parent
    provenance_path = evidence_root / "PROVENANCE-CORRECTION.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    source_manifest_path = evidence_root / "source-sha256.json"
    source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else {}
    groups = defaultdict(lambda: {"materials": 0, "tets": 0, "visits": 0})
    for row in rows:
        row["estimated_tangent_visits"] = (
            row["active_fem_tetrahedra"] * 9 * row["tangent_instruction_count"]
        )
        g = groups[row["family"]]
        g["materials"] += 1
        g["tets"] += row["active_fem_tetrahedra"]
        g["visits"] += row["estimated_tangent_visits"]
    stateful = [r for r in rows if r["state_count"] > 0]
    implicit_tets = sum(r["active_fem_tetrahedra"] for r in rows if r["implicit_state"])
    generic_implicit_tets = sum(
        r["active_fem_tetrahedra"] for r in rows
        if r["implicit_state"] and r["projection"] == "generic"
    )
    canonical_candidates = [r for r in rows if r["canonical_tangent_candidate"]]
    canonical_enabled = [r for r in rows if r["canonical_tangent_enabled"]]
    shared_candidates = [r for r in rows if r["shared15_candidate"]]
    shared_enabled = [r for r in rows if r["shared15_enabled"]]
    ranking = sorted(rows, key=lambda r: (-r["estimated_tangent_visits"], r["material_index"]))
    total_visits = sum(r["estimated_tangent_visits"] for r in rows)
    stress_weight = sum(
        r["active_fem_tetrahedra"] * r["stress_instruction_count"] for r in rows
    )
    one_direction_weight = sum(
        r["active_fem_tetrahedra"] * r["tangent_instruction_count"] for r in rows
    )

    out = [
        "# Human FEM material tangent census",
        "",
        f"Input log: {log_path}",
        f"Binary SHA256: {invocation.get('binary_sha256', 'not recorded')}",
        f"Execution source HEAD (provenance correction): {provenance.get('execution_source_head', 'not recorded')}",
        f"Invocation source_head (inherited template field, not authoritative): {invocation.get('source_head', 'not recorded')}",
        f"Working-source runtime.mm SHA256 (source-sha256.json): {source_manifest.get('matter/src/runtime.mm', 'not recorded')}",
        f"Working-source FEM shader SHA256 (source-sha256.json): {source_manifest.get('matter/src/metal/fem.metalinc', 'not recorded')}",
        "",
        "The runtime emitted this census once while uploading the compiled Human FEM world with NM_FEM_TANGENT_CENSUS=1. Instruction and opcode counts are from compiled scalar bytecode descriptors; tetrahedra are active object/tet assignments. This is source-work accounting, not shader instruction execution, call counts, cache hits, projection iterations, factorization counts, or timing.",
        "",
        "Estimated tangent visits use the requested formula: active_fem_tetrahedra * 9 * tangent_instruction_count, where tangent_instruction_count sums the nine compiled tangent scalar programs. This is a potential bytecode-work weight across nine deformation directions, not a runtime visit count. The stress weight assumes one stress-program-set visit per active tetrahedron only as a reference.",
        "",
        "## Totals",
        "",
        f"- Compiled material descriptors: **{fmt(len(rows))}**.",
        f"- FEM tetrahedron slots: **{fmt(summary['stored_tetrahedra'])} stored; {fmt(summary['active_fem_tetrahedra'])} active**.",
        "- Constitutive families: " + ", ".join(
            f"{name}={values['materials']}" for name, values in sorted(groups.items())
        ) + ".",
        f"- Stateful descriptors: **{len(stateful)}**; active implicit-state FEM tetrahedra: **{fmt(implicit_tets)}**.",
        f"- Active generic implicit-projection tetrahedra (potential generic bytecode state solve): **{fmt(generic_implicit_tets)}**; this is static eligibility, not observed fallback calls.",
        f"- Canonical tangent candidates/enabled: **{len(canonical_candidates)}/{len(canonical_enabled)}**.",
        f"- Shared 15-state factor candidates/enabled: **{len(shared_candidates)}/{len(shared_enabled)}**.",
        f"- Estimated tangent bytecode visits over nine directions: **{fmt(total_visits)}**.",
        f"- Reference stress weight (active tets * stress instructions): **{fmt(stress_weight)}**; tangent weight for one direction: **{fmt(one_direction_weight)}**.",
        "",
        "## Per-material breakdown ranked by estimated tangent visits",
        "",
        "| Rank | Material | Family/projection | Active tets | States / implicit | Canonical candidate/enabled | Stress instructions (exp/log) | Tangent instructions (exp/log) | Estimated tangent visits |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, r in enumerate(ranking, 1):
        out.append(
            f"| {rank} | {r['name']} | {r['family']}/{r['projection']} "
            f"| {fmt(r['active_fem_tetrahedra'])} "
            f"| {r['state_count']}/{int(r['implicit_state'])} "
            f"| {int(r['canonical_tangent_candidate'])}/{int(r['canonical_tangent_enabled'])} "
            f"| {fmt(r['stress_instruction_count'])} ({r['stress_exp_count']}/{r['stress_log_count']}) "
            f"| {fmt(r['tangent_instruction_count'])} ({r['tangent_exp_count']}/{r['tangent_log_count']}) "
            f"| {fmt(r['estimated_tangent_visits'])} |"
        )
    out += [
        "",
        "## Family-weighted tangent visits",
        "",
        "| Family | Materials | Active tets | Estimated visits |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, g in sorted(groups.items(), key=lambda item: (-item[1]["visits"], item[0])):
        out.append(f"| {name} | {g['materials']} | {fmt(g['tets'])} | {fmt(g['visits'])} |")
    out += [
        "",
        "The earlier 8.06-second tangent-build profile remains unattributed by material: this census does not measure dynamic call counts or disjoint GPU stage time.",
        "",
    ]
    return "\n".join(out)


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log", nargs="?", type=Path,
        default=root / "global-icb-census-1" / "stderr.log",
    )
    parser.add_argument("--invocation", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=root / "TANGENT-CENSUS.md")
    args = parser.parse_args()
    invocation = args.invocation or args.log.parent / "invocation.json"
    args.output.write_text(report(args.log, invocation))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
