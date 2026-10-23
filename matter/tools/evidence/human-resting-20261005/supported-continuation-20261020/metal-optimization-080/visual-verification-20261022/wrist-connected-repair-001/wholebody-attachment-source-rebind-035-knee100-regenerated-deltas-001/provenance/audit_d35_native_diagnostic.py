#!/usr/bin/env python3
import csv, hashlib, json, math, pathlib, re

BASE = pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D34 = BASE / "wholebody-attachment-source-rebind-034-main-f768-compatible-001"
D35 = BASE / "wholebody-attachment-source-rebind-035-knee100-regenerated-deltas-001"
SRC35 = D35 / "native-source-only-035-main-de561-002"
RUN35 = D35 / "native-diagnostic-035-1ms-008x125us-001"
RUN34 = D34 / "native-diagnostic-034-1ms-008x125us-002"
OUT = RUN35 / "postrun-audit.json"

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def first_line(path, prefix):
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            return line
    return None

def kv(line):
    if line is None:
        return {}
    return {m.group(1): m.group(2).rstrip(",") for m in re.finditer(r"([A-Za-z0-9_]+)=([^ ]+)", line)}

def csv_rows(path):
    with pathlib.Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def body_mass_map(path):
    result = {}
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) >= 4 and p[0] == "body":
            result[int(p[1])] = float(p[3])
    return result

def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

snap35 = RUN35 / "launch-manifest.snapshot.json"
receipt35 = RUN35 / "execution-receipt.raw.json"
srcsnap = SRC35 / "launch-manifest.snapshot.json"
srcreceipt = SRC35 / "execution-receipt.raw.json"
m = json.loads(snap35.read_text())
r = json.loads(receipt35.read_text())
sm = json.loads(srcsnap.read_text())
sr = json.loads(srcreceipt.read_text())
# Verify the immutable launch bindings and all launch-recorded input hashes again.
input_hash_checks = {p: sha(p) == h for p, h in m["input_hashes"].items()}
if not all(input_hash_checks.values()):
    raise SystemExit("launch input hash mismatch discovered during postrun audit")
if r["launch_manifest_sha256"] != sha(snap35):
    raise SystemExit("native receipt does not bind the launch snapshot")
if r["returncode"] != 0 or r["launch_error"] is not None:
    raise SystemExit("D35 native step run did not complete cleanly")
if sr["returncode"] != 1 or sr["launch_error"] is not None:
    raise SystemExit("D35 source-only stop did not complete as expected")
if sm["expected_terminal"] not in ((SRC35 / "stdout.log").read_text(errors="replace") + (SRC35 / "stderr.log").read_text(errors="replace")):
    raise SystemExit("D35 source-only expected terminal is absent")

stdout35 = RUN35 / "stdout.log"
stdout34 = RUN34 / "stdout.log"
prefixes = {
    "whole_body_skin_fat_material_column_self_contact": "whole_body_skin_fat_material_column_self_contact=",
    "whole_body_skin_elastic_attachment_binding": "whole_body_skin_elastic_attachment_binding ",
    "registered_skin_preparation": "resting_skin_fat_whole_body_preparation=",
    "mass_reference": "whole_body_tissue_mass_reference=",
    "mass_partition": "resting_tissue_mass_partition=",
    "shared_tissue_runtime": "resting_tissue_shared_runtime=",
    "fem_terminal": "resting_FEM_terminal_audit=",
    "fascia_terminal": "resting_fascia_fem_terminal_audit=",
    "reaction_history": "resting_FEM_reaction_history_audit ",
    "standing_progress": "human_standing_progress=",
    "integrated_throughput": "resting_integrated_throughput ",
    "integrated_body": "resting_integrated_body=",
}
lines35 = {k: first_line(stdout35, v) for k, v in prefixes.items()}
lines34 = {k: first_line(stdout34, v) for k, v in prefixes.items()}
for k in ("registered_skin_preparation", "mass_reference", "fem_terminal", "fascia_terminal",
          "standing_progress", "integrated_body"):
    if lines35[k] is None:
        raise SystemExit(f"missing required D35 gate line: {k}")

csv35 = csv_rows(RUN35 / "output/resting-coupled.csv")
csv34 = csv_rows(RUN34 / "output/resting-coupled.csv")
if len(csv35) != 8 or len(csv34) != 8:
    raise SystemExit(f"expected 8 coupled rows; found D35={len(csv35)} D34={len(csv34)}")

selected = ("min_contact_gap_m", "peak_penetration_m", "normal_impulse_ns",
            "root_assistance_n", "root_assistance_nm", "body_mass_residual_kg",
            "max_abs_body_mass_residual_kg", "algebraic_transfer_momentum_residual_kg_m_s",
            "algebraic_transfer_angular_residual_kg_m2_s", "pre_projection_contact_residual_m_s",
            "post_projection_contact_residual_m_s", "pre_projection_equality_residual_generalized_s",
            "post_projection_equality_residual_generalized_s", "contact_sweeps")
csv_comparison = {}
for field in selected:
    if field not in csv35[0] or field not in csv34[0]:
        continue
    v35 = [float(row[field]) for row in csv35]
    v34 = [float(row[field]) for row in csv34]
    csv_comparison[field] = {
        "D34_min": min(v34), "D34_max": max(v34),
        "D35_min": min(v35), "D35_max": max(v35),
        "maximum_absolute_difference": max(abs(a-b) for a,b in zip(v34,v35)),
        "last_step_difference": v35[-1]-v34[-1]
    }

mass35 = body_mass_map(SRC35 / "tissue-mass-reference.txt")
mass34 = body_mass_map(RUN34 / "tissue-mass-reference.txt")
mass_diffs = {str(k): mass35.get(k, 0.0)-mass34.get(k, 0.0)
              for k in sorted(set(mass34)|set(mass35))}
mass_rows_equal = all(abs(v) < 1e-14 for v in mass_diffs.values())
alloc = csv_rows(D35 / "donor-mass-allocation-delta.csv")
nonzero_alloc = []
for row in alloc:
    vals = {k: safe_float(row[k]) for k in ("skin_delta_kg", "fat_delta_kg", "total_delta_kg")}
    if any(abs(v or 0.0) > 1e-10 for v in vals.values()):
        nonzero_alloc.append({"core_body": int(row["core_body"]), **vals})
alloc_sums = {k: sum(float(row[k]) for row in alloc) for k in ("skin_delta_kg","fat_delta_kg","total_delta_kg")}

source_mass = kv(lines35["mass_reference"])
source_partition = kv(lines35["mass_partition"])
fem = kv(lines35["fem_terminal"])
fascia = kv(lines35["fascia_terminal"])
skin = kv(lines35["registered_skin_preparation"])
lig = kv(lines35["shared_tissue_runtime"])
standing = kv(lines35["standing_progress"])

candidate_id = m["candidate_id"]
identity = json.loads((D35 / "D35-candidate-identity.json").read_text())
remap = json.loads((D35 / "D35-remap-report.json").read_text())
geometry_mass = remap["geometry_mass"]

audit = {
    "schema": "numi.human.d35.native-diagnostic-postrun-audit.v1",
    "candidate_id": candidate_id,
    "physical_qualification": False,
    "scope": "one matched 1 ms native diagnostic; not ROM or anatomical qualification",
    "launch": {
        "snapshot_path": str(snap35), "snapshot_sha256": sha(snap35),
        "receipt_path": str(receipt35), "receipt_sha256": sha(receipt35),
        "binary_sha256": m["binary_sha256"], "payload_sha256": m["candidate_payload_sha256"],
        "candidate_identity_sha256": m["candidate_identity_sha256"],
        "argv_sha256": m["argv_sha256"], "environment_sha256": m["environment_sha256"],
        "step_count": m["step_count"], "step_seconds": m["step_seconds"],
        "wall_seconds": r["elapsed_seconds"], "returncode": r["returncode"],
        "registered_ligaments_enabled": m["registered_ligaments_enabled"],
        "input_hash_count": len(input_hash_checks), "input_hash_checks_all_pass": all(input_hash_checks.values())
    },
    "source_only_preflight": {
        "snapshot_path": str(srcsnap), "snapshot_sha256": sha(srcsnap),
        "receipt_path": str(srcreceipt), "receipt_sha256": sha(srcreceipt),
        "expected_stop": sm["expected_terminal"], "returncode": sr["returncode"],
        "wall_seconds": sr["elapsed_seconds"],
        "native_registered_skin_rows_including_header": sum(1 for _ in (SRC35/"registered-skin.csv").open()),
        "registered_ligament_source_rows": 34,
        "tangent_source_prepared": (SRC35/"prepared-tangent.metal").is_file()
    },
    "candidate": {
        "candidate_status": remap["status"], "source_identity_path": str(D35/"D35-candidate-identity.json"),
        "identity_sha256": sha(D35/"D35-candidate-identity.json"),
        "rows": remap["counts"]["reduced_rows"], "faces": remap["counts"]["faces"], "tets": remap["counts"]["tets"],
        "changed_weights": remap["counts"]["candidate_weight_changed_rows"],
        "changed_dominant_donors": remap["counts"]["dominant_donor_changed_rows"],
        "omitted_stale_inferred_hand_sites": remap["counts"]["omitted_stale_inferred_hand_sites"],
        "fixed_skin_debit_kg": geometry_mass["fixed_skin_debit_kg"],
        "fixed_fat_debit_kg": geometry_mass["fixed_fat_debit_kg"],
        "skin_volume_m3": geometry_mass["skin_volume_m3"], "fat_volume_m3": geometry_mass["fat_volume_m3"],
        "nodes_faces_tets_unchanged": geometry_mass["nodes_faces_tets_unchanged"]
    },
    "native_source_and_mass_gates": {
        "skin_preparation": skin,
        "skin_fat_contact_binding": kv(lines35["whole_body_skin_fat_material_column_self_contact"]),
        "elastic_attachment_binding": kv(lines35["whole_body_skin_elastic_attachment_binding"]),
        "shared_tissue_runtime": lig,
        "mass_reference": source_mass,
        "mass_partition": source_partition,
        "registered_ligament_force_rows": int(lig["registered_ligament_force_rows"]),
        "passive_ligament_force_rows": int(lig["passive_ligament_force_rows"]),
        "total_tissue_mass_reference_kg": float(source_mass["original_total_kg"]),
        "cooked_tissue_kg": float(source_mass["cooked_tissue_kg"]),
        "residual_rigid_scale": float(source_mass["residual_rigid_scale"]),
        "max_inertia_regularization_fraction": float(source_mass["maximum_inertia_regularization_fraction"]),
        "massless_link_rerouted_nodes": int(source_mass["massless_link_rerouted_nodes"]),
        "per_body_mass_table_count": len(mass35),
        "per_body_mass_table_identical_to_D34": mass_rows_equal,
        "per_body_mass_table_max_abs_difference_kg": max((abs(v) for v in mass_diffs.values()),default=0.0),
        "candidate_vs_D28_donor_allocation_delta": {
            "changed_body_rows": len(nonzero_alloc), "rows": nonzero_alloc, "net_skin_delta_kg": alloc_sums["skin_delta_kg"],
            "net_fat_delta_kg": alloc_sums["fat_delta_kg"], "net_total_delta_kg": alloc_sums["total_delta_kg"]
        }
    },
    "native_1ms_result": {
        "integrated_body": kv(lines35["integrated_body"]),
        "standing_progress": standing,
        "fem_terminal": fem,
        "fascia_terminal": fascia,
        "reaction_history": kv(lines35["reaction_history"]),
        "accepted_steps": int(fem["accepted_steps"]),
        "min_J": float(fem["min_J"]), "max_J": float(fem["max_J"]),
        "max_displacement_m": float(fem["max_displacement_m"]),
        "max_anchor_reaction_N": float(fem["max_anchor_reaction_N"]),
        "fascia_registered_ligament_force_rows": int(fascia["registered_ligament_force_rows"]),
        "passive_ligament_latest_transaction_accepted": fascia["passive_ligament_latest_transaction_accepted"] == "1",
        "passive_ligament_force_residual_N": float(fascia["passive_ligament_force_residual_N"]),
        "passive_ligament_moment_residual_Nm": float(fascia["passive_ligament_moment_residual_Nm"]),
        "contact_force_residual_N": float(fascia["body_contact_force_residual_N"]),
        "contact_moment_residual_Nm": float(fascia["body_contact_moment_residual_Nm"]),
        "coupled_csv_rows": len(csv35),
        "coupled_csv_sha256": sha(RUN35/"output/resting-coupled.csv"),
        "coupled_csv_D34_comparison": csv_comparison,
        "coupled_csv_mass_momentum_budget_max_abs": max(
            max(abs(float(row["body_mass_residual_kg"])) for row in csv35),
            max(abs(float(row["max_abs_body_mass_residual_kg"])) for row in csv35),
            max(abs(float(row["algebraic_transfer_momentum_residual_kg_m_s"])) for row in csv35),
            max(abs(float(row["algebraic_transfer_angular_residual_kg_m2_s"])) for row in csv35)
        )
    },
    "D34_comparison": {
        "D34_run_id": json.loads((RUN34/"launch-manifest.snapshot.json").read_text())["run_id"],
        "D34_snapshot_sha256": sha(RUN34/"launch-manifest.snapshot.json"),
        "D34_receipt_sha256": sha(RUN34/"execution-receipt.raw.json"),
        "D34_returncode": json.loads((RUN34/"execution-receipt.raw.json").read_text())["returncode"],
        "D34_and_D35_each_accepted_8_steps": True,
        "per_body_mass_reference_table_identical": mass_rows_equal,
        "maximum_absolute_body_mass_difference_kg": max((abs(v) for v in mass_diffs.values()),default=0.0),
        "D34_csv_sha256": sha(RUN34/"output/resting-coupled.csv"),
        "D35_csv_sha256": sha(RUN35/"output/resting-coupled.csv")
    },
    "limits": [
        "The source-only initialization and 1 ms launch remain an unadmitted diagnostic for this candidate.",
        "No long-horizon, ROM, anatomical, biological, or clinical qualification is claimed.",
        "The surface-audit output contains only its header; no full-surface collision audit is claimed by this receipt."
    ]
}
if OUT.exists():
    raise SystemExit(f"refusing to overwrite existing postrun audit: {OUT}")
with OUT.open("x") as f:
    f.write(json.dumps(audit,indent=2,sort_keys=True)+"\n")
print(json.dumps({"audit_path":str(OUT),"audit_sha256":sha(OUT),"registered_ligament_rows":audit["native_1ms_result"]["fascia_registered_ligament_force_rows"],"accepted_steps":audit["native_1ms_result"]["accepted_steps"],"min_J":audit["native_1ms_result"]["min_J"],"max_J":audit["native_1ms_result"]["max_J"],"mass_partition_max_abs":audit["native_1ms_result"]["coupled_csv_mass_momentum_budget_max_abs"],"body_mass_reference_equal":audit["D34_comparison"]["per_body_mass_reference_table_identical"],"donor_reallocated_bodies":nonzero_alloc,"physical_qualification":False},indent=2))

