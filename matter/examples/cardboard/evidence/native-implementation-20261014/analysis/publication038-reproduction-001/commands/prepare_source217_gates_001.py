from pathlib import Path
import json,hashlib,shutil,difflib,subprocess,sys,datetime,os
q=Path("/Users/n/cardboard-implementation-20261014-host-macmini");base=q/"snapshots/source-composed-strict-periodic-freebox-210-buildfix-003";s=q/"snapshots/source-capacity-damage-debond-gates-217-001";d=q/"analysis/source217-capacity-damage-debond-gates-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base/"frozen-source.json")=="477fe6943fb1c28e782af9ce0c1c383c013146ab12f422bc55523a6834597931"
f=json.loads((base/"frozen-source.json").read_text());s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in f["source_sha256"].items():
 assert sha(base/n)==h,n
 dst=s/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,dst)
rel="matter/tools/cardboard_stage_resolution_study.py";p=s/rel;t=p.read_text()
anchor="\ndef _impact_histories(run: StageRun)"
assert t.count(anchor)==1
helper='''

def _capacity_loss_debond_gate(history_a: dict[str, Any], history_b: dict[str, Any],
                                differences: dict[str, Any]) -> dict[str, Any]:
    """Irreversible failed reference area includes closed, contact-bearing faces."""
    roles: dict[str, Any] = {}
    available_status = "available_zero_tensile_shear_capacity_not_geometric_open_area"
    for role in ("starch", "hma"):
        left = (history_a.get("fully_failed_cohesive_capacity_area") or {}).get(role, {})
        right = (history_b.get("fully_failed_cohesive_capacity_area") or {}).get(role, {})
        if left.get("status") == right.get("status") == "not_enabled":
            roles[role] = {"status": "not_enabled"}
            continue
        item = differences.get(role, {})
        difference = item.get("absolute_difference")
        if left.get("status") != available_status or right.get("status") != available_status or \\
                item.get("status") != "available" or \\
                not isinstance(difference, (float, int)) or not math.isfinite(difference):
            roles[role] = {"status": "inconclusive_missing_or_incompatible_accepted_reference_area"}
            continue
        roles[role] = {
            "status": "pass" if 0.0 <= difference <= 0.02 else "fail",
            "absolute_fraction_difference": difference,
            "absolute_fraction_limit": 0.02,
        }
    states = [item["status"] for item in roles.values()]
    status = ("fail" if "fail" in states else
              "inconclusive" if any(value not in ("pass", "not_enabled") for value in states) else
              "pass" if "pass" in states else "not_enabled")
    return {
        "status": status, "per_interface": roles,
        "definition": "reference quadrature area with accepted irreversible damage D exactly 1, divided by matching activated reference interface area",
        "threshold": "D == 1; no tensile or shear capacity remains",
        "quadrature": "three equal reference-area weights per mixed-mode face",
        "closed_failed_interfaces_count_as_debonded": True,
        "geometric_open_area_is_a_separate_observable": True,
    }


def _capacity_history_gates(report: dict[str, Any]) -> dict[str, Any]:
    damage = report.get("metrics", {}).get("damage_and_debond", {})
    left, right = damage.get("left", {}), damage.get("right", {})
    # The existing paper index normalizes by a mesh-dependent sum of all tet
    # cut-plane areas. It is not an objective crack-area fraction. A force-only
    # match must not let an enabled but unmeasured localization field qualify.
    paper = {
        "status": ("not_enabled" if left.get("paper_status") == right.get("paper_status") == "not_enabled"
                   else "inconclusive_missing_mesh_objective_paper_damage_fraction"),
        "absolute_fraction_limit": 0.02,
        "constitutive_damage_index_substituted_for_area": False,
    }
    debond = damage.get("constitutive_debond_fraction_gate") or {"status": "inconclusive"}
    return {"paper_damage_fraction": paper, "irreversible_debond_fraction": debond}

'''
t=t.replace(anchor,helper+anchor)
anchor='    report["metrics"] = metrics\n    report["status"] = "descriptive_metrics_available_with_unresolved_observables"'
replacement='''    metrics["damage_and_debond"]["constitutive_debond_fraction_gate"] = _capacity_loss_debond_gate(
        history_a, history_b, metrics["damage_and_debond"]["fully_failed_cohesive_capacity_area"])
    report["metrics"] = metrics
    report["status"] = "descriptive_metrics_available_with_unresolved_observables"'''
assert t.count(anchor)==1;t=t.replace(anchor,replacement)
anchor='''        finest_metrics = pairs[-1].get("metrics", {}).get("compression_force_work", {})
    metric_names = ('''
replacement='''        finest_metrics = pairs[-1].get("metrics", {}).get("compression_force_work", {})
    history_gates = []
    for pair_index, pair in enumerate(pairs):
        gates = _capacity_history_gates(pair)
        history_gates.append(gates)
        for name, gate in gates.items():
            if gate.get("status") not in ("pass", "not_enabled"):
                reasons.append(f"adjacent_pair_{pair_index}_{name}_{gate.get('status', 'inconclusive')}")
    if len(history_gates) != 2:
        reasons.append("missing_two_adjacent_damage_and_debond_comparisons")
    metric_names = ('''
assert t.count(anchor)==1;t=t.replace(anchor,replacement)
anchor='''        "finest_pair_within_tolerance": threshold_pass,
        "adjacent_pair_differences_decrease": adjacent_differences_decrease,'''
replacement='''        "finest_pair_within_tolerance": threshold_pass,
        "accepted_damage_and_debond_gates": history_gates,
        "capacity_requires_damage_and_debond_observables": True,
        "adjacent_pair_differences_decrease": adjacent_differences_decrease,'''
assert t.count(anchor)==1;t=t.replace(anchor,replacement)
p.write_text(t)
reltest="matter/tools/test_cardboard_stage_resolution_study.py";p=s/reltest;t=p.read_text()
anchor="                 fully_failed_fraction: float = 0.25) -> Path:"
assert t.count(anchor)==1;t=t.replace(anchor,"                 fully_failed_fraction: float = 0.25,\n                 paper_damage_enabled: bool = True) -> Path:")
anchor='"accepted": 1, "paper_status": "available_model_area_weighted_damage_index",'
assert t.count(anchor)==1;t=t.replace(anchor,'"accepted": 1, "paper_status": ("available_model_area_weighted_damage_index" if paper_damage_enabled else "not_enabled"),')
# The existing positive capacity fixture now explicitly disables paper damage:
# its paper localization observable is deliberately unavailable.
start=t.index("    def test_capacity_evidence_requires_complete_mesh_and_time_triplets(")
end=t.index("    def test_capacity_evidence_rejects_half_speed_only_or_prefix_cases(",start)
block=t[start:end]
assert block.count("force_scale=")==5
for val in ["1.03","1.01","1.0"]:
 block=block.replace("force_scale="+val+")","force_scale="+val+", paper_damage_enabled=False)")
t=t[:start]+block+t[end:]
anchor="    def test_material_change_outside_declared_axis_rejects_pair(self) -> None:"
assert t.count(anchor)==1
tests='''    def test_closed_failed_area_counts_as_constitutive_debond_and_enforces_two_points(self) -> None:
        left = fake_box_run(self.root, "left", fully_failed_fraction=0.2)
        right = fake_box_run(self.root, "right", nx=16, fully_failed_fraction=0.219)
        report = study.compare_runs(left, right, "mesh_nx", "all")
        gate = report["metrics"]["damage_and_debond"]["constitutive_debond_fraction_gate"]
        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["closed_failed_interfaces_count_as_debonded"])
        self.assertTrue(gate["geometric_open_area_is_a_separate_observable"])
        farther = fake_box_run(self.root, "farther", nx=16, fully_failed_fraction=0.221)
        failed = study.compare_runs(left, farther, "mesh_nx", "all")
        self.assertEqual(failed["metrics"]["damage_and_debond"]["constitutive_debond_fraction_gate"]["status"], "fail")

    def test_capacity_rejects_force_match_with_unmeasured_enabled_paper_damage(self) -> None:
        cases = [
            fake_box_run(self.root, "mesh-coarse", nx=8, force_scale=1.03),
            fake_box_run(self.root, "middle", nx=16, dt=0.15, force_scale=1.01),
            fake_box_run(self.root, "mesh-fine", nx=32, force_scale=1.0),
            fake_box_run(self.root, "time-coarse", nx=16, dt=0.3, force_scale=1.03),
            fake_box_run(self.root, "time-fine", nx=16, dt=0.075, force_scale=1.0),
        ]
        a, b, c, d, e = cases
        target = self.root / "blocked-paper" / "capacity.properties"
        report = study.create_capacity_qualification(b, a, b, c, d, b, e, target)
        self.assertEqual(report["status"], "not_qualified")
        self.assertFalse(target.exists())
        self.assertTrue(report["mesh_study"]["finest_pair_within_tolerance"])
        self.assertTrue(any("paper_damage_fraction_inconclusive" in reason
                            for reason in report["mesh_study"]["reasons"]))

    def test_capacity_rejects_excessive_debond_difference_even_with_force_match(self) -> None:
        a = fake_box_run(self.root, "coarse", nx=8, force_scale=1.03,
                         fully_failed_fraction=0.1, paper_damage_enabled=False)
        b = fake_box_run(self.root, "middle", nx=16, force_scale=1.01,
                         fully_failed_fraction=0.15, paper_damage_enabled=False)
        c = fake_box_run(self.root, "fine", nx=32, force_scale=1.0,
                         fully_failed_fraction=0.2, paper_damage_enabled=False)
        _, gate = study._capacity_triplet_report(a, b, c, "mesh_nx")
        self.assertFalse(gate["qualified"])
        self.assertTrue(gate["finest_pair_within_tolerance"])
        self.assertTrue(any("irreversible_debond_fraction_fail" in reason
                            for reason in gate["reasons"]))

    def test_capacity_does_not_admit_legacy_face_maximum_or_mismatched_reference_areas(self) -> None:
        a = fake_box_run(self.root, "left", paper_damage_enabled=False)
        b = fake_box_run(self.root, "right", nx=16, paper_damage_enabled=False)
        file = b / "shipping-history-observables.csv"
        fields, rows = study._rows(file)
        rows[0]["starch_fully_failed_status"] = "available_zero_tensile_shear_capacity_fraction_not_geometric_open_area"
        write_csv(file, fields, rows)
        report = study.compare_runs(a, b, "mesh_nx", "compression")
        self.assertEqual(study._capacity_history_gates(report)["irreversible_debond_fraction"]["status"], "inconclusive")
        rows[0]["starch_fully_failed_status"] = "available_quadrature_zero_tensile_shear_capacity_fraction_not_geometric_open_area"
        rows[0]["starch_active_reference_area_m2"] = "0.09"
        rows[0]["starch_fully_failed_cohesive_area_m2"] = "0.0225"
        write_csv(file, fields, rows)
        report = study.compare_runs(a, b, "mesh_nx", "compression")
        self.assertEqual(study._capacity_history_gates(report)["irreversible_debond_fraction"]["status"], "inconclusive")

'''
t=t.replace(anchor,tests+anchor);p.write_text(t)
reldoc="matter/tools/cardboard_stage_resolution_study.md";p=s/reldoc
p.write_text(p.read_text()+"""
## Irreversible debond fraction and capacity admission

A fully failed cohesive quadrature point has permanently lost tensile and shear
capacity. Its reference area counts as constitutively debonded even if subsequent
closure brings the faces back into compressive contact. The 0.02 absolute fraction
gate uses the three equal reference quadrature weights, accepted D exactly 1, and
the same activated reference-area denominator on each mesh. It reports starch
and HMA separately. Geometric opening remains a separate unavailable observable.
Legacy maximum-face-damage exports and incompatible activation coverage are not
admitted.

Capacity qualification now requires accepted paper-damage and irreversible-debond
comparisons as well as decreasing force, curve and work differences. A force-only
match cannot qualify Fpeak while an enabled paper localization field remains
unmeasured. The current per-tet area-weighted damage index is retained but is not
substituted for a mesh-objective fractured-area fraction. Its missing observable
keeps the damage-enabled capacity gate inconclusive. Explicitly disabled mechanisms
may be not applicable; that does not qualify a damage-enabled cardboard model.
""")
paths=set(f["source_sha256"])
for n in paths:f["source_sha256"][n]=sha(s/n)
changed=[n for n in paths if f["source_sha256"][n]!=json.loads((base/"frozen-source.json").read_text())["source_sha256"][n]]
assert sorted(changed)==sorted([rel,reltest,reldoc]),changed
f["source217_capacity_observable_gates"]={"parent_snapshot":str(base),"parent_frozen_sha256":sha(base/"frozen-source.json"),"changed":sorted(changed),"constitutive_debond_fraction_definition":"irreversible accepted D=1 quadrature reference area; independent of current opening","physical_operator_changed":False,"enabled_paper_damage_area_remains_unresolved":True}
(s/"frozen-source.json").write_text(json.dumps(f,indent=2,sort_keys=True)+"\n")
patch="".join("".join(difflib.unified_diff((base/n).read_text().splitlines(True),(s/n).read_text().splitlines(True),fromfile="a/"+n,tofile="b/"+n)) for n in sorted(changed))
(d/"source-overlay.patch").write_text(patch)
cmd=[sys.executable,"-B",str(s/reltest)]
env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"}
with (d/"tests.stdout").open("xb") as out,(d/"tests.stderr").open("xb") as err:r=subprocess.run(cmd,stdout=out,stderr=err,env=env,cwd=s)
report={"schema":"numi.cardboard.source217.v1","source_snapshot":str(s),"frozen_sha256":sha(s/"frozen-source.json"),"source_count":len(paths),"changed":sorted(changed),"patch_sha256":sha(d/"source-overlay.patch"),"test_argv":cmd,"test_exit_code":r.returncode,"test_stdout_sha256":sha(d/"tests.stdout"),"test_stderr_sha256":sha(d/"tests.stderr"),"gpu_execution":False,"physical_transactions":0,"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/"report.json").write_text(json.dumps(report,indent=2)+"\n")
(d/"manifest.json").write_text(json.dumps({"schema":"numi.cardboard.evidence-files.v1","files":{p.name:sha(p) for p in d.iterdir() if p.is_file()}},indent=2)+"\n")
print(json.dumps(report),flush=True);sys.exit(r.returncode)
