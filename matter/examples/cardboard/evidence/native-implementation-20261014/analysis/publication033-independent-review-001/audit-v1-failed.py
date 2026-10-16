from pathlib import Path
import datetime, gzip, hashlib, json, subprocess

Q = Path("/Users/n/cardboard-implementation-20261014-host-macmini")
R = Q / "checkout-publication032"
P = "matter/examples/cardboard/evidence/native-implementation-20261014/"
D = R / P
O = Q / "analysis/publication033-independent-review-001"
assert not O.exists()
O.mkdir(parents=True)

def H(data):
    return hashlib.sha256(data).hexdigest()
def filehash(path):
    return H(Path(path).read_bytes())
def git(*args):
    return subprocess.check_output(["git", *args], cwd=R)

assert git("rev-parse", "HEAD").decode().strip() == "877cd6e69657c9a7b4e5032bcd475705bcc06c85"
binding_path = D / "publication-binding-reference033.json"
binding_bytes = binding_path.read_bytes()
binding = json.loads(binding_bytes)
assert binding["schema"] == "numi.cardboard.reference033-publication-binding.v1"
assert binding["parent"] == "877cd6e69657c9a7b4e5032bcd475705bcc06c85"

head_summary = git("show", "HEAD:" + P + "summary.json")
old_summary = json.loads(head_summary)
historical_file = D / binding["historical_prefix"]["path"]
assert historical_file.read_bytes() == head_summary
assert H(head_summary) == binding["historical_prefix"]["sha256"]
assert len(old_summary["records"]) == 1669
summary_path = D / "summary.json"
summary_bytes = summary_path.read_bytes()
summary = json.loads(summary_bytes)
assert H(summary_bytes) == binding["summary_sha256"]
assert summary["records"][:1669] == old_summary["records"]
assert len(summary["records"]) == 1669 + binding["new_case_records"]
assert summary["records"][1669:] == [x for x in summary["records"][1669:]]
new_records = summary["records"][1669:]
cases = binding["cases"]
assert len(cases) == 145 and len(set(cases)) == len(cases)
assert [x["case"] for x in new_records] == cases
assert H((D / "publication-binding-reference032.json").read_bytes()) == binding["prior_binding_sha256"]

selection_path = Q / "publication033-case-selection.json"
selection_bytes = selection_path.read_bytes()
selection = json.loads(selection_bytes)
assert H(selection_bytes) == binding["case_selection_sha256"]
assert selection["cases"] == cases
assert selection["active521_excluded"] is True
assert selection["source199_ongoing_excluded"] is True
assert "solver/fresh-damped-cgls-complete-coupon-521" not in cases
assert "analysis/native521-fresh-damped-coupon-command-002" not in cases
assert not any(c.startswith("native/") and "521" in c for c in cases)
assert not any(c.startswith("solver/") and "521" in c for c in cases)
# Historical command and correction preflights may be retained; the active mutable run and live output are excluded.
historical_521_preflights = [c for c in cases if c.startswith("analysis/native521-")]
assert historical_521_preflights == [
    "analysis/native521-fresh-damped-coupon-command-001",
    "analysis/native521-fresh-damped-coupon-command-002-integrity-before-001",
    "analysis/native521-fresh-damped-coupon-command-002-integrity-correction-001",
    "analysis/native521-question-label-correction-001",
]

raw_count = gzip_count = original_bytes = stored_bytes = 0
expected_checkout_paths = set(binding["changed_source_files"])
case_file_count = 0
for rec in new_records:
    case = rec["case"]
    case_root = (Q / case).resolve()
    assert case_root.is_dir()
    for row in rec["files"]:
        case_file_count += 1
        assert row.get("physical_box_qualification") is not True
        raw_key = "repository_copy" if "repository_copy" in row else "repository_compressed_copy"
        payload_rel = row[raw_key]
        published_path = (D / payload_rel)
        assert published_path.is_file() and not published_path.is_symlink()
        packed = published_path.read_bytes()
        stored_bytes += len(packed)
        if raw_key == "repository_copy":
            original = packed
            raw_count += 1
            assert row["bytes"] == len(packed)
        else:
            gzip_count += 1
            assert H(packed) == row["compressed_sha256"]
            assert len(packed) == row["compressed_bytes"]
            original = gzip.decompress(packed)
        assert len(original) == row["bytes"]
        assert H(original) == row["sha256"]
        external = Path(row["external_path"]).resolve()
        external.relative_to(case_root)
        assert external.is_file() and not external.is_symlink()
        assert external.read_bytes() == original
        original_bytes += len(original)
        expected_checkout_paths.add(P + payload_rel)
assert case_file_count == binding["copied_files"] == 1615
assert raw_count == binding["raw_files"] == 1449
assert gzip_count == binding["gzip_files"] == 166
assert original_bytes == binding["original_payload_bytes"] == 668254149
assert stored_bytes == binding["stored_payload_bytes"] == 123637754
assert not any("solver/fresh-damped-cgls-complete-coupon-521" in p for p in expected_checkout_paths)
assert not any("native/fresh-damped" in p and "521" in p for p in expected_checkout_paths)

# Source198 source map, publication copy, and exact build artifacts.
S = Q / "snapshots/source-contact-damped-memory-198-001"
frozen_path = S / "frozen-source.json"
frozen_bytes = frozen_path.read_bytes()
assert H(frozen_bytes) == binding["source198_frozen_sha256"] == "16474dca955f9a953f78cf211054c10967c8d31cd5daa8bf5d8476741ffe8cfc"
frozen = json.loads(frozen_bytes)
source_map = frozen["source_sha256"]
assert len(source_map) == binding["selected_source_files"] == 816
for rel, digest in source_map.items():
    assert filehash(R / rel) == digest, rel
    assert filehash(S / rel) == digest, rel
source_changes = sorted(set(binding["changed_source_files"]))
assert len(source_changes) == 12
actual_source_changes = set(x.decode() for x in git("diff", "--name-only", "-z", "HEAD", "--").split(b"\0") if x)
actual_source_changes |= set(x.decode() for x in git("ls-files", "--others", "-z", "--").split(b"\0") if x)
actual_source_changes = {p for p in actual_source_changes if p in source_map}
assert actual_source_changes == set(source_changes), (actual_source_changes, source_changes)

build_path = Q / "build-source198-001/source-build.json"
build_bytes = build_path.read_bytes()
assert H(build_bytes) == binding["source198_host_build_sha256"] == "8f6c95304c23b6c7149ec6b270d54eafdf4507ea26f3420e39e523c09dd500ad"
build = json.loads(build_bytes)
assert build["success"] is True and build["source_sha256"] == source_map
for rel, digest in build["artifacts_sha256"].items():
    assert filehash(rel) == digest, rel
for rel, digest in build["build_metadata_sha256"].items():
    assert filehash(Q / "build-source198-001" / rel) == digest, rel
assert filehash(Q / "build-source198-001/numi-matter-cardboard-full-contact-armijo-check") == "c067e5ea69b5fb748a878be24151f2f27b2de7b73701dd20df61a57a37e0bda6"
assert filehash(Q / "build-source198-001/shaders/NumiMatter.metallib") == "edd02f04b12713e735a8a7d364b696893f74f5693e2a7fef3566bbe8da11c21f"
assert filehash(R / "matter/examples/cardboard/materials/nagasawa2013_medium_orthotropic.nmatter") == "ce39e29918c0111ab1e25e9af8a69eb0e3f7f6f07d1227bfd443da735d9ea348"
assert len(build["artifacts_sha256"]) == 11

# CPU receipts are successful, but remain implementation checks.
cpu_dir = Q / "analysis/source198-cpu-validation-001"
cpu_receipts = {}
for name, expected_hash in [
    ("cpu-checks", "40fc8b5d1f02e01265b6f1dc413d8f57f080d2d31b0e61175e6ee4b036013e38"),
    ("solver-environment", "e702bec103e4730f92e36ac0c2b48f8ef0d5d8b7efc6f484097a395cc13fc0112"),
    ("damped-cgls-oracle", "4fad561e50213823c93af0f5ac97e0cf1850a7280e97e4392dd8a9f2997d545b"),
]:
    receipt = cpu_dir / name / "receipt.json"
    assert H(receipt.read_bytes()) == expected_hash, name
    r = json.loads(receipt.read_text())
    assert r["exit_code"] == 0, (name, r)
    cpu_receipts[name] = r["exit_code"]

# Private CGLS helper: both optimization levels, no physical FEM transactions.
helper = Q / "native/damped-cgls-kernel-replay-523"
for opt in ["O0", "O3"]:
    receipt = json.loads((helper / opt / "receipt.json").read_text())
    assert receipt["exit_code"] == 0 and receipt["physical_transactions"] == 0
assert (helper / "O0/stdout.json").read_bytes() == (helper / "O3/stdout.json").read_bytes()
h = json.loads((helper / "O0/stdout.json").read_text())
assert len(h["cases"]) == 8
assert all(c["status"] == "breakdown" and c["error_bits"] == 32 and c["iterations"] == 6
           for c in h["cases"] if c["mu_kg"] == 0)
assert all(c["status"] == "iteration_cap" and c["error_bits"] == 0 and
           c["iterations"] == c["requested_cap"]
           for c in h["cases"] if c["mu_kg"] > 0)

# Contact transaction limits and rollback. Regional and scalar runs are fixtures, not a complete coupon.
status = {}
for number, dirname, expected_code, expected_index in [
    ("526", "integrated198-contact-regional-526", 1, 0),
    ("527", "integrated198-contact-restart16-527", 10, 18),
]:
    root = Q / "native" / dirname
    tx = [json.loads((root / "baseline" / f"transaction-{i:06d}.json").read_text()) for i in range(3)]
    assert [x["accepted"] for x in tx] == [True, True, False]
    assert tx[0]["after"]["control_step"] == 1 and tx[1]["after"]["control_step"] == 2
    assert tx[2]["before"]["control_step"] == tx[2]["after"]["control_step"] == 2
    assert tx[2]["after"]["solver_certificates"][0]["validity"][3] == 0
    assert tx[2]["before"]["accepted_state_vector_sha256"] == tx[2]["after"]["accepted_state_vector_sha256"]
    s = tx[2]["after"]["statuses"][0]
    assert s["code"] == expected_code and s["failing_index"] == expected_index
    status[number] = {"code": s["code"], "failing_index": s["failing_index"]}
    if number == "526":
        assert tx[2]["before"]["solver_certificates"][0]["nonlinear"][0] == 9.63568425e-8
        assert s["diagnostics"] == [-3, -3, -3, -3]
    else:
        assert tx[2]["before"]["statuses"][0]["code"] == 0
        assert tx[2]["after"]["statuses"][0]["diagnostics"][0] > 1e-7
    baseline = {str(p.relative_to(root / "baseline")): p.read_bytes()
                for p in (root / "baseline").rglob("*") if p.is_file()}
    aliased = {str(p.relative_to(root / "aliased")): p.read_bytes()
               for p in (root / "aliased").rglob("*") if p.is_file()}
    assert len(baseline) == len(aliased) == 13 and baseline == aliased
    if number == "527":
        older = Q / "native/integrated195-contact-restart16-525"
        old_arm = {str(p.relative_to(older / "baseline")): p.read_bytes()
                   for p in (older / "baseline").rglob("*") if p.is_file()}
        assert len(old_arm) == len(baseline) == 13 and old_arm == baseline

# Six axes remain unqualified and binding copies the summary verdicts exactly.
assert set(summary["verdicts"]) == {
    "solver_correctness", "numerical_convergence", "component_calibration",
    "analogue_validation", "physical_assembly_in_simulation", "crush_impact_prediction"
}
assert binding["six_axis_verdicts"] == summary["verdicts"]
positive_statuses = {"pass", "qualified", "complete", "completed"}
assert all(v.get("status") not in positive_statuses for v in summary["verdicts"].values())
assert all(binding["six_axis_verdicts"][k]["status"] == v["status"]
           for k, v in summary["verdicts"].items())
assert binding["physical_validation"] is False
assert summary["historical_summary_snapshots"]["reference032"]["sha256"] == H(head_summary)

# Reconstruct the intended exact staging set and compare to the actual checkout.
for rel in ["summary.json", "summary-reference032.json", "publication-binding-reference033.json"]:
    expected_checkout_paths.add(P + rel)
expected_checkout_paths.update([
    "matter/examples/cardboard/solver-milestone-source198.md",
    "matter/examples/cardboard/implementation-20261014.md",
])
pathlist = (Q / "publication033-staged-paths.nul").read_bytes()
paths_list = [x.decode() for x in pathlist.split(b"\0") if x]
assert len(paths_list) == 1632 and len(paths_list) == len(set(paths_list))
assert paths_list == sorted(paths_list)
assert set(paths_list) == expected_checkout_paths
actual = {x.decode() for x in git("diff", "--name-only", "-z") if x}
actual |= {x.decode() for x in git("ls-files", "--others", "-z") if x}
assert actual == expected_checkout_paths, (len(actual), len(expected_checkout_paths), sorted(actual-expected_checkout_paths)[:5], sorted(expected_checkout_paths-actual)[:5])
assert not git("diff", "--cached", "--name-only", "-z")
assert not git("diff", "--check")

# Public-facing source198 milestone is inspected for explicit bounds.
milestone = (R / "matter/examples/cardboard/solver-milestone-source198.md").read_text()
assert "native521" in milestone and "excluded while active" in milestone
assert "All six overall axes remain open" in milestone
assert "does not establish" in milestone or "No complete coupon" in milestone
impl = (R / "matter/examples/cardboard/implementation-20261014.md").read_text()
assert "Source198" in impl and "remain open" in impl

report = {
    "schema": "numi.cardboard.publication033-independent-review.v1",
    "verdict": "PASS",
    "head": git("rev-parse", "HEAD").decode().strip(),
    "binding_sha256": H(binding_bytes),
    "summary_sha256": H(summary_bytes),
    "historical_records_byte_identical": 1669,
    "new_case_records": len(new_records),
    "payloads_verified": case_file_count,
    "raw_payloads": raw_count,
    "compressed_payloads": gzip_count,
    "decompressed_original_bytes": original_bytes,
    "stored_payload_bytes": stored_bytes,
    "source198_frozen_sha256": H(frozen_bytes),
    "source198_files_verified": len(source_map),
    "source198_changed_checkout_files": len(source_changes),
    "build_artifacts_verified": len(build["artifacts_sha256"]),
    "cpu_receipts_exit0": cpu_receipts,
    "native523": {
        "O0_O3_equal": True, "physical_transactions_each": 0,
        "undamped": "breakdown at iteration 6, error bit 32",
        "damped": "iteration cap 32/64, not convergence"
    },
    "native526_527": status,
    "active_native521_outputs_included": False,
    "historical_native521_preflight_cases_included": historical_521_preflights,
    "six_axes_open": True,
    "staged_paths": len(paths_list),
    "staged_pathlist_sha256": H(pathlist),
    "checkout_diff_matches_staged_paths": True,
    "physical_validation": False,
    "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
}
report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
(O / "report.json").write_bytes(report_bytes)
script_path = Path(__file__)
(O / "audit.py").write_bytes(script_path.read_bytes())
manifest = {
    "schema": "numi.cardboard.publication033-independent-review.manifest.v1",
    "inputs": {
        "binding_sha256": H(binding_bytes),
        "summary_sha256": H(summary_bytes),
        "case_selection_sha256": H(selection_bytes),
        "source198_frozen_sha256": H(frozen_bytes),
        "source198_build_sha256": H(build_bytes),
        "staged_pathlist_sha256": H(pathlist)
    },
    "outputs": {}
}
for name in ["audit.py", "report.json"]:
    data = (O / name).read_bytes()
    manifest["outputs"][name] = {"bytes": len(data), "sha256": H(data)}
manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
(O / "manifest.json").write_bytes(manifest_bytes)
print(json.dumps({"report": report, "manifest_sha256": H(manifest_bytes)}, indent=2))

