import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "run_quality_policy025_variant_v4.py"
POLICY_PATH = HERE / "quality_policy025_variant.py"
OWNER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/focused-clearance-541/topology/repair_focused_clearance_541.py")

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_five_argument_owner_parity_binding_matches_local_and_roundtrips_json(tmp_path):
    runner = load("runner_test_v4", RUNNER_PATH)
    policy = load("policy_test_v4", POLICY_PATH)
    owner = load("owner_test_v4", OWNER_PATH)
    runner.bind_parity_owner(owner, policy)

    points_mm = np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    faces = np.asarray([[0,1,2],[1,3,2]], dtype=np.int64)
    candidate = {"source_O_faces": faces, "area_reference_037_source_O_faces": faces.copy()}
    quality = owner.face_quality(points_mm, points_mm, faces, faces)
    op = {"face_rows": [0,1], "new_faces": faces.tolist()}

    local, local_reason = policy.evaluate_op_local(owner, candidate, op, quality, points_mm, points_mm)
    # This is the exact 5-argument call site used by the topology owner.
    parity, parity_reason = owner.evaluate_op(candidate, op, quality, points_mm, points_mm)
    assert local is not None and parity is not None
    assert local_reason == parity_reason == "eligible"
    assert local["score"] == parity["score"]
    assert local["deltas"] == parity["deltas"]

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"status":"construction-only"}) + "\n")
    annotated = runner.annotate_report(report_path)
    raw = report_path.read_bytes()
    assert json.loads(raw) == annotated
    assert raw.endswith(b"\n") and not raw.endswith(b"\\n")


def test_chain_cli_parses_real_flags_then_fails_closed_on_bad_candidate_hash(tmp_path):
    root = Path("/Users/numi/Work/numi-human-connected-surface-069")
    base = root / "D/surface-clearance-546/local-witness-retention-567/retained-552-001"
    candidate = base / "candidate.npz"
    audit159 = base / "full-check/audit-159/report.json"
    audit130 = base / "full-check/audit-130/summary.json"
    self_pairs = base / "full-check/audit-130/surface-self-pairs.npy"
    output = tmp_path / "parsed-chain-output"
    cmd = [sys.executable, str(RUNNER_PATH), "--chain",
        "--candidate", str(candidate), "--audit159", str(audit159),
        "--audit130", str(audit130), "--self-pairs130", str(self_pairs),
        "--output-root", str(output), "--expected-candidate-sha256", "0"*64,
        "--expected-audit159-sha256", "7d6818776a047ffc3e272f3721348fb06f8527adf1f6bba07a5439c01fc7e00e",
        "--expected-audit130-sha256", "251dd08c4503e8c6723ce84fa3d7507c69c2b0b2e2f61947641d140e73240591",
        "--expected-self-pairs130-sha256", "55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de",
        "--rounds", "3", "--max-candidates", "2048", "--max-accept", "64", "--prefilter-factor", "1"]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "candidate input SHA mismatch" in result.stderr
    assert output.is_dir()
    assert not list(output.glob("round-*") )
