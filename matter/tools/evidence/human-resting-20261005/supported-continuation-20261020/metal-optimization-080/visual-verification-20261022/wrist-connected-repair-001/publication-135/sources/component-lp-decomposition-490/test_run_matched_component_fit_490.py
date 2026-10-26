import copy
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = Path("/Users/numi/Work/numi-human-connected-surface-069/D/D/component-lp-decomposition-490/run_matched_component_fit_490.py")
spec = importlib.util.spec_from_file_location("matched_fit_490_tested", SOURCE)
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
INPUTS = {
    "candidate": D / "D/outward-plane-backtrack-461/run-001/backtrack-373/pass-011/candidate.npz",
    "audit159": D / "D/outward-plane-backtrack-461/run-001/backtrack-373/pass-011/audit-159/report.json",
    "face_target_npz": D / "D/connected-continuation-prep476/run-001/surface-targets-335/targets.npz",
    "point_target_npz": D / "D/connected-continuation-prep476/run-001/point-targets-390/nearest-union-exits.npz",
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make_config(path):
    inputs = {role: {"path": str(p), "sha256": sha(p)} for role, p in INPUTS.items()}
    sources = {role: {"path": str(source.resolve()), "sha256": digest}
               for role, (source, digest) in mod.MATCHED_SOURCE_PINS.items()}
    config = {"schema": mod.SCHEMA, "inputs": inputs, "source_pins": sources}
    path.write_text(json.dumps(config, sort_keys=True))
    return config

def test_current_476_exact_matched_inputs_validate(tmp_path):
    config_path = tmp_path / "matched.json"
    config = make_config(config_path)
    checked = mod.validate_matched_inputs(config_path, INPUTS)
    assert checked["inputs"]["candidate"]["sha256"] == config["inputs"]["candidate"]["sha256"]
    assert checked["inputs"]["face_target_npz"]["sha256"] == config["inputs"]["face_target_npz"]["sha256"]
    assert set(checked["source_pins"]) == {"runner458", "owner408", "objective458", "decomposition475"}
    assert len(mod.validate_source_lock()) >= 10

def test_stale_input_hash_is_rejected(tmp_path):
    config_path = tmp_path / "matched.json"
    config = make_config(config_path)
    config["inputs"]["point_target_npz"]["sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    import pytest
    with pytest.raises(ValueError, match="file/hash mismatch"):
        mod.validate_matched_inputs(config_path, INPUTS)

def test_wrong_cli_input_path_is_rejected(tmp_path):
    config_path = tmp_path / "matched.json"
    make_config(config_path)
    actual = dict(INPUTS)
    actual["candidate"] = D / "D/outward-plane-backtrack-461/run-001/backtrack-373/pass-010/candidate.npz"
    import pytest
    with pytest.raises(ValueError, match="path does not match"):
        mod.validate_matched_inputs(config_path, actual)

def test_source_pin_drift_is_rejected(tmp_path):
    config_path = tmp_path / "matched.json"
    config = make_config(config_path)
    config["source_pins"]["owner408"]["sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    import pytest
    with pytest.raises(ValueError, match="source pin owner408 hash mismatch"):
        mod.validate_matched_inputs(config_path, INPUTS)

def test_non475_schema_is_rejected(tmp_path):
    config_path = tmp_path / "matched.json"
    config = make_config(config_path)
    config["schema"] = "unknown"
    config_path.write_text(json.dumps(config))
    import pytest
    with pytest.raises(ValueError, match="unknown matched-input schema"):
        mod.validate_matched_inputs(config_path, INPUTS)

def test_490_options_are_removed_without_changing_fit_paths(tmp_path):
    config_path = tmp_path / "matched.json"
    audit = INPUTS["audit159"]
    argv = ["--matched-inputs-json-490", str(config_path), "--audit159-490", str(audit),
            "--base-candidate", str(INPUTS["candidate"]), "--face-target-npz",
            str(INPUTS["face_target_npz"]), "--point-target-npz",
            str(INPUTS["point_target_npz"]), "--output", str(tmp_path / "out"),
            "--allow-soft-target-regression-485"]
    remaining, parsed_config, output, actual = mod.extract_fit_inputs(argv)
    assert parsed_config == config_path
    assert output == tmp_path / "out"
    assert actual == {"candidate": INPUTS["candidate"], "audit159": audit,
                      "face_target_npz": INPUTS["face_target_npz"],
                      "point_target_npz": INPUTS["point_target_npz"]}
    assert "--matched-inputs-json-490" not in remaining
    assert "--audit159-490" not in remaining
    assert "--allow-soft-target-regression-485" in remaining
