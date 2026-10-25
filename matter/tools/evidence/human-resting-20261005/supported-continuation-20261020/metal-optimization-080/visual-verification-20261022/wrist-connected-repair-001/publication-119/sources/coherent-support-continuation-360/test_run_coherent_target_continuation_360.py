from pathlib import Path
import importlib.util

RUNNER = Path(__file__).with_name("run_coherent_target_continuation_360.py")
spec = importlib.util.spec_from_file_location("runner360_tests", RUNNER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_real_357_actual491_preflight_is_exact_322_and_185_complete():
    state = mod.read(mod.DEFAULT_STATE)
    candidate = Path(state["candidate"])
    result = mod.validate_point_depth_preflight(
        candidate, mod.POINT_DEPTH_ROWS_NPZ, mod.POINT_DEPTH_ROWS_REPORT, mod.POINT_DEPTH_OWNER)
    assert result["candidate_sha256"] == mod.PINS["283_candidate_sha256"]
    assert result["row_count"] == result["unique_node_count"] == result["schema_row_count"] == 491
    assert result["inside_row_count"] == 489
    assert result["outside_row_count"] == 2
    assert result["bank_count"] == 185
    assert result["complete_bank_accounting_verified"] is True


def test_additional_exact_self_audit_roots_are_in_configured_inventory():
    roots = mod.configured_seed_roots()
    names = {p.name for p in roots}
    assert {"coherent-support-continuation-344", "coherent-support-continuation-348",
            "coherent-support-continuation-352"} <= names
    assert len(roots) == len(set(map(lambda p: p.resolve(), roots)))
