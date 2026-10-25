from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path('/Users/numi/Work/numi-human-connected-surface-069/D/coherent-support-continuation-346/run_coherent_target_continuation_346.py')
spec = importlib.util.spec_from_file_location('coherent_target_346_tested', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

@pytest.fixture(scope='module')
def context():
    state = mod.read(mod.DEFAULT_STATE)
    candidate = np.load(state['candidate'], allow_pickle=False)
    faces = np.asarray(candidate['source_O_faces'], dtype=np.int64)
    archive, archive_receipt = mod.seed_archive_index()
    return faces, archive, archive_receipt

def test_337_retains_every_complete_exact_audit(context):
    faces, archive, _ = context
    root = mod.FOURTH_SEED_ROOT
    expected_files = sorted(root.rglob('surface-self-pairs.npy'))
    pairs, receipts = mod.collect_pairs(root, faces, archive)
    assert len(expected_files) == len(receipts) == 5
    assert len(pairs) == 7837
    assert all(x['candidate']['mode'] == 'local_candidate_bytes_verified' for x in receipts)
    assert all(x['topology_dimensions'] == {'points': 20889, 'faces': len(faces)} for x in receipts)

def test_union_includes_all_four_pinned_seed_roots(context):
    faces, archive, _ = context
    pairs, receipts = mod.collect_seed_pairs(faces, archive)
    roots = {Path(x['audit_root']) for x in receipts}
    assert roots == {x.resolve() for x in (mod.SEED_ROOT, mod.EXTRA_SEED_ROOT, mod.ADDITIONAL_SEED_ROOT, mod.FOURTH_SEED_ROOT)}
    assert len(receipts) == 39
    assert len(pairs) == 11321
    assert mod.FOURTH_SEED_ROOT.resolve() in roots

def test_337_candidate_topology_mismatch_fails_closed(context):
    faces, archive, _ = context
    wrong = faces.copy()
    wrong[0, [0, 1]] = wrong[0, [1, 0]]
    with pytest.raises(ValueError, match='topology mismatch'):
        mod.collect_pairs(mod.FOURTH_SEED_ROOT, wrong, archive)


def test_initial_trial_gradient_guard_reduces_unsafe_f32_emission():
    base = np.array([[1., 0., 0.], [1.01, 0., 0.], [1., 1., 0.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    scales, emitted, receipt = mod.emit_gradient_checked_trial(
        base, faces, np.array([1], dtype=np.int64), np.array([[0.02, 0., 0.]]),
        np.array([0], dtype=np.int64))
    assert emitted is not None and receipt['candidate_usable'] is True
    assert scales[1] < 1.0
    assert receipt['edge_constraints_checked_after_float32_emission'] is True
    edges = mod.load_module(mod.LOCAL_HELPER_339, 'edge_gradient_test_339').mesh_unique_edges(faces)
    moved = emitted.astype(np.float64) - base.astype(np.float64)
    lengths = np.linalg.norm(base[edges[:, 0]].astype(np.float64) - base[edges[:, 1]].astype(np.float64), axis=1)
    ratios = np.max(np.abs(moved[edges[:, 0]] - moved[edges[:, 1]]), axis=1) / lengths
    assert np.max(ratios) <= 0.5


def test_initial_trial_gradient_guard_preserves_passing_raw_f32_bytes():
    base = np.array([[1., 0., 0.], [1.01, 0., 0.], [1., 1., 0.]], dtype=np.float32)
    raw = base.copy()
    raw[1, 1] = np.float32(0.0005)
    displacement = raw[[1]].astype(np.float64) - base[[1]].astype(np.float64)
    scales, emitted, receipt = mod.emit_gradient_checked_trial(
        base, np.array([[0, 1, 2]], dtype=np.int64), np.array([1], dtype=np.int64),
        displacement, np.array([0], dtype=np.int64))
    assert receipt['candidate_usable'] is True
    assert np.all(scales == 1.0)
    assert np.array_equal(emitted.view(np.uint32), raw.view(np.uint32))


def _fitter_report(*, fallback=False):
    solver = {
        "success": True,
        "fallback_used": fallback,
        "stage1": {"success": True, "status": 0},
        "stage2": {"success": not fallback, "status": 1 if fallback else 0,
                   "time_limit": fallback, "message": "Time limit reached" if fallback else "Optimal"},
    }
    if fallback:
        solver["solution_source"] = "certified_stage1_primal_after_stage2_time_limit"
        solver["stage1_primal_certificate"] = {
            "passed": True, "status": "CERTIFIED_STAGE1_PRIMAL"
        }
    return {
        "emitted_clearance_proxy_gate_deferred": True,
        "stage1_feasible_timeout_fallback_opt_in": True,
        "solver": solver,
    }


def test_fitter_optins_accept_normal_lexicographic_solution():
    mod.validate_fitter_optins(_fitter_report(), require_solution=True)


def test_fitter_optins_accept_only_certified_stage1_timeout_fallback():
    mod.validate_fitter_optins(_fitter_report(fallback=True), require_solution=True)


@pytest.mark.parametrize("field", [
    "emitted_clearance_proxy_gate_deferred",
    "stage1_feasible_timeout_fallback_opt_in",
])
def test_fitter_optins_reject_missing_optin_receipt(field):
    report = _fitter_report()
    report[field] = False
    with pytest.raises(ValueError):
        mod.validate_fitter_optins(report, require_solution=True)


def test_fitter_optins_reject_uncertified_timeout_fallback():
    report = _fitter_report(fallback=True)
    report["solver"]["stage1_primal_certificate"]["passed"] = False
    with pytest.raises(ValueError, match="certified stage-1"):
        mod.validate_fitter_optins(report, require_solution=True)


def test_lp_no_proposal_report_without_candidate_optin_markers_is_retained():
    report = {
        "status": "LP_NO_PROPOSAL",
        "classification": "construction-only bounded area SCP; no candidate admitted",
        "native_admission": False,
        "accepted_physical_steps": 0,
        "solver": {"success": False, "fallback_used": False},
    }
    mod.validate_fitter_optins(report, require_solution=False)


def test_candidate_report_still_requires_both_optin_markers():
    report = _fitter_report()
    report["emitted_clearance_proxy_gate_deferred"] = False
    with pytest.raises(ValueError, match="proxy deferral"):
        mod.validate_fitter_optins(report, require_solution=True)

def test_resume_requires_explicit_sha_for_nondefault_state(tmp_path):
    state = tmp_path / "selected.json"
    state.write_text('{"status":"CONSTRUCTION_PROGRESS_ONLY"}')
    with pytest.raises(ValueError, match="requires --resume-state-sha256"):
        mod.validate_state(state)
    with pytest.raises(ValueError, match="explicit --resume-state-sha256"):
        mod.validate_state(state, resume_state_sha256="0" * 64)

def test_resume_validation_accepts_complete_pinned_initial_state():
    state = mod.read(mod.DEFAULT_STATE)
    point = Path(state["point_summary"])
    surface = Path(state["surface_report"])
    validated, candidate, faces, anchors = mod.validate_state(
        mod.DEFAULT_STATE,
        resume_state_sha256=mod.sha(mod.DEFAULT_STATE),
        resume_point_summary_sha256=mod.sha(point),
        resume_surface_report_sha256=mod.sha(surface),
    )
    assert validated["_resume_verified_hashes"]["candidate_sha256"] == mod.sha(candidate)
    assert validated["_resume_verified_hashes"]["surface_report_sha256"] == mod.sha(surface)
    assert faces.shape == (mod.EXPECTED_FACES, 3)
    assert len(anchors) == len(np.unique(anchors))

def test_resume_rejects_wrong_report_pin():
    state = mod.read(mod.DEFAULT_STATE)
    with pytest.raises(ValueError, match="explicit resume point summary SHA"):
        mod.validate_state(
            mod.DEFAULT_STATE,
            resume_state_sha256=mod.sha(mod.DEFAULT_STATE),
            resume_point_summary_sha256="0" * 64,
            resume_surface_report_sha256=mod.sha(Path(state["surface_report"])),
        )

def test_seed_roots_are_repeatable_and_checked_for_overlap(tmp_path, monkeypatch):
    h = tmp_path / "evidence"; h.mkdir()
    roots = [h / f"fixed-{i}" for i in range(4)]
    for root in roots: root.mkdir()
    monkeypatch.setattr(mod, "H", h)
    monkeypatch.setattr(mod, "SEED_ROOT", roots[0])
    monkeypatch.setattr(mod, "EXTRA_SEED_ROOT", roots[1])
    monkeypatch.setattr(mod, "ADDITIONAL_SEED_ROOT", roots[2])
    monkeypatch.setattr(mod, "FOURTH_SEED_ROOT", roots[3])
    mod.EXTRA_SEED_AUDIT_ROOTS = []
    extra1 = h / "coherent-continuation-344"; extra1.mkdir()
    extra2 = h / "later-run"; extra2.mkdir()
    all_roots = mod.configure_seed_audit_roots([extra1, extra2])
    assert all_roots == (*roots, extra1.resolve(), extra2.resolve())
    assert mod.configure_seed_audit_roots([]) == tuple(roots)
    with pytest.raises(ValueError, match="overlapping"):
        mod.configure_seed_audit_roots([h])
    with pytest.raises(ValueError, match="escapes"):
        outside = tmp_path / "outside"; outside.mkdir()
        mod.configure_seed_audit_roots([outside])

def test_seed_root_rejects_duplicate_explicit_roots(tmp_path, monkeypatch):
    h=tmp_path/"evidence";h.mkdir()
    fixed=[h/f"fixed-{i}" for i in range(4)]
    for x in fixed:x.mkdir()
    monkeypatch.setattr(mod,"H",h)
    monkeypatch.setattr(mod,"SEED_ROOT",fixed[0]);monkeypatch.setattr(mod,"EXTRA_SEED_ROOT",fixed[1])
    monkeypatch.setattr(mod,"ADDITIONAL_SEED_ROOT",fixed[2]);monkeypatch.setattr(mod,"FOURTH_SEED_ROOT",fixed[3])
    mod.EXTRA_SEED_AUDIT_ROOTS=[]
    extra=h/"new";extra.mkdir()
    with pytest.raises(ValueError,match="duplicate"):
        mod.configure_seed_audit_roots([extra,extra])

def test_core_provenance_comparison_uses_exact_bytes_with_nan_payloads():
    base={"files":["points_world_m","node_parameter","faces"],
          "points_world_m":np.zeros((1,3),np.float32),
          "node_parameter":np.array([np.nan],dtype=np.float64),
          "faces":np.array([[0,0,0]],dtype=np.int64)}
    same={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()}
    same["points_world_m"][0,0]=1
    assert mod.candidate_core_array_mismatches(base,same)==[]
    altered={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in same.items()}
    altered["faces"][0,0]=1
    assert mod.candidate_core_array_mismatches(base,altered)==["faces"]

def test_lp_method_pin_and_supported_methods():
    assert mod.sha(mod.SOLVER)==mod.PINS["solver_347_sha256"]
    assert mod.LP_METHODS==("highs","highs-ipm","highs-ds")
