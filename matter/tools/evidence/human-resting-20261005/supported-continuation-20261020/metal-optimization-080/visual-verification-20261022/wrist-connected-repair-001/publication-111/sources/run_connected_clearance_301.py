#!/usr/bin/env python3
"""Three-round fixed-prefix continuation on the exact refined-283 topology.

Builds proposals only. Geometry subprocesses use the shared slot wrapper;
native and physical admission are outside this runner.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np

D = Path('/Users/n/Work/numi-human-connected-surface-069')
H = D / 'barycentric-connected-clearance-170'
PYTHON = Path('/Users/n/numi-human-prep-venv-20261005/bin/python')
SLOT = Path('/tmp/numi-human-geometry-slot.py')
SEED_ROOT = H / 'fixed-prefix-continuation-291'
DEFAULT_STATE = SEED_ROOT / 'selected-result.json'
AREA_REFERENCE = H / 'connected-refined-278/candidate.npz'
SOURCE_BASE = D / 'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
SOLVER = D / 'connected-local-witness-proposal-279/propose_joint_clearance_witnesses_279.py'
FACE_BUILDER = H / 'source-outward-proposal-300/build_outward_nearest_union_targets_303.py'
FACE_CHECKER = H / 'full-face-target-preflight-302/full_face_target_preflight_302.py'
INTERVAL_SELECTOR = H / 'source-outward-proposal-300/first_representable_union_target_300.py'
SEED_ARCHIVE_MANIFEST = D / 'publication-110/evidence/archive-manifest.json'
SEED_ARCHIVE = D / 'publication-110/evidence/connected-trials-291-299.tar.xz'
SEED_ARCHIVE_REMOTE_RECEIPT = D / 'publication-110/remote-recovery-verification.json'
SEED_ARCHIVE_LOCAL_RECEIPT = D / 'publication-110/verified-local-retirement.json'
SEED_ARCHIVE_MANIFEST_SHA256 = '587d8ed5356063edc4eebe52544b43c2616857d8edd6824d6ffb92277b72e724'
SEED_ARCHIVE_SHA256 = 'b089b7d9cdc5bd1aa89cc6748abf4fb42aa6b2fb50924421567e8393488dd222'
SEED_SURFACE_PREDICATE_SHA256 = 'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
CACHED_293_REPORT_SHA256 = 'c40201ea6adee235475d76ac9fd3d65dce7e809c31b6101b2e7906f27fbf4fb4'
CACHED_293_NPZ_SHA256 = '2c02c3acfdc61c5fa8754a1c4a2d6033f40cb3944af4ada15f4617c11001f306'
CACHED_293_DIR = H / 'outward-nearest-targets-293'
GLOBAL_HELPER = H / 'global_step_backtracking_275.py'
LOCAL_HELPER = D / 'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
BACKTRACKER = H / 'run_local_witness_backtracking_287.py'
PREP185 = H / 'prepare_refined_proposal_185.py'
POINT_EMITTER = D / 'connected-local-witness-proposal-235/emit_empty_point_targets_235.py'
PAIR_OWNER = H / 'connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py'
AREA_OWNER = H / 'connected-local-continuation-189-run001/round-005/proposal/round-001/fit/area_scp_support_023.py'
BARY_OWNER = D / 'publication-091/canonical/matter/tools/human_skin_fat_barycentric_constraints.py'
AUDIT130 = D / 'audit_connected_outer_geometry_130.py'
AUDIT159 = D / 'audit_skin_bone_surfaces_159.py'

PINS = {
    'face_checker_302_sha256': 'caa94e5df1d7363ac3fbdb1d66ed943a806a08bb132a9b49ebeb2236d59d7d50',
    'interval_selector_300_sha256': '5e613e21b7bdebcc036dc9712ad21903df7d081de32bfdb83fae0a0c344ac094',
    '283_state_sha256': '52fc2d9b80e17f25274168ecac0b5072385367b76fc2a1c5bbbeec57b7a51d97',
    '283_candidate_sha256': '4228f95305aae5bf5477264f976ee53ace3850c043b1b37d530b6b2ecd0fe7f8',
    '283_point_summary_sha256': 'ba78d06d1b4015c1308ef45a8a0d6e274ec727a37a2583eba07bc4e3a7e0d0f1',
    '283_surface_report_sha256': 'c13881c681bf37e4b7c9551b3155b4fb7da7ac06336dacd571a4e2576d69012a',
    'area_reference_sha256': 'c2a076267c2a8261a5f1a51b742a0b4e44471d31b200f5b809b8c98d6205348f',
    'source_base_sha256': '70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31',
    'solver_279_sha256': 'fdfd30358d139607feb7ad8416ab307d12cf0ac9e54accdd3cdc7609454693be',
    'face_builder_292_sha256': 'c8fe5b7be2efb2c6867a2cde178ebc35280814c37f90eec5aea47e964fc17a1f',
    'global_backtracker_275_sha256': '18891b1f700592563e139edfc8bd1361d230ff6fd5af2ae5ee036c9625c1826a',
    'local_backtracker_287_sha256': 'b1c7e0ca0f2704a1c4628d5c61816e8131fe33c78eabaaac697dd6f717a82d12',
    'local_step_helper_183_sha256': '05b1beb2105a184deb885974db892fa0f8e519fc3e1fdfde5aa441c180c97b34',
    'prepare_refined_185_sha256': '81d749a969d7a2e659ce1d3d9766eea5fa64909e2405a7fe303d2bb10ab15b3b',
    'point_emitter_235_sha256': '4367ecca6a2506d61d83a76ede4ea6449606c1b792fbb45fab9182a4a5a1fd43',
    'pair_owner_sha256': 'c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3',
    'area_owner_sha256': 'aaa9dbe32fbfeb71818a2afd673ef4eb7dfc0c209cee6d0ee1dbff1a6ce71abd',
    'bary_owner_sha256': '965a166e86952d79c019302c98e7684f2f8ef3a9d63b328430ac9c8c8edbaff2',
    'audit130_sha256': '52d45f287162d22dca417793c1393bcfbbe506c5ce85b828f009b4fd7ccd94b9',
    'audit159_sha256': 'de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1',
    'surface_damping_258_sha256': '5099756b1244845be2fb61a476c15d3a2c6095e41b3e41f08a9517148ba7764d',
    'point_damping_254_sha256': 'f78831353656e6b7e5bfee6d0edca89c876dac25ccd85eeaf522c350b58dd894',
    'merit_owner_175_sha256': '9d8c4d7bc8397a07192b395f0ed42cb8cf60f15ef00cfe7fd9f48eeb17133dba',
}
FIXED_PREFIX = 18430
AREA_POINTS_KEY = 'area_reference_037_points_world_m_f64'
AREA_FACES_KEY = 'area_reference_037_source_O_faces'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value

def write(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def free_bytes() -> int:
    s = os.statvfs('/')
    return int(s.f_bavail * s.f_frsize)

def require_headroom(stage: str) -> None:
    available = free_bytes()
    if available < 200 * 1024**2:
        raise RuntimeError(f'200 MiB disk floor blocks {stage}: free={available}')

def run(command, log: Path, *, slot: bool = True, timeout: int = 900) -> dict:
    require_headroom(str(log))
    full = [str(PYTHON), str(SLOT), *map(str, command)] if slot else list(map(str, command))
    started = time.monotonic()
    with Path(log).open('x') as out:
        proc = subprocess.run(full, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
    receipt = {'command': full, 'returncode': int(proc.returncode), 'wall_seconds': time.monotonic() - started,
               'log_sha256': sha(log), 'geometry_slot_used': bool(slot)}
    write(Path(str(log) + '.execution.json'), receipt)
    if proc.returncode:
        raise RuntimeError(f'child exited {proc.returncode}: {log}')
    return receipt

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify_file_pins() -> dict:
    paths = {
      'face_checker_302_sha256': FACE_CHECKER, 'interval_selector_300_sha256': INTERVAL_SELECTOR,
      'area_reference_sha256': AREA_REFERENCE, 'source_base_sha256': SOURCE_BASE,
      'solver_279_sha256': SOLVER, 'face_builder_292_sha256': FACE_BUILDER,
      'global_backtracker_275_sha256': GLOBAL_HELPER, 'local_backtracker_287_sha256': BACKTRACKER,
      'local_step_helper_183_sha256': LOCAL_HELPER, 'prepare_refined_185_sha256': PREP185,
      'point_emitter_235_sha256': POINT_EMITTER, 'pair_owner_sha256': PAIR_OWNER,
      'area_owner_sha256': AREA_OWNER, 'bary_owner_sha256': BARY_OWNER,
      'audit130_sha256': AUDIT130, 'audit159_sha256': AUDIT159,
      'surface_damping_258_sha256': H / 'surface-regression-witnesses-258.py',
      'point_damping_254_sha256': H / 'point-regression-witnesses-254.py',
      'merit_owner_175_sha256': D / 'D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py',
    }
    actual = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[key] = sha(path)
        if actual[key] != PINS[key]:
            raise ValueError(f'pinned source changed: {key} expected {PINS[key]} got {actual[key]}')
    return actual

def surface_pair_count(report: dict, *, expected_banks: int = 185, expected_faces: int | None = None) -> int:
    banks=report.get('banks',[])
    keys={(b['body'],b['stable_bone_id'])for b in banks}
    if report.get('complete')is not True or len(banks)!=expected_banks or len(keys)!=expected_banks:
        raise ValueError('incomplete or duplicate typed bank inventory')
    if expected_faces is not None and report.get('surface_face_count')!=expected_faces:
        raise ValueError('surface report topology domain mismatch')
    owner=load_module(D/'D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py','merit295')
    return owner.summarize_surface_crossing_report(report,expected_bank_keys=keys)['total_crossing_pairs']

def validate_state(state_path: Path):
    if sha(state_path) != PINS['283_state_sha256']:
        raise ValueError('initial state JSON is not the pinned 283 selected result')
    state = read(state_path)
    candidate = Path(state['candidate'])
    point_path = Path(state['point_summary'])
    surface_path = Path(state['surface_report'])
    if sha(candidate) != PINS['283_candidate_sha256']:
        raise ValueError('initial candidate is not the pinned 283 output')
    if sha(point_path) != PINS['283_point_summary_sha256'] or sha(surface_path) != PINS['283_surface_report_sha256']:
        raise ValueError('initial 283 exact audit receipts differ from pins')
    point = read(point_path); surface = read(surface_path)
    if point.get('candidate_sha256') != sha(candidate) or surface.get('candidate_sha256') != sha(candidate):
        raise ValueError('283 candidate/audit hash mismatch')
    if point.get('complete_surface_audit') is not True or point.get('exact_PLC_surface_pairs') != 0:
        raise ValueError('283 seed lacks complete zero-self audit')
    if surface.get('complete') is not True or surface.get('candidate_sha256') != sha(candidate) or surface_pair_count(surface, expected_faces=41733) != 3886:
        raise ValueError('283 seed does not have its matched complete exact 185-bank surface report')
    z = np.load(candidate, allow_pickle=False)
    points = np.asarray(z['points_world_m'], dtype=np.float32)
    faces = np.asarray(z['source_O_faces'], dtype=np.int64)
    if points.shape != (20889, 3) or faces.shape != (41733, 3):
        raise ValueError('283 topology shape changed')
    if 'typed_coarse_anchor_output_rows_018' not in z.files:
        raise ValueError('typed anchor identity rows missing')
    anchors = np.asarray(z['typed_coarse_anchor_output_rows_018'], dtype=np.int64)
    if np.any(anchors < 0) or np.any(anchors >= len(points)) or np.unique(anchors).size != len(anchors):
        raise ValueError('typed anchor rows malformed')
    refz = np.load(AREA_REFERENCE, allow_pickle=False)
    areap = np.asarray(refz[AREA_POINTS_KEY], dtype=np.float64)
    areaf = np.asarray(refz[AREA_FACES_KEY], dtype=np.int64)
    reference_start = np.asarray(refz['points_world_m'], dtype=np.float32)
    if areap.shape != points.shape or not np.array_equal(areaf, faces):
        raise ValueError('pinned 278 area-reference domain/topology differs from 283')
    if not np.array_equal(points[:FIXED_PREFIX].view(np.uint32), reference_start[:FIXED_PREFIX].view(np.uint32)):
        raise ValueError('283 fixed prefix differs from the 278 candidate prefix')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(areap)):
        raise ValueError('nonfinite candidate/reference coordinates')
    return state, candidate, faces, anchors

def seed_archive_index() -> tuple[dict[str, dict], dict]:
    """Validate the independently verified archive that retains retired 283 candidates."""
    for path in (SEED_ARCHIVE_MANIFEST, SEED_ARCHIVE, SEED_ARCHIVE_REMOTE_RECEIPT, SEED_ARCHIVE_LOCAL_RECEIPT):
        if not path.is_file():
            raise FileNotFoundError(f"missing 283 recovery evidence: {path}")
    if sha(SEED_ARCHIVE_MANIFEST) != SEED_ARCHIVE_MANIFEST_SHA256 or sha(SEED_ARCHIVE) != SEED_ARCHIVE_SHA256:
        raise ValueError("283 retained-trial archive or manifest hash changed")
    manifest = read(SEED_ARCHIVE_MANIFEST)
    remote = read(SEED_ARCHIVE_REMOTE_RECEIPT)
    local = read(SEED_ARCHIVE_LOCAL_RECEIPT)
    for receipt in (remote, local):
        if receipt.get("archive_sha256") != SEED_ARCHIVE_SHA256 or receipt.get("verified_members") != manifest.get("member_count"):
            raise ValueError("283 archive recovery receipt does not match retained archive manifest")
        if receipt.get("bytes_hash_mode_owner_mtime_verified") is not True:
            raise ValueError("283 archive recovery receipt lacks full content/metadata verification")
    members = {m.get("source_path"): m for m in manifest.get("members", [])
               if isinstance(m, dict) and isinstance(m.get("source_path"), str)}
    if len(members) != manifest.get("member_count"):
        raise ValueError("283 archive manifest member inventory is incomplete or duplicated")
    return members, {"manifest": str(SEED_ARCHIVE_MANIFEST), "manifest_sha256": sha(SEED_ARCHIVE_MANIFEST),
        "archive": str(SEED_ARCHIVE), "archive_sha256": sha(SEED_ARCHIVE),
        "remote_verification": str(SEED_ARCHIVE_REMOTE_RECEIPT), "remote_verification_sha256": sha(SEED_ARCHIVE_REMOTE_RECEIPT),
        "verified_members": int(remote["verified_members"]), "verified_bytes": int(remote["verified_bytes"])}

def collect_pairs(root: Path, faces: np.ndarray, archive_members: dict[str, dict] | None = None,
                  *, allow_missing_local_candidate: bool = False):
    """Union complete 283-topology exact rows; retired candidates resolve via verified archive."""
    pairs: set[tuple[int, int]] = set()
    receipts = []
    root = root.resolve()
    if not root.exists():
        return pairs, receipts
    if archive_members is None:
        archive_members, archive_receipt = seed_archive_index()
    else:
        archive_receipt = None
    for pair_path in sorted(root.rglob("surface-self-pairs.npy")):
        resolved = pair_path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"self-pair file escapes allowed root: {pair_path}")
        summary_path = pair_path.parent / "summary.json"
        candidate_path = pair_path.parent.parent / "candidate.npz"
        if not summary_path.is_file():
            raise ValueError(f"self-pair artifact lacks adjacent exact audit summary: {pair_path}")
        summary = read(summary_path)
        if summary.get("complete_surface_audit") is not True:
            raise ValueError(f"incomplete exact surface audit: {summary_path}")
        if summary.get("surface_predicate_sha256") != SEED_SURFACE_PREDICATE_SHA256:
            raise ValueError(f"unexpected exact surface predicate: {summary_path}")
        if int(summary.get("points", -1)) != 20889 or int(summary.get("faces", -1)) != len(faces):
            raise ValueError(f"self-pair report does not match 283 topology dimensions: {summary_path}")
        if summary.get("geometry_admitted") is not False or summary.get("accepted_physical_steps") != 0:
            raise ValueError(f"283 historical exact-pair report has unexpected admission status: {summary_path}")
        candidate_sha = summary.get("candidate_sha256")
        if not isinstance(candidate_sha, str) or len(candidate_sha) != 64:
            raise ValueError(f"missing candidate SHA in exact audit summary: {summary_path}")
        if candidate_path.is_file():
            actual_candidate_sha = sha(candidate_path)
            if candidate_sha != actual_candidate_sha:
                raise ValueError(f"self-pair candidate hash mismatch: {summary_path}")
            audited = np.load(candidate_path, allow_pickle=False)
            f = np.asarray(audited["source_O_faces"], dtype=np.int64)
            if f.shape != faces.shape or not np.array_equal(f, faces):
                raise ValueError(f"self-pair topology mismatch: {candidate_path}")
            candidate_receipt = {"mode": "local_candidate_bytes_verified", "path": str(candidate_path), "sha256": actual_candidate_sha}
        else:
            archived = archive_members.get(str(candidate_path))
            if archived is not None and archived.get("sha256") == candidate_sha and int(archived.get("bytes", -1)) > 0:
                candidate_receipt = {
                    "mode": "candidate_bytes_retired_verified_in_remote_archive", "source_path": str(candidate_path),
                    "sha256": candidate_sha, "archive_member": archived.get("archive_path"), "archive_bytes": int(archived["bytes"])}
            elif allow_missing_local_candidate:
                candidate_receipt = {
                    "mode": "candidate_bytes_not_retained_locally", "source_path": str(candidate_path),
                    "sha256_as_reported": candidate_sha, "topology_inherited_from_pinned_candidate": True}
            else:
                raise ValueError(f"retired self-pair candidate is not pinned by verified archive manifest: {candidate_path}")
        a = np.load(pair_path, allow_pickle=False)
        if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in "iu":
            raise ValueError(f"malformed exact self-pair rows: {pair_path}")
        a = np.asarray(a, dtype=np.int64)
        if np.any(a < 0) or np.any(a >= len(faces)) or np.any(a[:, 0] == a[:, 1]):
            raise ValueError(f"invalid exact self-pair face IDs: {pair_path}")
        if summary.get("exact_PLC_surface_pairs") != len(a):
            raise ValueError(f"self-pair row count differs from exact audit: {pair_path}")
        canonical = np.sort(a, axis=1)
        if len(np.unique(canonical, axis=0)) != len(canonical):
            raise ValueError(f"duplicate unordered face-pair rows: {pair_path}")
        pairs.update((int(x), int(y)) for x, y in canonical)
        receipts.append({"pair_file": str(pair_path), "pair_file_sha256": sha(pair_path),
                         "summary": str(summary_path), "summary_sha256": sha(summary_path),
                         "candidate": candidate_receipt, "pair_rows": int(len(a)),
                         "topology_dimensions": {"points": 20889, "faces": int(len(faces))},
                         "predicate_sha256": SEED_SURFACE_PREDICATE_SHA256,
                         "audit_root": str(root), "archive_recovery": archive_receipt})
    return pairs, receipts

def accumulate_pairs(faces: np.ndarray, archive_members: dict[str, dict]):
    initial, ir = collect_pairs(SEED_ROOT, faces, archive_members)
    later, lr = collect_pairs(CURRENT_OUT, faces, archive_members, allow_missing_local_candidate=True)
    return initial | later, ir + lr

def run_point_targets(m185, state: dict, candidate: Path, out: Path):
    point = Path(state['point_summary']); ip = read(point)
    query = point.parent / 'query'
    target_dir = out / 'point-targets'; target_dir.mkdir()
    npz = target_dir / 'nearest-union-exits.npz'
    report = target_dir / 'nearest-union-exits.json'
    if int(ip.get('inside_nodes', -1)) > 0:
        expected = {'candidate': sha(candidate), **{'query_' + k: sha(query / (k + '.csv')) for k in ('points', 'pairs', 'nodes')}}
        targetpy = target_dir / 'build_targets.py'
        m185.emit(m185.TARGET, targetpy,
                  dict(CAND=candidate, QUERY=query, OUT=target_dir, RPT=report, NPZ=npz,
                       EXPECTED_QUERY_HASHES=expected), 'target')
        run([PYTHON, targetpy], target_dir / 'targets.log', slot=True)
    else:
        run([PYTHON, POINT_EMITTER, '--candidate', candidate, '--summary', point, '--query-dir', query,
             '--npz', npz, '--report', report], target_dir / 'targets.log', slot=True)
    t = read(report)
    if t.get('status') != 'PASS_FIRST_UNION_EXIT_TARGETS_ONLY' or t.get('passing_targets') != int(ip['inside_nodes']):
        raise ValueError(f'point target builder did not certify every current inside row: {t.get("status")}')
    if t.get('inputs', {}).get('candidate_sha256') != sha(candidate):
        raise ValueError('point target report is for a different candidate')
    return npz, report

def run_face_targets(candidate: Path, surface: Path, out: Path, first_cache: Path | None = None):
    target_dir = out / 'face-targets'
    if first_cache is not None:
        target_dir.mkdir()
        # A cache is valid only for the exact frozen 283 candidate and surface report.
        cached_report = first_cache / 'report.json'; cached_npz = first_cache / 'targets.npz'
        cached_rows = first_cache / 'target-records.jsonl'
        if sha(candidate) != PINS['283_candidate_sha256'] or sha(surface) != PINS['283_surface_report_sha256']:
            raise ValueError('--first-face-target-dir only applies to the exact initial 283 candidate/report')
        if not cached_report.is_file() or not cached_npz.is_file() or not cached_rows.is_file():
            raise ValueError('requested 293 cache is incomplete; omit it to run the pinned 292 builder')
        if sha(cached_report) != CACHED_293_REPORT_SHA256 or sha(cached_npz) != CACHED_293_NPZ_SHA256:
            raise ValueError('requested 293 cache byte hashes do not match frozen pins')
        t = read(cached_report)
        if (t.get('runner_sha256') != sha(FACE_BUILDER) or t.get('direction_mode') != 'outward-nearest'
            or t.get('inputs', {}).get('candidate_sha256') != sha(candidate)
            or t.get('inputs', {}).get('surface_report_sha256') != sha(surface)
            or t.get('matched_159_complete') is not True
            or t.get('passing_targets') != t.get('crossing_pairs')
            or t.get('passing_targets') != 4002 or t.get('failure_counts') != {}):
            raise ValueError('requested 293 cache is not a complete matched all-target result')
        shutil.copy2(cached_npz, target_dir / 'targets.npz')
        shutil.copy2(cached_report, target_dir / 'report.json')
        shutil.copy2(cached_rows, target_dir / 'target-records.jsonl')
        write(target_dir / 'cache-use.json', {'source_dir': str(first_cache), 'report_sha256': sha(cached_report),
            'targets_sha256': sha(cached_npz), 'runner_sha256': t['runner_sha256'], 'cache_complete': True,
            'classification': 'construction target proposals only; inherited from 293, not clearance admission'})
    else:
        run([PYTHON, FACE_BUILDER, '--candidate', candidate, '--surface-report', surface,
             '--output-dir', target_dir, '--direction-mode', 'outward-nearest'],
            out / 'face-targets.log', slot=True)
    npz = target_dir / 'targets.npz'; report = target_dir / 'report.json'
    t = read(report)
    if t.get('runner_sha256') != sha(FACE_BUILDER) or t.get('direction_mode') != 'outward-nearest':
        raise ValueError('292 face-target runner/direction pin mismatch')
    if t.get('inputs', {}).get('candidate_sha256') != sha(candidate):
        raise ValueError('292 face-target report is for a different candidate')
    if not npz.is_file():
        raise ValueError('303 face targets NPZ missing')
    check = load_module(FACE_CHECKER, 'full_face_target_preflight_302_for301')
    receipt = check.validate_full_face_target_report(t, expected_candidate_sha256=sha(candidate), expected_surface_report_sha256=sha(surface), expected_surface_pair_count=surface_pair_count(read(surface), expected_faces=41733), expected_banks=185)
    write(target_dir / 'completeness-preflight.json', receipt)
    return npz, report

def ensure_prefix(base_path: Path, trial_path: Path):
    base = np.load(base_path, allow_pickle=False); trial = np.load(trial_path, allow_pickle=False)
    x = np.asarray(base['points_world_m']); y = np.asarray(trial['points_world_m'])
    if x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError('trial point domain differs from base')
    if np.ascontiguousarray(x[:FIXED_PREFIX]).tobytes() != np.ascontiguousarray(y[:FIXED_PREFIX]).tobytes():
        raise ValueError('trial changed fixed prefix before backtracking')
    if not np.array_equal(base['source_O_faces'], trial['source_O_faces']):
        raise ValueError('trial changed surface topology')

def run_fit_attempt(state: dict, candidate: Path, self_pairs: np.ndarray, point_npz: Path,
                    point_report: Path, face_npz: Path, face_report: Path, out: Path):
    proposal = out / 'proposal'
    pairs_path = out / 'self-pairs.npy'; np.save(pairs_path, self_pairs.astype(np.int64, copy=False))
    args = [PYTHON, SOLVER,
        '--base-candidate', candidate, '--raw-trial', candidate, '--self-pairs', pairs_path,
        '--source-base-npz', SOURCE_BASE,
        '--area-reference-npz', AREA_REFERENCE,
        '--area-reference-points-key', AREA_POINTS_KEY, '--area-reference-faces-key', AREA_FACES_KEY,
        '--pair-owner', PAIR_OWNER, '--area-owner', AREA_OWNER, '--bary-owner', BARY_OWNER,
        '--point-target-npz', point_npz, '--point-target-report', point_report,
        '--face-target-npz', face_npz, '--face-target-report', face_report,
        '--fixed-prefix-nodes', FIXED_PREFIX, '--slack-tolerance', '0.05',
        '--time-limit-s', '60', '--area-scp-iterations', '4', '--output', proposal]
    run(args, out / 'fit.log', slot=True, timeout=900)
    raw_lp = proposal / 'raw-lp-candidate.npz'
    report_path = proposal / 'report.json'
    if not raw_lp.is_file():
        if report_path.is_file():
            write(out / 'no-raw-lp.json', {'report': str(report_path), 'report_sha256': sha(report_path),
                                           'status': read(report_path).get('status'), 'native_admission': False,
                                           'accepted_physical_steps': 0})
        return None
    raw = np.load(raw_lp, allow_pickle=False)
    z = np.load(candidate, allow_pickle=False)
    keys = [k for k in raw.files if k.endswith('_lp_delta_mm_f64')]
    if len(keys) != 1:
        raise ValueError(f'expected one 279 LP delta, found {keys}')
    prefix = keys[0][:-len('delta_mm_f64')]
    support = np.asarray(raw[prefix + 'support_rows'], dtype=np.int64)
    delta = np.asarray(raw[keys[0]], dtype=np.float64)
    if support.ndim != 1 or len(np.unique(support)) != len(support) or np.any(support < 0) or np.any(support >= len(z['points_world_m'])):
        raise ValueError('279 LP support rows malformed')
    if delta.shape != (len(support), 3) or not np.isfinite(delta).all():
        raise ValueError('279 LP displacement vector malformed')
    X = np.asarray(z['points_world_m'], dtype=np.float32)
    Xtrial = X.copy()
    Xtrial[support] = ((X[support].astype(np.float64) * 1000.0 + delta) * 0.001).astype(np.float32)
    anchors = np.asarray(z['typed_coarse_anchor_output_rows_018'], dtype=np.int64)
    Xtrial[anchors] = X[anchors]
    if np.ascontiguousarray(Xtrial[:FIXED_PREFIX]).tobytes() != np.ascontiguousarray(X[:FIXED_PREFIX]).tobytes():
        raise ValueError('half-step emission changed fixed prefix')
    if not np.isfinite(Xtrial).all():
        raise ValueError('half-step emission contains nonfinite coordinates')
    require_headroom('half-step candidate write')
    trial = out / 'trial-halfstep.npz'
    payload = {k: z[k] for k in z.files}; payload['points_world_m'] = Xtrial
    np.savez_compressed(trial, **payload)
    ensure_prefix(candidate, trial)
    write(out / 'trial-provenance.json', {
        'base_candidate': str(candidate), 'base_candidate_sha256': sha(candidate),
        'raw_lp_candidate': str(raw_lp), 'raw_lp_candidate_sha256': sha(raw_lp),
        'fitter': str(SOLVER), 'fitter_sha256': sha(SOLVER), 'halfstep_alpha': 1.0,
        'fixed_prefix_nodes': FIXED_PREFIX, 'fixed_prefix_bitwise_unchanged': True,
        'trial': str(trial), 'trial_sha256': sha(trial),
        'native_admission': False, 'accepted_physical_steps': 0,
    })
    return trial

def run_backtracker(state: dict, trial: Path, out: Path, helper: Path, entry: str,
                    max_audits: int, max_passes: int):
    out.mkdir(exist_ok=False)
    cfg = {
      'state': state, 'trial': str(trial), 'out': str(out / 'backtrack'),
      'parent_fit': state['parent_fit'], 'helper': str(helper), 'helper_sha256': sha(helper),
      'helper_entry': entry, 'initial_scale': 0.5, 'zero_below': 1.0 / 64.0,
      'maximum_exact_audits': max_audits, 'maximum_construction_passes': max_passes,
      'fixed_prefix_nodes': FIXED_PREFIX, 'area_reference_npz': str(AREA_REFERENCE),
      'area_reference_sha256': sha(AREA_REFERENCE),
      'area_reference_points_key': AREA_POINTS_KEY, 'area_reference_faces_key': AREA_FACES_KEY,
    }
    cfgpath = out / 'backtrack-config.json'; write(cfgpath, cfg)
    run([PYTHON, BACKTRACKER, cfgpath], out / 'backtracker.log', slot=False, timeout=5400)
    selected = out / 'backtrack/selected-result.json'
    if selected.is_file():
        result = read(selected)
        ensure_prefix(Path(state['candidate']), Path(result['candidate']))
        return selected
    return None

def self_test():
    assert FIXED_PREFIX == 18430
    assert AREA_POINTS_KEY == 'area_reference_037_points_world_m_f64'
    assert AREA_FACES_KEY == 'area_reference_037_source_O_faces'
    assert sha(FACE_BUILDER) == PINS['face_builder_292_sha256']
    empty = np.empty((0, 2), dtype=np.int64)
    assert empty.shape == (0, 2) and empty.dtype.kind in 'iu'
    assert 1 <= 2 <= 2 and 1 <= 3 <= 3
    print(json.dumps({'status': 'PASS', 'checks': ['fixed prefix and area-reference keys', '292 builder pin', 'empty pair seed shape', 'attempt and round limits']}))

def main():
    ap = argparse.ArgumentParser(description='Bounded 290 continuation from 283; source-only preflight by default.')
    ap.add_argument('--state', type=Path, default=DEFAULT_STATE)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--attempts-per-round', type=int, default=2)
    ap.add_argument('--first-face-target-dir', type=Path,
                    help='optional complete pinned 293 cache for the exact initial 283 state only')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if not 1 <= args.rounds <= 3 or not 1 <= args.attempts_per_round <= 2:
        ap.error('rounds must be in [1,3] and attempts-per-round in [1,2]')
    pins = verify_file_pins()
    state, candidate, faces, anchors = validate_state(args.state)
    archive_members, archive_receipt = seed_archive_index()
    seed_pairs, seed_receipts = collect_pairs(SEED_ROOT, faces, archive_members)
    if not seed_receipts:
        raise ValueError('283 seed contains no complete exact self-audit pair receipts')
    if args.preflight_only:
        print(json.dumps({'status': 'PREFLIGHT_PASS_NO_GEOMETRY_LAUNCHED',
            'state': str(args.state), 'state_sha256': sha(args.state),
            'candidate': str(candidate), 'candidate_sha256': sha(candidate),
            'point_inside_nodes': read(Path(state['point_summary']))['inside_nodes'],
            'surface_pairs': surface_pair_count(read(Path(state['surface_report'])), expected_faces=len(faces)),
            'self_pair_seed_count': len(seed_pairs), 'self_pair_receipts': len(seed_receipts),
            'self_pair_seed_root_only': str(SEED_ROOT), 'seed_archive_recovery': archive_receipt, 'fixed_prefix_nodes': FIXED_PREFIX,
            'anchor_count': len(anchors), 'area_reference': str(AREA_REFERENCE),
            'area_reference_sha256': sha(AREA_REFERENCE), 'file_pins': pins}, indent=2))
        return
    if args.out is None:
        ap.error('--out is required unless --preflight-only or --self-test')
    require_headroom('290 output initialization')
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=False)
    global CURRENT_OUT
    CURRENT_OUT = OUT
    m185 = load_module(PREP185, 'prepare_refined_proposal_185_for_290')
    configuration = {
      'status': 'PREPARED_BOUNDED_CONTINUATION', 'initial_state': str(args.state),
      'initial_state_sha256': sha(args.state), 'initial_candidate_sha256': sha(candidate),
      'seed_self_pair_root_only': str(SEED_ROOT), 'seed_self_pair_sources': seed_receipts,
      'seed_self_pair_unique_count': len(seed_pairs), 'source_files': pins,
      'solver_279': str(SOLVER), 'face_builder_292': str(FACE_BUILDER),
      'optional_first_face_target_cache': str(args.first_face_target_dir) if args.first_face_target_dir else None,
      'face_direction_mode': 'outward-nearest', 'global_backtracker_275': str(GLOBAL_HELPER),
      'local_backtracker_183': str(LOCAL_HELPER), 'backtracker_runner_287': str(BACKTRACKER),
      'area_reference_278': str(AREA_REFERENCE), 'area_reference_keys': [AREA_POINTS_KEY, AREA_FACES_KEY],
      'fixed_prefix_nodes': FIXED_PREFIX, 'rounds': args.rounds, 'attempts_per_round': args.attempts_per_round,
      'global_first': {'initial_scale': 0.5, 'zero_below': 1.0/64.0, 'maximum_exact_audits': 8, 'maximum_construction_passes': 16},
      'local_fallback_same_trial': {'initial_scale': 0.5, 'zero_below': 1.0/64.0, 'maximum_exact_audits': 8, 'maximum_construction_passes': 32},
      'disk_floor_bytes': 200*1024**2, 'native_admission': False, 'accepted_physical_steps': 0,
    }
    write(OUT / 'configuration.json', configuration)
    write(OUT / 'initial-self-pair-seed.json', {'only_root': str(SEED_ROOT), 'unique_count': len(seed_pairs), 'sources': seed_receipts, 'archive_recovery': archive_receipt,
          'semantics': 'union of complete 283-topology exact audits only; no older-topology/self-pair files loaded'})
    current_state = state
    current_state_path = args.state
    for round_index in range(args.rounds):
        require_headroom(f'round {round_index+1} initialization')
        rd = OUT / f'round-{round_index+1:03d}'; rd.mkdir()
        candidate = Path(current_state['candidate'])
        if sha(candidate) != current_state['candidate_sha256']:
            raise ValueError('round state candidate hash changed')
        point_path = Path(current_state['point_summary']); surface_path = Path(current_state['surface_report'])
        point_info = read(point_path); surface_info = read(surface_path)
        if point_info.get('candidate_sha256') != sha(candidate) or surface_info.get('candidate_sha256') != sha(candidate):
            raise ValueError('round state point/surface receipts do not match candidate')
        if point_info.get('complete_surface_audit') is not True:
            raise ValueError('round state lacks a complete 130 audit')
        if int(point_info.get('inside_nodes', -1)) == 0 and surface_pair_count(surface_info, expected_faces=len(faces)) == 0:
            write(rd / 'result.json', {'status': 'ZERO_POINT_AND_SURFACE_WITNESSES_UNADMITTED',
                                        'native_admission': False, 'accepted_physical_steps': 0})
            break
        write(rd / 'input-state.json', current_state)
        face_npz, face_report = run_face_targets(candidate, surface_path, rd, args.first_face_target_dir if round_index == 0 else None)
        point_npz, point_report = run_point_targets(m185, current_state, candidate, rd)
        round_selected = None
        for attempt in range(args.attempts_per_round):
            attempt_dir = rd / f'attempt-{attempt+1:03d}'; attempt_dir.mkdir()
            pairs, receipts = accumulate_pairs(faces, archive_members)
            self_pairs = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
            pair_input = attempt_dir / 'self-pairs-input.npy'; np.save(pair_input, self_pairs)
            write(attempt_dir / 'self-pairs-input.json', {'array': str(pair_input), 'sha256': sha(pair_input),
                  'unique_pair_count': len(self_pairs), 'sources': receipts,
                  'seed_root_only': str(SEED_ROOT), 'no_older_topology_pairs': True})
            trial = run_fit_attempt(current_state, candidate, self_pairs, point_npz, point_report, face_npz, face_report, attempt_dir)
            if trial is None:
                write(attempt_dir / 'result.json', {'status': 'NO_RAW_LP_PROPOSAL', 'native_admission': False, 'accepted_physical_steps': 0})
                break
            global_selected = run_backtracker(current_state, trial, attempt_dir / 'global-275', GLOBAL_HELPER,
                    'reduce_global_step_scales', 8, 16)
            selected = global_selected
            selected_mode = 'global-275'
            if selected is None:
                selected = run_backtracker(current_state, trial, attempt_dir / 'local-183-fallback', LOCAL_HELPER,
                        'reduce_surface_witness_node_scales_183', 8, 32)
                selected_mode = 'local-183-fallback'
            fresh_pairs, fresh_receipts = accumulate_pairs(faces, archive_members)
            write(attempt_dir / 'learned-self-pairs.json', {'unique_pairs_after_attempt': len(fresh_pairs),
                  'seed_root_only': str(SEED_ROOT), 'sources': fresh_receipts,
                  'no_older_topology_pairs': True})
            if selected is None:
                write(attempt_dir / 'result.json', {'status': 'NO_GLOBAL_OR_LOCAL_BACKTRACK_SELECTION',
                    'same_trial_used_for_both_backtrackers': True, 'trial': str(trial), 'trial_sha256': sha(trial),
                    'learned_self_pairs': len(fresh_pairs), 'new_self_pairs': len(fresh_pairs - pairs),
                    'native_admission': False, 'accepted_physical_steps': 0})
                if fresh_pairs == pairs:
                    write(attempt_dir / 'stop-repeated-input.json', {
                        'status': 'STOP_IDENTICAL_NEXT_LP_INPUT', 'reason': 'no selected progress and no newly observed exact self-pairs',
                        'pair_input_sha256': sha(pair_input), 'unique_pair_count': len(pairs),
                        'native_admission': False, 'accepted_physical_steps': 0})
                    break
                continue
            result = read(selected)
            result['selected_backtracker'] = selected_mode
            result['selected_backtracker_result'] = str(selected)
            result['selected_backtracker_result_sha256'] = sha(selected)
            round_selected = attempt_dir / 'selected-result.json'
            write(round_selected, result)
            write(attempt_dir / 'result.json', result)
            break
        if round_selected is None:
            write(rd / 'result.json', {'status': 'NO_SELECTED_CONSTRUCTION_PROGRESS',
                'attempts': args.attempts_per_round, 'native_admission': False, 'accepted_physical_steps': 0})
            break
        current_state = read(round_selected)
        current_state_path = round_selected
        write(rd / 'result.json', {'status': 'CONSTRUCTION_PROGRESS_ONLY', 'selected_state': str(round_selected),
               'selected_state_sha256': sha(round_selected), 'candidate': current_state['candidate'],
               'candidate_sha256': current_state['candidate_sha256'], 'selected_backtracker': current_state['selected_backtracker'],
               'native_admission': False, 'accepted_physical_steps': 0})
        write(OUT / 'selected-result.json', current_state)
        print(json.dumps({'round': round_index+1, 'candidate_sha256': current_state['candidate_sha256'],
              'selected_backtracker': current_state['selected_backtracker'],
              'point_merit': current_state.get('point_merit'), 'joint_merit': current_state.get('joint_merit')}, sort_keys=True), flush=True)
    write(OUT / 'completed.json', {'status': 'BOUNDED_290_CONTINUATION_FINISHED', 'final_state': str(current_state_path),
          'final_candidate': current_state.get('candidate'), 'final_candidate_sha256': current_state.get('candidate_sha256'),
          'native_admission': False, 'accepted_physical_steps': 0})

if __name__ == '__main__':
    main()

