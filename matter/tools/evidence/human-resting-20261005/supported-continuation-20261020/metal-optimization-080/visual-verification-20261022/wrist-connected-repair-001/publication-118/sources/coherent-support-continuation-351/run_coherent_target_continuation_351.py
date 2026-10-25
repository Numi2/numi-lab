#!/usr/bin/env python3
"""Bounded coherent continuation with SHA-gated construction-state resume.

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
SEED_ROOT = H / 'joint-boundary-continuation-322'
ADDITIONAL_SEED_ROOT = H / 'connected-fixed-boundary-318'
FOURTH_SEED_ROOT = H / 'coherent-support-continuation-337'
DEFAULT_STATE = SEED_ROOT / 'selected-result.json'
EXTRA_SEED_ROOT = H / 'coherent-source-outward-309'
PREPARED_TARGET_ROOT = H / 'connected-fixed-boundary-318/round-001'
PREPARED_BASE_SHA = 'c1a62013fcce1d3454da824f22bee920455818b59933b24c99167328d04e9340'
AREA_REFERENCE = H / 'connected-refined-278/candidate.npz'
SOURCE_BASE = D / 'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
SOLVER = D / 'D/connected-local-witness-proposal-347/propose_joint_clearance_witnesses_347.py'
FIXED_OWNER = D / 'connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py'
FIXED_EMITTER = D / 'D/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py'
FACE_BUILDER = H / 'source-outward-proposal-300/build_outward_nearest_union_targets_335.py'
FACE_CHECKER = H / 'full-face-target-preflight-302/full_face_target_preflight_302.py'
INTERVAL_SELECTOR = H / 'source-outward-proposal-300/first_representable_union_target_300.py'
SEED_ARCHIVE_MANIFEST = D / 'publication-112/evidence/archive-manifest.json'
SEED_ARCHIVE = D / 'publication-112/evidence/coherent-connected-trials-309.tar.xz'
SEED_ARCHIVE_REMOTE_RECEIPT = D / 'publication-112/remote-recovery-verification.json'
SEED_ARCHIVE_LOCAL_RECEIPT = D / 'publication-112/verified-local-retirement.json'
SEED_ARCHIVE_MANIFEST_SHA256 = '6ff66d1e19b295e174ddd3374fd25f4cabd62dd7c4434d991e680375dd7cdfee'
SEED_ARCHIVE_SHA256 = '772f33abe7d9b358219555840e6590ffd729afd00a28a4229662620e0cc8f930'
SEED_SURFACE_PREDICATE_SHA256 = 'f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
CACHED_293_REPORT_SHA256 = 'c40201ea6adee235475d76ac9fd3d65dce7e809c31b6101b2e7906f27fbf4fb4'
CACHED_293_NPZ_SHA256 = '2c02c3acfdc61c5fa8754a1c4a2d6033f40cb3944af4ada15f4617c11001f306'
CACHED_293_DIR = H / 'outward-nearest-targets-293'
GLOBAL_HELPER = H / 'global_step_backtracking_275.py'
LOCAL_HELPER = D / 'D/local-witness-backtracking-183/canonical/matter/tools/human_skin_fat_surface_fit_constraints.py'
BACKTRACKER = H / 'no-progress-backtracker-306/run_local_witness_backtracking_306.py'
LOCAL_BACKTRACKER = D / 'D/local-witness-backtracking-340/run_local_witness_backtracking_340.py'
LOCAL_HELPER_339 = D / 'D/local-witness-backtracking-339/local_witness_backtracking_339.py'
PREP185 = H / 'prepare_refined_proposal_185.py'
POINT_BUILDER = D / 'D/connected-coherent-point-targets-327/build_coherent_point_targets_327.py'
CONSTRUCTION_CLEARANCE_M = 0.00025
INITIAL_TARGET_CACHE_DIR = H / 'coherent-support-unlock-334/round-001'
INITIAL_TARGET_CACHE_PINS = {
    'candidate_sha256': '2346d174df8b363bca2200c24589a36cf1ad6a4642a4b73a9a156a48e6826b25',
    'point_summary_sha256': 'eb730b14a144130d066ed1a467cd39816c9842361449b78f0816a06f20f9f4e6',
    'surface_report_sha256': '1d72e2bdf7da60d4fba7d400d343d9a9bb9a0b20b36580ebb29b96bf9df02eb9',
    'point_report_sha256': '590f62af5eb3a9ef4722d59b0a7e76465d11cf667d5ca119f9691ed8a4673ee4',
    'point_npz_sha256': '2f42c3d89ebbf9d4c84b997eb91b9c44c80447cd11831b8c5135c2b1c5a56304',
    'face_report_sha256': '3e5eb7efdac0b1f27408049a257866ae7b35267fccc085968d012a33eed52a96',
    'face_npz_sha256': 'd0a58d2cbbf16b5dda8f43f715a13d16484c36be8219c9a39956b168cdd3b4e7',
    'face_targets_jsonl_sha256': '4942f8200eeb6aefb21c338aef324673762db9002e452591a3bb7164ac62ed20',
}
POINT_EMITTER = D / 'connected-local-witness-proposal-235/emit_empty_point_targets_235.py'
PAIR_OWNER = H / 'connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py'
AREA_OWNER = H / 'connected-local-continuation-189-run001/round-005/proposal/round-001/fit/area_scp_support_023.py'
BARY_OWNER = D / 'publication-091/canonical/matter/tools/human_skin_fat_barycentric_constraints.py'
AUDIT130 = D / 'audit_connected_outer_geometry_130.py'
AUDIT159 = D / 'audit_skin_bone_surfaces_159.py'

PREPARED_POINT_REPORT_SHA = 'fcafd12e133aa4fc0426c5c1f0694858c04f7f8e7224140b0231269617a0d962'
PREPARED_POINT_NPZ_SHA = '05303456945acd704529358f8d0befc4e2edfd5e2ca39a6afbdce748681d3a7c'
PREPARED_FACE_REPORT_SHA = 'e8e411170804e863929340d14facf774f6006937c1ccf0b9f75b41e225260496'
PREPARED_FACE_NPZ_SHA = 'defcca6ac91ea6d266c7e432300480d4b8deb41a4632d284ef7d11270106352a'
PINS = {
    'coherent_point_builder_327_sha256':'e86af08eee35cd2335072892cb11580f49097b2a30979ded5e23df0b280f2e22',
    'fixed_owner_313_sha256': 'dc9d4f953ecede00567c59cbbb6b6b63e0a3556b97972689de499cbce29d4c2e',
    'fixed_emitter_314_sha256': '3422f30a61db0724d234c487d70fc3d74237a802261c2f9a72af9e15843a1c78',
    'face_checker_302_sha256': 'caa94e5df1d7363ac3fbdb1d66ed943a806a08bb132a9b49ebeb2236d59d7d50',
    'interval_selector_300_sha256': '5e613e21b7bdebcc036dc9712ad21903df7d081de32bfdb83fae0a0c344ac094',
    '283_state_sha256': '326f477097dc738e1eb97541cd8d46d28fe8a2e8e5af742d5f69fab009d9dd6d',
    '283_candidate_sha256': '2346d174df8b363bca2200c24589a36cf1ad6a4642a4b73a9a156a48e6826b25',
    '283_point_summary_sha256': 'eb730b14a144130d066ed1a467cd39816c9842361449b78f0816a06f20f9f4e6',
    '283_surface_report_sha256': '1d72e2bdf7da60d4fba7d400d343d9a9bb9a0b20b36580ebb29b96bf9df02eb9',
    'area_reference_sha256': 'c2a076267c2a8261a5f1a51b742a0b4e44471d31b200f5b809b8c98d6205348f',
    'source_base_sha256': '70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31',
    'solver_347_sha256': 'ac1fe7a720cf5aaf66345d2db802303d562a3c3cb91c45840ed9883393708dde',
    'face_builder_292_sha256': '1c3468b6e6285afed77e8ae7ca463abb797f6a22d562ccf878734c39161947c9',
    'global_backtracker_275_sha256': '18891b1f700592563e139edfc8bd1361d230ff6fd5af2ae5ee036c9625c1826a',
    'local_backtracker_287_sha256': '4b6c66f3ebbe860a11b1c60243731cc27835c10de3a2e1e91b944ff6e924e84a',
    'local_step_helper_183_sha256': '05b1beb2105a184deb885974db892fa0f8e519fc3e1fdfde5aa441c180c97b34',
    'local_step_helper_339_sha256': '1a561f00d9c769346aa1fd9e4c8f27b59e61fde3d050c8f5335ec949cd94e946',
    'local_backtracker_340_sha256': 'cfe5f5b951e8b2c6593db4d8a4f13a919eee80b13caa6b77190a6fbfd290c6bf',
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
FIXED_PREFIX = 18430  # strict validation of the initial 322 seed only
FIT_FIXED_PREFIX = 0  # later construction moves only rows inside the explicit 315 LP support
AREA_POINTS_KEY = 'area_reference_037_points_world_m_f64'
AREA_FACES_KEY = 'area_reference_037_source_O_faces'

EXTRA_SEED_AUDIT_ROOTS: list[Path] = []
EXPECTED_POINTS = 20889
EXPECTED_FACES = 41733
EXPECTED_BANKS = 185
RESUME_SOURCE_CAP_MM = 12.0
LP_METHODS = ('highs','highs-ipm','highs-ds')

def configured_seed_roots() -> tuple[Path, ...]:
    return (SEED_ROOT, EXTRA_SEED_ROOT, ADDITIONAL_SEED_ROOT, FOURTH_SEED_ROOT, *EXTRA_SEED_AUDIT_ROOTS)

def configure_seed_audit_roots(paths) -> tuple[Path, ...]:
    global EXTRA_SEED_AUDIT_ROOTS
    fixed = {p.resolve() for p in (SEED_ROOT, EXTRA_SEED_ROOT, ADDITIONAL_SEED_ROOT, FOURTH_SEED_ROOT)}
    extras=[]
    for raw in paths:
        path=Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f'additional self-audit root is not an existing directory: {path}')
        if not path.is_relative_to(H.resolve()):
            raise ValueError(f'additional self-audit root escapes the registered H evidence tree: {path}')
        if path in fixed or path in extras:
            raise ValueError(f'duplicate self-audit root: {path}')
        if any(path.is_relative_to(old) or old.is_relative_to(path) for old in (*fixed,*extras)):
            raise ValueError(f'overlapping self-audit roots would double-count receipts: {path}')
        extras.append(path)
    EXTRA_SEED_AUDIT_ROOTS=extras
    return configured_seed_roots()

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
      'coherent_point_builder_327_sha256':POINT_BUILDER,
      'fixed_owner_313_sha256': FIXED_OWNER, 'fixed_emitter_314_sha256': FIXED_EMITTER,
      'face_checker_302_sha256': FACE_CHECKER, 'interval_selector_300_sha256': INTERVAL_SELECTOR,
      'area_reference_sha256': AREA_REFERENCE, 'source_base_sha256': SOURCE_BASE,
      'solver_347_sha256': SOLVER, 'face_builder_292_sha256': FACE_BUILDER,
      'global_backtracker_275_sha256': GLOBAL_HELPER, 'local_backtracker_287_sha256': BACKTRACKER,
      'local_step_helper_183_sha256': LOCAL_HELPER, 'local_step_helper_339_sha256': LOCAL_HELPER_339,
      'local_backtracker_340_sha256': LOCAL_BACKTRACKER, 'prepare_refined_185_sha256': PREP185,
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

def validate_initial_state(state_path: Path):
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
    if surface.get('complete') is not True or surface.get('candidate_sha256') != sha(candidate) or surface_pair_count(surface, expected_faces=41733) != 3600:
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

def candidate_core_array_mismatches(base, resumed) -> list[str]:
    """Exact byte comparison for all persisted baseline arrays except mutable coordinates."""
    mismatches=[]
    base_files=set(base.files if hasattr(base,'files') else base.keys())
    resumed_files=set(resumed.files if hasattr(resumed,'files') else resumed.keys())
    for key in sorted(base_files):
        if key=='points_world_m': continue
        if key not in resumed_files:
            mismatches.append(key); continue
        a=np.asarray(base[key]); b=np.asarray(resumed[key])
        if (a.shape!=b.shape or a.dtype!=b.dtype or
            not np.array_equal(np.ascontiguousarray(a).view(np.uint8),
                               np.ascontiguousarray(b).view(np.uint8))):
            mismatches.append(key)
    return mismatches

def validate_resume_state(state_path: Path, *, expected_state_sha256: str,
                          expected_point_summary_sha256: str | None = None,
                          expected_surface_report_sha256: str | None = None):
    actual_state_sha=sha(state_path)
    if not isinstance(expected_state_sha256,str) or actual_state_sha!=expected_state_sha256.lower():
        raise ValueError('resume state does not match explicit --resume-state-sha256')
    state=read(state_path)
    if state.get('status')!='CONSTRUCTION_PROGRESS_ONLY':
        raise ValueError('resume state is not selected construction-only progress')
    if state.get('native_admission') is not False or state.get('accepted_physical_steps')!=0:
        raise ValueError('resume state carries native admission or physical steps')
    if state.get('geometry_admitted') is True:
        raise ValueError('resume state claims geometry admission')
    candidate=Path(state.get('candidate',''))
    point_path=Path(state.get('point_summary',''))
    surface_path=Path(state.get('surface_report',''))
    for label,path in (('candidate',candidate),('point summary',point_path),('surface report',surface_path)):
        if not path.is_file():
            raise FileNotFoundError(f'resume {label} is missing: {path}')
    candidate_sha=sha(candidate)
    if state.get('candidate_sha256')!=candidate_sha:
        raise ValueError('resume candidate bytes do not match selected state candidate SHA')
    def check_receipt_hash(label, declared, supplied, path):
        actual=sha(path)
        if declared not in (None,'') and declared!=actual:
            raise ValueError(f'resume state {label} hash does not match file bytes')
        if supplied not in (None,'') and supplied.lower()!=actual:
            raise ValueError(f'explicit resume {label} SHA does not match file bytes')
        if declared in (None,'') and supplied in (None,''):
            raise ValueError(f'resume lacks a pinned {label} SHA; pass its explicit CLI SHA')
        return actual
    point_sha=check_receipt_hash('point summary',state.get('point_summary_sha256'),
                                 expected_point_summary_sha256,point_path)
    surface_sha=check_receipt_hash('surface report',state.get('surface_report_sha256'),
                                   expected_surface_report_sha256,surface_path)
    point=read(point_path); surface=read(surface_path)
    for label,report in (('point',point),('surface',surface)):
        if report.get('candidate_sha256')!=candidate_sha:
            raise ValueError(f'resume {label} report candidate SHA mismatch')
    if point.get('complete_surface_audit') is not True or point.get('exact_PLC_surface_pairs')!=0:
        raise ValueError('resume point/self audit is incomplete or has nonzero exact self pairs')
    if int(point.get('points',-1))!=EXPECTED_POINTS or int(point.get('faces',-1))!=EXPECTED_FACES:
        raise ValueError('resume point audit topology dimensions changed')
    if point.get('surface_predicate_sha256')!=SEED_SURFACE_PREDICATE_SHA256:
        raise ValueError('resume point audit predicate is not the registered exact predicate')
    if surface.get('complete') is not True or surface.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT':
        raise ValueError('resume surface audit is incomplete')
    if surface.get('script_sha256')!=PINS['audit159_sha256']:
        raise ValueError('resume surface audit script hash is not the pinned exact owner')
    if surface.get('exact_predicate_sha256')!=SEED_SURFACE_PREDICATE_SHA256:
        raise ValueError('resume surface audit predicate is not the registered exact predicate')
    pair_count=surface_pair_count(surface,expected_banks=EXPECTED_BANKS,expected_faces=EXPECTED_FACES)
    joint=state.get('joint_merit')
    if not isinstance(joint,dict) or joint.get('classification')!='JOINT_CLEARANCE_CONSTRUCTION_PROGRESS':
        raise ValueError('resume state lacks a joint construction merit receipt')
    checks={
        'after_candidate_sha256':candidate_sha,
        'after_surface_pairs':pair_count,
        'after_point_pairs':int(point.get('inside_node_bank_pairs',-1)),
        'topology_points':EXPECTED_POINTS,'topology_faces':EXPECTED_FACES,'typed_bank_count':EXPECTED_BANKS,
    }
    for key,want in checks.items():
        if joint.get(key)!=want:
            raise ValueError(f'resume joint-merit {key} does not match complete audit inputs')
    if joint.get('construction_progress') is not True or joint.get('geometry_admitted') is not False or joint.get('accepted_physical_steps')!=0:
        raise ValueError('resume joint-merit status is not construction-only progress')
    points_npz=np.load(candidate,allow_pickle=False)
    points=np.asarray(points_npz['points_world_m'],dtype=np.float32)
    faces=np.asarray(points_npz['source_O_faces'],dtype=np.int64)
    if points.shape!=(EXPECTED_POINTS,3) or faces.shape!=(EXPECTED_FACES,3):
        raise ValueError('resume candidate topology shape changed')
    if not np.isfinite(points).all() or np.any(faces<0) or np.any(faces>=len(points)):
        raise ValueError('resume candidate has nonfinite coordinates or invalid face indices')
    # Every persisted baseline array except coordinates must remain byte-identical.
    base_state=read(DEFAULT_STATE); base_path=Path(base_state['candidate'])
    if sha(base_path)!=PINS['283_candidate_sha256']:
        raise ValueError('pinned 283 baseline candidate changed')
    base=np.load(base_path,allow_pickle=False)
    if not np.array_equal(faces,np.asarray(base['source_O_faces'],dtype=np.int64)):
        raise ValueError('resume candidate face topology differs from pinned 322 ancestry')
    mismatches=[]
    for key in base.files:
        if key=='points_world_m': continue
        if key not in points_npz.files:
            mismatches.append(key); continue
        a=np.asarray(base[key]); b=np.asarray(points_npz[key])
        if (a.shape!=b.shape or a.dtype!=b.dtype or
            not np.array_equal(np.ascontiguousarray(a).view(np.uint8),
                               np.ascontiguousarray(b).view(np.uint8))):
            mismatches.append(key)
    if mismatches:
        raise ValueError('resume candidate changed immutable topology/core ancestry arrays: '+','.join(mismatches[:12]))
    anchors=np.asarray(points_npz['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
    base_anchors=np.asarray(base['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
    if not np.array_equal(anchors,base_anchors) or len(anchors)!=len(np.unique(anchors)):
        raise ValueError('resume typed coarse anchor identity changed')
    if not np.array_equal(points[anchors].view(np.uint32),np.asarray(base['points_world_m'],dtype=np.float32)[anchors].view(np.uint32)):
        raise ValueError('resume candidate moved a typed coarse anchor')
    refz=np.load(AREA_REFERENCE,allow_pickle=False)
    area_points=np.asarray(refz[AREA_POINTS_KEY],dtype=np.float64)
    area_faces=np.asarray(refz[AREA_FACES_KEY],dtype=np.int64)
    if area_points.shape!=points.shape or not np.array_equal(area_faces,faces):
        raise ValueError('resume fixed-037 area reference topology differs')
    if ('area_reference_037_points_world_m_f64' not in points_npz.files or
        not np.array_equal(np.asarray(points_npz['area_reference_037_points_world_m_f64']),area_points) or
        not np.array_equal(np.asarray(points_npz['area_reference_037_source_O_faces']),area_faces)):
        raise ValueError('resume candidate does not preserve the fixed-037 area reference bytes')
    solver=load_module(SOLVER,'solver347_resume_validation')
    # The selected-state receipt pins the orientation gate from its immediate parent step.
    # Re-evaluate F32 degeneracy/orientation and fixed-037 magnitude on the exact selected bytes.
    area=solver.full_f32_area_gate(points,points.astype(np.float64)*1000.0,
                                   area_points*1000.0,faces,float(solver.AREA_FLOOR))
    if area.get('pass') is not True:
        raise ValueError('resume candidate fails full F32 current-orientation/fixed-037 area gates')
    if state.get('anchors_bitwise_unchanged') is not True or state.get('source_cap_pass') is not True:
        raise ValueError('resume selected-state anchor/source-cap receipt is not passing')
    source_ref,source_receipt=solver.compose_source_reference(points_npz,SOURCE_BASE)
    source_norms=np.linalg.norm(points.astype(np.float64)-source_ref,axis=1)*1000.0
    actual_source_max=float(source_norms.max(initial=0.0))
    claimed_source_max=float(state.get('maximum_source_displacement_mm',float('nan')))
    if not np.isfinite(claimed_source_max) or abs(claimed_source_max-actual_source_max)>5e-5:
        raise ValueError('resume selected-state source displacement disagrees with composed source provenance')
    if actual_source_max>RESUME_SOURCE_CAP_MM+1e-6:
        raise ValueError('resume candidate exceeds registered 12 mm source construction cap')
    claimed_oriented=float(state.get('minimum_allface_oriented_area_ratio',float('nan')))
    claimed_fixed=float(state.get('minimum_fixed_reference_area_magnitude_ratio',float('nan')))
    if not np.isfinite(claimed_oriented) or claimed_oriented<float(solver.AREA_FLOOR)-1e-12:
        raise ValueError('resume selected-state current-oriented area receipt is below floor')
    if not np.isfinite(claimed_fixed) or claimed_fixed<float(solver.AREA_FLOOR)-1e-12:
        raise ValueError('resume selected-state fixed-037 area receipt is below floor')
    if abs(float(area['minimum_fixed_037_reference_magnitude_ratio'])-claimed_fixed)>5e-10:
        raise ValueError('resume fixed-037 area receipt disagrees with selected candidate bytes')
    state['_resume_verified_hashes']={'state_sha256':actual_state_sha,'candidate_sha256':candidate_sha,
        'point_summary_sha256':point_sha,'surface_report_sha256':surface_sha,
        'area_gate':area,'maximum_source_displacement_mm_recomputed':actual_source_max,
        'source_reference_receipts':source_receipt,'resume_mode':'explicit-sha-pinned-construction-progress'}
    return state,candidate,faces,anchors

def validate_state(state_path: Path, *, resume_state_sha256=None,
                   resume_point_summary_sha256=None, resume_surface_report_sha256=None):
    if resume_state_sha256 is None:
        if Path(state_path).resolve()!=DEFAULT_STATE.resolve():
            raise ValueError('non-default selected state requires --resume-state-sha256')
        return validate_initial_state(state_path)
    return validate_resume_state(state_path,expected_state_sha256=resume_state_sha256,
        expected_point_summary_sha256=resume_point_summary_sha256,
        expected_surface_report_sha256=resume_surface_report_sha256)

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
    pairs: set[tuple[int, int]] = set()
    receipts = []
    for root in configured_seed_roots():
        current, current_receipts = collect_pairs(root, faces, archive_members)
        pairs.update(current)
        receipts.extend(current_receipts)
    later, later_receipts = collect_pairs(CURRENT_OUT, faces, archive_members, allow_missing_local_candidate=True)
    pairs.update(later)
    receipts.extend(later_receipts)
    return pairs, receipts

def collect_seed_pairs(faces: np.ndarray, archive_members: dict[str, dict]):
    roots = configured_seed_roots()
    pairs: set[tuple[int, int]] = set()
    receipts = []
    for root in roots:
        current, current_receipts = collect_pairs(root, faces, archive_members)
        pairs.update(current)
        receipts.extend(current_receipts)
    return pairs, receipts

def validate_point_source_ancestry(targets, candidate_points, candidate_faces):
    """Validate 327's per-target face ancestry before using its source-normal targets."""
    if not isinstance(targets, list) or not targets:
        raise ValueError('coherent point source ancestry missing target records')
    seen_query_rows = set()
    for target in targets:
        if not isinstance(target, dict) or target.get('status') != 'PASS_FIRST_UNION_EXIT':
            raise ValueError('coherent point target row is not a certified first union exit')
        query_row = target.get('query_row')
        if type(query_row) is not int or not 0 <= query_row < candidate_points or query_row in seen_query_rows:
            raise ValueError('coherent point target query row is invalid or duplicated')
        seen_query_rows.add(query_row)
        provenance = target.get('source_normal_provenance')
        if not isinstance(provenance, dict):
            raise ValueError('coherent point source ancestry record missing')
        face_rows = provenance.get('incident_candidate_face_rows')
        if (not isinstance(face_rows, list) or not face_rows or
            any(type(row) is not int or row < 0 or row >= candidate_faces for row in face_rows) or
            len(set(face_rows)) != len(face_rows)):
            raise ValueError('coherent point incident source-face ancestry is invalid')
        if provenance.get('unique_incident_face_count') != len(face_rows):
            raise ValueError('coherent point incident source-face ancestry count mismatch')
        weights = provenance.get('source_face_area_weights')
        if not isinstance(weights, list) or len(weights) != len(face_rows):
            raise ValueError('coherent point source-face ancestry weights are incomplete')
        weight_rows = []
        area_sum = 0.0
        for item in weights:
            if not isinstance(item, dict):
                raise ValueError('coherent point source-face ancestry weight is malformed')
            row = item.get('candidate_face_row')
            area = item.get('source037_area_m2')
            normal = np.asarray(item.get('registered_source_normal_world_unit'), dtype=np.float64)
            ancestry = item.get('source_ancestry')
            if type(row) is not int or row not in face_rows:
                raise ValueError('coherent point source-face ancestry weight row mismatch')
            if not isinstance(area, (int, float)) or not np.isfinite(area) or area <= 0.0:
                raise ValueError('coherent point source-face ancestry area is invalid')
            if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1.0, rtol=0.0, atol=2e-12):
                raise ValueError('coherent point source-face ancestry normal is invalid')
            if not isinstance(ancestry, dict) or ancestry.get('kind') not in ('original_NHSKIN_face', 'original_coarse_O_face'):
                raise ValueError('coherent point registered source ancestry kind is missing')
            source_row = ancestry.get('row')
            if type(source_row) is not int or source_row < 0:
                raise ValueError('coherent point registered source ancestry row is invalid')
            weight_rows.append(row)
            area_sum += float(area)
        if set(weight_rows) != set(face_rows) or len(weight_rows) != len(set(weight_rows)):
            raise ValueError('coherent point source-face ancestry does not cover incident faces exactly')
        reported_area = provenance.get('source037_area_sum_m2')
        if (not isinstance(reported_area, (int, float)) or not np.isfinite(reported_area) or reported_area <= 0.0 or
            not np.isclose(float(reported_area), area_sum, rtol=2e-12, atol=0.0)):
            raise ValueError('coherent point source-face ancestry area total mismatch')
        ratio = provenance.get('resultant_over_area_sum')
        if not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or not 1e-12 < ratio <= 1.0 + 2e-12:
            raise ValueError('coherent point source-normal ancestry resultant is invalid')
    return sorted(seen_query_rows)


def run_point_targets(m185,state,candidate,out,cache_dir=None):
    point=Path(state['point_summary']);ip=read(point);query=point.parent/'query'
    target_dir=out/'point-targets'
    if cache_dir is not None:
        expected=INITIAL_TARGET_CACHE_PINS
        source=Path(cache_dir)/'point-targets'
        source_report=source/'nearest-union-exits.json';source_npz=source/'nearest-union-exits.npz'
        if sha(candidate)!=expected['candidate_sha256'] or sha(point)!=expected['point_summary_sha256']:
            raise ValueError('334 point cache is only valid for its exact initial candidate and point summary')
        if not source_report.is_file() or not source_npz.is_file() or sha(source_report)!=expected['point_report_sha256'] or sha(source_npz)!=expected['point_npz_sha256']:
            raise ValueError('334 point cache report/NPZ hash mismatch')
        target_dir.mkdir()
        shutil.copy2(source_report,target_dir/'nearest-union-exits.json')
        shutil.copy2(source_npz,target_dir/'nearest-union-exits.npz')
        write(target_dir/'cache-use.json',{'source_dir':str(source),'source_report_sha256':sha(source_report),
            'source_npz_sha256':sha(source_npz),'candidate_sha256':sha(candidate),
            'point_summary_sha256':sha(point),'matched_cache':True,'native_admission':False,'accepted_physical_steps':0})
        npz=target_dir/'nearest-union-exits.npz';report=target_dir/'nearest-union-exits.json'
    elif int(ip.get('inside_nodes',-1))==0:
        target_dir=out/'point-targets';target_dir.mkdir()
        npz=target_dir/'nearest-union-exits.npz';report=target_dir/'nearest-union-exits.json'
        run([PYTHON,POINT_EMITTER,'--candidate',candidate,'--summary',point,'--query-dir',query,
             '--npz',npz,'--report',report],target_dir/'targets.log',slot=True)
    elif cache_dir is None:
        run([PYTHON,POINT_BUILDER,'--candidate',candidate,'--point-summary',point,
             '--query-dir',query,'--output-dir',target_dir,'--clearance-m',CONSTRUCTION_CLEARANCE_M],
             out/'point-targets.log',slot=True)
        npz=target_dir/'nearest-union-exits.npz';report=target_dir/'nearest-union-exits.json'
    t=read(report)
    n=int(ip['inside_nodes'])
    if t.get('status')!='PASS_FIRST_UNION_EXIT_TARGETS_ONLY' or t.get('passing_targets') != n:
        raise ValueError('coherent point builder did not certify every inside row')
    inputs=t.get('inputs',{})
    if inputs.get('candidate_sha256')!=sha(candidate):
        raise ValueError('coherent point targets are for a different candidate')
    if inputs.get('point_summary_sha256') != sha(point):
        raise ValueError('coherent point targets are for a different point summary')
    query_receipt=query/'query-receipt.json'
    if not query_receipt.is_file() or inputs.get('query_receipt_sha256') != sha(query_receipt):
        raise ValueError('coherent point targets are for a different query receipt')
    for key, filename in (('query_points_sha256','points.csv'), ('query_nodes_sha256','nodes.csv'),
                          ('query_pairs_sha256','pairs.csv')):
        path=query/filename
        if not path.is_file() or inputs.get(key) != sha(path):
            raise ValueError('coherent point targets are for a different matched query export: '+filename)
    outputs=t.get('outputs',{})
    if outputs.get('npz_sha256') != sha(npz):
        raise ValueError('coherent point target NPZ hash does not match its report')
    z=np.load(npz,allow_pickle=False)
    if n>0:
        unavailable = t.get('source_normal_unavailable_rows')
        if unavailable is not None and unavailable != []:
            raise ValueError('coherent point report marks source normals unavailable')
        candidate_data=np.load(candidate,allow_pickle=False)
        candidate_points=len(candidate_data['points_world_m'])
        candidate_faces=len(candidate_data['source_O_faces'])
        if (t.get('query_rows') != n or t.get('inside_query_nodes') != n or
            t.get('all_bank_ray_results') != n):
            raise ValueError('coherent point report target inventory is incomplete')
        targets=t.get('targets')
        report_rows=validate_point_source_ancestry(targets, candidate_points, candidate_faces)
        for key in ('pass_mask','source_normal_available_mask','target_exactly_outside_all185'):
            if key not in z or z[key].shape != (n,) or not np.all(z[key]):
                raise ValueError('coherent point endpoint/source proof incomplete: '+key)
        if ('query_rows' not in z or z['query_rows'].shape != (n,) or
            z['query_rows'].dtype.kind not in 'iu' or not np.array_equal(z['query_rows'],report_rows)):
            raise ValueError('coherent point NPZ query-row identity differs from its report')
        target_points=np.asarray(z.get('target_world_f32_m'),dtype=np.float32)
        report_points=np.asarray([r.get('target_world_f32_m') for r in targets],dtype=np.float32)
        if target_points.shape != (n,3) or not np.isfinite(target_points).all() or not np.array_equal(target_points,report_points):
            raise ValueError('coherent point NPZ target coordinates differ from its report')
        directions=np.asarray(z.get('proposal_direction_world_unit'),dtype=np.float64)
        normals=np.asarray(z.get('source_surface_normal_world_unit'),dtype=np.float64)
        report_directions=np.asarray([r.get('proposal_direction_world_unit') for r in targets],dtype=np.float64)
        report_normals=np.asarray([r.get('source_surface_normal_world_unit') for r in targets],dtype=np.float64)
        if (directions.shape != (n,3) or normals.shape != (n,3) or
            not np.isfinite(directions).all() or not np.isfinite(normals).all() or
            not np.array_equal(directions,report_directions) or not np.array_equal(normals,report_normals) or
            not np.allclose(np.linalg.norm(directions,axis=1),1.0,rtol=0.0,atol=2e-12) or
            not np.allclose(np.linalg.norm(normals,axis=1),1.0,rtol=0.0,atol=2e-12)):
            raise ValueError('coherent point NPZ source directions differ from its report')
        for row in targets:
            if (row.get('target_exactly_outside_all185') is not True or
                row.get('end_outside_all185_by_event_parity') is not True or
                row.get('exact_endpoint_bank_count') != 185):
                raise ValueError('coherent point report lacks complete all-185 endpoint proof')
        distances=np.asarray(z.get('target_distance_m'),dtype=np.float64)
        report_distances=np.asarray([r.get('target_distance_m') for r in targets],dtype=np.float64)
        if (distances.shape != (n,) or not np.isfinite(distances).all() or np.any(distances <= 0.0) or
            not np.array_equal(distances,report_distances)):
            raise ValueError('coherent point NPZ target distances differ from its report')
    else:
        if (t.get('zero_targets_emitted') is not True or t.get('query_rows') != 0 or
            t.get('inside_query_nodes') != 0 or t.get('all_bank_ray_results') != 0 or t.get('targets') != []):
            raise ValueError('235 empty target report is not a complete zero-row result')
        for key, shape in (('pass_mask',(0,)), ('source_normal_available_mask',(0,)),
                            ('target_world_f32_m',(0,3)), ('proposal_direction_world_unit',(0,3)),
                            ('source_surface_normal_world_unit',(0,3))):
            if key not in z or z[key].shape != shape:
                raise ValueError('235 empty target NPZ has an invalid '+key)
    return npz,report

def run_face_targets(candidate: Path, surface: Path, out: Path, first_cache: Path | None = None, cache_dir: Path | None = None):
    target_dir = out / 'face-targets'
    if first_cache is not None and cache_dir is not None:
        raise ValueError('choose one face-target cache source')
    if cache_dir is not None:
        expected=INITIAL_TARGET_CACHE_PINS
        source=Path(cache_dir)/'face-targets'
        source_report=source/'report.json';source_npz=source/'targets.npz';source_rows=source/'target-records.jsonl'
        if sha(candidate)!=expected['candidate_sha256'] or sha(surface)!=expected['surface_report_sha256']:
            raise ValueError('334 face cache is only valid for its exact initial candidate and surface report')
        if (not source_report.is_file() or not source_npz.is_file() or not source_rows.is_file() or
            sha(source_report)!=expected['face_report_sha256'] or sha(source_npz)!=expected['face_npz_sha256'] or
            sha(source_rows)!=expected['face_targets_jsonl_sha256']):
            raise ValueError('334 face cache report/NPZ/record hashes mismatch')
        cached=read(source_report)
        if (cached.get('runner_sha256')!=sha(FACE_BUILDER) or cached.get('direction_mode')!='outward-nearest' or
            cached.get('inputs',{}).get('candidate_sha256')!=sha(candidate) or
            cached.get('inputs',{}).get('surface_report_sha256')!=sha(surface) or
            cached.get('matched_159_complete') is not True or
            cached.get('passing_targets')!=cached.get('crossing_pairs') or
            cached.get('passing_targets')!=cached.get('matched_surface_bank_triangle_pairs') or
            cached.get('passing_targets')!=surface_pair_count(read(surface),expected_faces=41733)):
            raise ValueError('334 face cache is not a complete matched all-pair result')
        target_dir.mkdir()
        shutil.copy2(source_report,target_dir/'report.json')
        shutil.copy2(source_npz,target_dir/'targets.npz')
        shutil.copy2(source_rows,target_dir/'target-records.jsonl')
        write(target_dir/'cache-use.json',{'source_dir':str(source),'source_report_sha256':sha(source_report),
            'source_npz_sha256':sha(source_npz),'source_target_records_sha256':sha(source_rows),
            'candidate_sha256':sha(candidate),'surface_report_sha256':sha(surface),
            'matched_cache':True,'native_admission':False,'accepted_physical_steps':0})
    elif first_cache is not None:
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
             '--output-dir', target_dir, '--direction-mode', 'outward-nearest', '--clearance-m', CONSTRUCTION_CLEARANCE_M],
            out / 'face-targets.log', slot=True)
    npz = target_dir / 'targets.npz'; report = target_dir / 'report.json'
    t = read(report)
    if t.get('runner_sha256') != sha(FACE_BUILDER) or t.get('direction_mode') != 'outward-nearest':
        raise ValueError('335 face-target runner/direction pin mismatch')
    if t.get('inputs', {}).get('candidate_sha256') != sha(candidate):
        raise ValueError('292 face-target report is for a different candidate')
    if not npz.is_file():
        raise ValueError('335 face targets NPZ missing')
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

def emit_gradient_checked_trial(base_points_f32, faces, support_rows, displacement_m, anchor_rows, max_component_gradient=0.5, max_sweeps=64):
    base = np.asarray(base_points_f32)
    face_rows = np.asarray(faces)
    support = np.asarray(support_rows)
    anchors = np.asarray(anchor_rows)
    delta = np.asarray(displacement_m, dtype=np.float64)
    if base.dtype != np.dtype(np.float32) or base.ndim != 2 or base.shape[1] != 3 or not np.isfinite(base).all():
        raise ValueError('base coordinates must be finite stored Float32 (N,3)')
    if face_rows.ndim != 2 or face_rows.shape[1] != 3 or face_rows.dtype.kind not in 'iu':
        raise ValueError('faces must be integer (F,3)')
    if support.ndim != 1 or support.dtype.kind not in 'iu' or len(np.unique(support)) != len(support) or np.any(support < 0) or np.any(support >= len(base)):
        raise ValueError('support rows must be unique in-range integers')
    if delta.shape != (len(support), 3) or not np.isfinite(delta).all():
        raise ValueError('displacement_m must be finite (len(support),3)')
    if anchors.ndim != 1 or anchors.dtype.kind not in 'iu' or len(np.unique(anchors)) != len(anchors) or np.any(anchors < 0) or np.any(anchors >= len(base)):
        raise ValueError('anchor rows must be unique in-range integers')
    gradient_owner = load_module(LOCAL_HELPER_339, 'edge_gradient_339')
    if sha(LOCAL_HELPER_339) != PINS['local_step_helper_339_sha256']:
        raise ValueError('339 edge-gradient helper hash mismatch')
    full_displacement_m = np.zeros(base.shape, dtype=np.float64)
    full_displacement_m[support] = delta
    full_displacement_m[anchors] = 0.0
    scales, emitted, receipt = gradient_owner.enforce_emitted_edge_gradient(
        base, full_displacement_m, np.ones(len(base), dtype=np.float64), gradient_owner.mesh_unique_edges(face_rows),
        max_component_gradient=max_component_gradient, max_sweeps=max_sweeps)
    if emitted is not None:
        emitted[anchors] = base[anchors]
    return scales, emitted, receipt


def validate_fit_scope_arrays(base_points, candidate_points, support_rows, anchor_rows):
    """Prove a later fit changed only its explicit support and preserved anchors bitwise."""
    base = np.asarray(base_points)
    candidate = np.asarray(candidate_points)
    raw_support = np.asarray(support_rows)
    raw_anchors = np.asarray(anchor_rows)
    if base.dtype != np.float32 or candidate.dtype != np.float32 or base.shape != candidate.shape or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("fit-scope points must be matching float32 (N,3) arrays")
    n = len(base)
    if raw_support.ndim != 1 or raw_support.dtype.kind not in "iu":
        raise ValueError("fit support rows must be a one-dimensional integer array")
    if raw_anchors.ndim != 1 or raw_anchors.dtype.kind not in "iu":
        raise ValueError("typed anchor rows must be a one-dimensional integer array")
    support = raw_support.astype(np.int64, copy=False)
    anchors = raw_anchors.astype(np.int64, copy=False)
    if len(np.unique(support)) != len(support) or np.any(support < 0) or np.any(support >= n):
        raise ValueError("fit support rows are duplicate or outside the point domain")
    if len(np.unique(anchors)) != len(anchors) or np.any(anchors < 0) or np.any(anchors >= n):
        raise ValueError("typed anchor rows are duplicate or outside the point domain")
    if not np.isfinite(base).all() or not np.isfinite(candidate).all():
        raise ValueError("fit-scope points contain nonfinite coordinates")
    b = np.ascontiguousarray(base).view(np.uint32).reshape(n, 3)
    c = np.ascontiguousarray(candidate).view(np.uint32).reshape(n, 3)
    if not np.array_equal(b[anchors], c[anchors]):
        raise ValueError("fit changed a typed coarse anchor")
    outside_mask = np.ones(n, dtype=bool)
    outside_mask[support] = False
    if not np.array_equal(b[outside_mask], c[outside_mask]):
        raise ValueError("fit changed a row outside its explicit LP support")
    changed = np.flatnonzero(np.any(b != c, axis=1))
    if np.any(np.isin(changed, anchors)) or np.any(~np.isin(changed, support)):
        raise ValueError("changed rows are not a subset of support excluding typed anchors")
    return {
        "point_rows": n, "support_rows": int(len(support)),
        "anchor_rows": int(len(anchors)), "changed_rows": int(len(changed)),
        "outside_support_bitwise_unchanged": True,
        "typed_anchors_bitwise_unchanged": True,
        "changed_rows_subset_support_minus_anchors": True,
    }


def ensure_fit_scope(base_path: Path, trial_path: Path, support_rows):
    base = np.load(base_path, allow_pickle=False)
    trial = np.load(trial_path, allow_pickle=False)
    if "typed_coarse_anchor_output_rows_018" not in base.files or "typed_coarse_anchor_output_rows_018" not in trial.files:
        raise ValueError("typed anchor identity rows are missing")
    anchors = np.asarray(base["typed_coarse_anchor_output_rows_018"])
    trial_anchors = np.asarray(trial["typed_coarse_anchor_output_rows_018"])
    if not np.array_equal(anchors, trial_anchors):
        raise ValueError("trial changed typed anchor identity rows")
    x = np.asarray(base["points_world_m"])
    y = np.asarray(trial["points_world_m"])
    if not np.array_equal(base["source_O_faces"], trial["source_O_faces"]):
        raise ValueError("trial changed surface topology")
    return validate_fit_scope_arrays(x, y, support_rows, anchors)


def trial_support_rows(trial_path: Path):
    """Read the frozen LP support under the legacy 333 or current 346 NPZ key.

    If both aliases are present, they must describe the same rows; this keeps
    compatibility explicit and prevents schema drift from silently changing
    which vertices the backtracker is allowed to move.
    """
    keys = ("continuation_333_fit_support_rows", "continuation_346_fit_support_rows")
    with np.load(trial_path, allow_pickle=False) as trial:
        present = [key for key in keys if key in trial.files]
        if not present:
            raise ValueError("trial is missing its frozen LP support rows")
        arrays = [np.asarray(trial[key]) for key in present]
        for key, rows in zip(present, arrays):
            if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(np.unique(rows)) != len(rows):
                raise ValueError(f"{key} is malformed")
        if len(arrays) == 2 and not np.array_equal(arrays[0], arrays[1]):
            raise ValueError("legacy and current trial support rows disagree")
        return arrays[-1].astype(np.int64, copy=True)


def typed_bone_pairs(report):
    owner=load_module(FIXED_EMITTER,'fixed_emitter_320_discovery')
    base=read(Path(read(DEFAULT_STATE)['surface_report']))
    expected=[(b['body'],b['stable_bone_id'])for b in base['banks']]
    return owner.validate_audit_report(report,
        candidate_sha256=report.get('candidate_sha256'),
        point_count=20889,face_count=41733,expected_bank_keys=expected,
        expected_script_sha256=PINS['audit159_sha256'],
        expected_predicate_sha256=SEED_SURFACE_PREDICATE_SHA256)


def complete_bone_reports():
    reports=[]
    for root in (*configured_seed_roots(), CURRENT_OUT):
        for path in sorted(root.rglob('audit-159/report.json')):
            report=read(path)
            if report.get('complete') is not True:
                raise ValueError('incomplete retained159 report: '+str(path))
            reports.append(path)
    return sorted(set(reports))

def learned_fixed_bone_pairs(state):
    base=typed_bone_pairs(read(Path(state['surface_report'])))
    result=set()
    for path in complete_bone_reports():
        result.update(typed_bone_pairs(read(path))-base)
    return result

def prepare_fixed_boundary(state, out):
    reports=complete_bone_reports()
    if not reports:
        raise ValueError('fixed-boundary learning has no complete trial audits')
    target=out/'fixed-boundary'
    args=[PYTHON,FIXED_EMITTER,'--base-candidate',state['candidate'],
          '--base-surface-report',state['surface_report'],'--output-dir',target]
    for path in reports:args+=['--trial-surface-report',path]
    run(args,out/'fixed-boundary.log',slot=True)
    report=target/'report.json';npz=target/'introduced-skin-bone-barriers.npz'
    data=read(report)
    if data['introduced_pair_count'] != len(learned_fixed_bone_pairs(state)):
        raise ValueError('fixed-boundary export lost typed witnesses')
    return npz,report

def validate_fitter_optins(report: dict, *, require_solution: bool) -> None:
    """Validate opt-in receipts when 347 emitted a candidate; LP_NO_PROPOSAL has no candidate receipt."""
    if not require_solution:
        return
    if report.get("emitted_clearance_proxy_gate_deferred") is not True:
        raise ValueError("347 did not record the requested emitted-clearance proxy deferral")
    if report.get("stage1_feasible_timeout_fallback_opt_in") is not True:
        raise ValueError("347 did not record the requested stage-1 fallback opt-in")
    solver = report.get("solver")
    if not isinstance(solver, dict) or solver.get("success") is not True:
        raise ValueError("347 raw candidate lacks a successful or certified-fallback solver receipt")
    stage1 = solver.get("stage1")
    stage2 = solver.get("stage2")
    if not isinstance(stage1, dict) or not isinstance(stage2, dict):
        raise ValueError("347 solver receipt is missing its lexicographic stage records")
    if solver.get("fallback_used") is True:
        certificate = solver.get("stage1_primal_certificate")
        if (solver.get("solution_source") != "certified_stage1_primal_after_stage2_time_limit"
                or stage1.get("success") is not True or stage1.get("status") != 0
                or stage2.get("success") is not False or stage2.get("status") != 1
                or stage2.get("time_limit") is not True
                or "time limit" not in str(stage2.get("message", "")).lower()
                or not isinstance(certificate, dict) or certificate.get("passed") is not True
                or certificate.get("status") != "CERTIFIED_STAGE1_PRIMAL"):
            raise ValueError("347 timeout fallback lacks the required independently certified stage-1 primal")
    elif stage1.get("success") is not True or stage2.get("success") is not True:
        raise ValueError("347 solver stages are incomplete without a certified timeout fallback")


def run_fit_attempt(state: dict, candidate: Path, self_pairs: np.ndarray, point_npz: Path,
                    point_report: Path, face_npz: Path, face_report: Path, out: Path,
                    lp_method: str = 'highs'):
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
        '--fixed-prefix-nodes', FIT_FIXED_PREFIX, '--max-edge-gradient', '0.5', '--slack-tolerance', '0.05',
        '--time-limit-s', '60', '--area-scp-iterations', '4', '--allow-stage1-feasible-timeout-fallback', '--defer-emitted-clearance-proxy', '--lp-method', lp_method, '--output', proposal]
    fixed_npz, fixed_report = prepare_fixed_boundary(state, out)
    args += ['--fixed-boundary-npz', fixed_npz, '--fixed-boundary-report', fixed_report, '--fixed-boundary-owner', FIXED_OWNER]
    run(args, out / 'fit.log', slot=True, timeout=900)
    raw_lp = proposal / 'raw-lp-candidate.npz'
    report_path = proposal / 'report.json'
    if not raw_lp.is_file():
        if report_path.is_file():
            no_candidate_report = read(report_path)
            validate_fitter_optins(no_candidate_report, require_solution=False)
            write(out / 'no-raw-lp.json', {'report': str(report_path), 'report_sha256': sha(report_path),
                                           'status': no_candidate_report.get('status'), 'native_admission': False,
                                           'accepted_physical_steps': 0})
        return None
    fit_report = read(report_path)
    validate_fitter_optins(fit_report, require_solution=True)
    if fit_report.get('support', {}).get('threatened_pairs') != len(self_pairs):
        raise ValueError('fitter did not retain every requested learned face pair')
    if fit_report.get('support', {}).get('fixed_prefix_nodes') != FIT_FIXED_PREFIX:
        raise ValueError('fitter used the wrong fixed-prefix setting')
    actual_pairs = {tuple(sorted(map(int, r['faces']))) for r in fit_report['constraints']['actual_pair_records']}
    if actual_pairs != {tuple(map(int, row)) for row in self_pairs}:
        raise ValueError('fitter pair-record identities differ from exact learned pair set')
    expected_inputs = {'base_candidate_sha256':sha(candidate), 'self_pairs_sha256':sha(pairs_path),
        'point_target_npz_sha256':sha(point_npz), 'point_target_report_sha256':sha(point_report),
        'face_target_npz_sha256':sha(face_npz), 'face_target_report_sha256':sha(face_report)}
    if any(fit_report['inputs'].get(k) != v for k,v in expected_inputs.items()):
        raise ValueError('fitter used different base, exact pairs or target inputs')
    fixed_count=read(fixed_report)['introduced_pair_count']
    if fit_report['constraints'].get('fixed_boundary_typed_pair_count') != fixed_count:
        raise ValueError('fitter did not retain every fixed bone barrier')
    fr=fit_report.get('fixed_boundary_receipt') or {}
    if fr.get('npz_sha256') != sha(fixed_npz) or fr.get('report_sha256') != sha(fixed_report) or fr.get('owner_sha256') != sha(FIXED_OWNER):
        raise ValueError('fitter fixed-boundary input/owner hash mismatch')
    eg=fit_report['constraints']['edge_gradient']
    if eg.get('enabled') is not True or eg.get('max_component_gradient') != 0.5 or fit_report['constraints']['edge_gradient_rows'] <= 0:
        raise ValueError('requested existing neighbor-gradient proposal constraint missing')
    write(out/'verified-fit-inputs.json', {'matched_inputs':expected_inputs, 'pair_count':len(self_pairs), 'edge_gradient':eg,
        'native_admission':False, 'accepted_physical_steps':0})
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
    if fit_report.get('support', {}).get('support_nodes') != len(support):
        raise ValueError('fitter report support count differs from raw LP support rows')
    X = np.asarray(z['points_world_m'], dtype=np.float32)
    anchors = np.asarray(z['typed_coarse_anchor_output_rows_018'], dtype=np.int64)
    raw_points = np.asarray(raw['points_world_m'])
    if raw_points.dtype != np.float32 or raw_points.shape != X.shape or not np.isfinite(raw_points).all():
        raise ValueError('338 raw emitted points must be finite Float32 on the base point domain')
    if not np.array_equal(raw['source_O_faces'], z['source_O_faces']):
        raise ValueError('338 raw LP topology differs from its matched base')
    outside_support = np.ones(len(X), dtype=bool); outside_support[support] = False
    if not np.array_equal(raw_points[outside_support].view(np.uint32), X[outside_support].view(np.uint32)):
        raise ValueError('338 raw LP changed rows outside its explicit support')
    raw_delta_m = raw_points[support].astype(np.float64) - X[support].astype(np.float64)
    edge_scales, Xtrial, edge_gradient_receipt = emit_gradient_checked_trial(
        X, z['source_O_faces'], support, raw_delta_m, anchors, max_component_gradient=0.5, max_sweeps=64)
    if Xtrial is not None and np.all(edge_scales == 1.0) and not np.array_equal(Xtrial.view(np.uint32), raw_points.view(np.uint32)):
        raise AssertionError('339 exact gradient check changed a passing 338 raw F32 candidate')
    if Xtrial is None or edge_gradient_receipt.get('candidate_usable') is not True:
        write(out / 'initial-edge-gradient-failed.json', {
            'status': edge_gradient_receipt.get('status', 'EDGE_GRADIENT_UNRESOLVED_FAIL_CLOSED'),
            'edge_gradient_receipt': edge_gradient_receipt,
            'native_admission': False, 'accepted_physical_steps': 0})
        return None
    Xtrial[anchors] = X[anchors]
    validate_fit_scope_arrays(X, Xtrial, support, anchors)
    if not np.isfinite(Xtrial).all():
        raise ValueError('gradient-checked Float32 emission contains nonfinite coordinates')
    write(out / 'initial-edge-gradient.json', {
        'helper': str(LOCAL_HELPER_339), 'helper_sha256': sha(LOCAL_HELPER_339),
        'receipt': edge_gradient_receipt,
        'proposal': 'exact stored 338 raw F32 points; displacement derived from those points, avoiding a second mm-to-m reconstruction',
        'raw_lp_scaled_node_count': int(len(support)),
        'guard_reduced_node_count': int(edge_gradient_receipt['edge_scale_updates']),
        'edge_scales_sha256': hashlib.sha256(np.ascontiguousarray(edge_scales).view(np.uint8)).hexdigest(),
        'emitted_points_sha256': hashlib.sha256(np.ascontiguousarray(Xtrial).view(np.uint8)).hexdigest(),
        'exact_emitted_float32_gradient_checked': True,
        'native_admission': False, 'accepted_physical_steps': 0})
    require_headroom('gradient-checked candidate write')
    trial = out / 'trial-halfstep.npz'
    payload = {k: z[k] for k in z.files}; payload['points_world_m'] = Xtrial
    payload['continuation_346_fit_support_rows'] = support.copy()
    np.savez_compressed(trial, **payload)
    fit_scope = ensure_fit_scope(candidate, trial, support)
    write(out / 'trial-provenance.json', {
        'base_candidate': str(candidate), 'base_candidate_sha256': sha(candidate),
        'raw_lp_candidate': str(raw_lp), 'raw_lp_candidate_sha256': sha(raw_lp),
        'fitter': str(SOLVER), 'fitter_sha256': sha(SOLVER), 'halfstep_alpha': 1.0,
        'initial_emitted_gradient_receipt': str(out / 'initial-edge-gradient.json'),
        'initial_emitted_gradient_sha256': sha(out / 'initial-edge-gradient.json'),
        'initial_prefix_validation_nodes': FIXED_PREFIX if not state.get('_resume_verified_hashes') else 0,
        'fit_fixed_prefix_nodes': FIT_FIXED_PREFIX,
        'fit_scope_receipt': fit_scope,
        'trial': str(trial), 'trial_sha256': sha(trial),
        'native_admission': False, 'accepted_physical_steps': 0,
    })
    return trial

def run_backtracker(state: dict, trial: Path, out: Path, helper: Path, entry: str,
                    max_audits: int, max_passes: int, support_rows):
    support_rows = np.asarray(support_rows, dtype=np.int64)
    ensure_fit_scope(Path(state['candidate']), trial, support_rows)
    out.mkdir(exist_ok=False)
    cfg = {
      'state': state, 'trial': str(trial), 'out': str(out / 'backtrack'),
      'parent_fit': state['parent_fit'], 'helper': str(helper), 'helper_sha256': sha(helper),
      'helper_entry': entry, 'initial_scale': 1.0, 'zero_below': 1.0 / 64.0,
      'maximum_exact_audits': max_audits, 'maximum_construction_passes': max_passes,
      'fixed_prefix_nodes': FIT_FIXED_PREFIX, 'fit_support_rows': support_rows.tolist(),
      'area_reference_npz': str(AREA_REFERENCE),
      'area_reference_sha256': sha(AREA_REFERENCE),
      'area_reference_points_key': AREA_POINTS_KEY, 'area_reference_faces_key': AREA_FACES_KEY,
    }
    cfgpath = out / 'backtrack-config.json'; write(cfgpath, cfg)
    selected_runner = LOCAL_BACKTRACKER if entry == 'reduce_witness_node_scales_with_edge_gradient' else BACKTRACKER
    if selected_runner == LOCAL_BACKTRACKER:
        cfg['edge_gradient_safe'] = True
        cfg['max_component_gradient'] = 0.5
        cfg['max_edge_sweeps'] = 64
        cfg['edge_gradient_helper_sha256'] = sha(LOCAL_HELPER_339)
        write(cfgpath, cfg)
    run([PYTHON, selected_runner, cfgpath], out / 'backtracker.log', slot=False, timeout=5400)
    selected = out / 'backtrack/selected-result.json'
    if selected.is_file():
        result = read(selected)
        selected_scope = ensure_fit_scope(Path(state['candidate']), Path(result['candidate']), support_rows)
        write(out / 'selected-fit-scope.json', {
            'base_candidate_sha256': sha(Path(state['candidate'])),
            'trial_sha256': sha(trial), 'selected_candidate_sha256': sha(Path(result['candidate'])),
            'support_rows': int(len(support_rows)), 'fit_scope_receipt': selected_scope,
            'selected_candidate_is_construction_only': True, 'native_admission': False,
            'accepted_physical_steps': 0,
        })
        return selected
    return None

def valid_limits(rounds, attempts):
    return type(rounds) is int and type(attempts) is int and 1 <= rounds <= 3 and 1 <= attempts <= 4

def self_test():
    assert FIXED_PREFIX == 18430 and FIT_FIXED_PREFIX == 0
    assert AREA_POINTS_KEY == 'area_reference_037_points_world_m_f64'
    assert AREA_FACES_KEY == 'area_reference_037_source_O_faces'
    assert sha(FACE_BUILDER) == PINS['face_builder_292_sha256']
    assert sha(SOLVER) == PINS['solver_347_sha256']
    assert LP_METHODS == ('highs','highs-ipm','highs-ds')
    empty = np.empty((0, 2), dtype=np.int64)
    assert empty.shape == (0, 2) and empty.dtype.kind in 'iu'
    assert valid_limits(3,4) and not valid_limits(4,1) and not valid_limits(1,5) and not valid_limits(True,1)
    print(json.dumps({'status': 'PASS', 'checks': ['fixed prefix and area-reference keys', '292 builder pin', '347 LP-method owner pin', 'empty pair seed shape', 'attempt and round limits']}))

def main():
    ap = argparse.ArgumentParser(description='Bounded 346 continuation; source-only preflight by default.')
    ap.add_argument('--state', type=Path, default=DEFAULT_STATE)
    ap.add_argument('--resume-state-sha256', help='required exact SHA-256 to resume a selected construction-progress state')
    ap.add_argument('--resume-point-summary-sha256', help='required only if the state does not embed its point-summary SHA')
    ap.add_argument('--resume-surface-report-sha256', help='required only if the state does not embed its surface-report SHA')
    ap.add_argument('--seed-audit-root', type=Path, action='append', default=[], help='additional in-scope exact self-audit root; repeatable')
    ap.add_argument('--lp-method', choices=('highs','highs-ipm','highs-ds'), default='highs')
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
    if not valid_limits(args.rounds,args.attempts_per_round):
        ap.error('rounds must be in [1,3] and attempts-per-round in [1,4]')
    if args.resume_state_sha256 is not None and args.first_face_target_dir is not None:
        ap.error('--first-face-target-dir is registered only for the pinned default seed; resumed states build fresh targets')
    if args.lp_method not in LP_METHODS:
        ap.error('lp-method must be highs, highs-ipm, or highs-ds')
    pins = verify_file_pins()
    configure_seed_audit_roots(args.seed_audit_root)
    state, candidate, faces, anchors = validate_state(args.state, resume_state_sha256=args.resume_state_sha256,
        resume_point_summary_sha256=args.resume_point_summary_sha256,
        resume_surface_report_sha256=args.resume_surface_report_sha256)
    archive_members, archive_receipt = seed_archive_index()
    seed_pairs, seed_receipts = collect_seed_pairs(faces, archive_members)
    if not seed_receipts:
        raise ValueError('continuation seed roots contain no complete exact self-audit pair receipts')
    if args.preflight_only:
        print(json.dumps({'status': 'PREFLIGHT_PASS_NO_GEOMETRY_LAUNCHED',
            'state': str(args.state), 'state_sha256': sha(args.state),
            'candidate': str(candidate), 'candidate_sha256': sha(candidate),
            'point_inside_nodes': read(Path(state['point_summary']))['inside_nodes'],
            'surface_pairs': surface_pair_count(read(Path(state['surface_report'])), expected_faces=len(faces)),
            'self_pair_seed_count': len(seed_pairs), 'self_pair_receipts': len(seed_receipts),
            'self_pair_seed_roots': [str(x) for x in configured_seed_roots()], 'seed_archive_recovery': archive_receipt, 'initial_prefix_validation_nodes': FIXED_PREFIX if args.resume_state_sha256 is None else 0,
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
      'resume_state_sha256_argument': args.resume_state_sha256,
      'resume_verified_hashes': state.get('_resume_verified_hashes'),
      'seed_self_pair_roots': [str(x) for x in (SEED_ROOT, EXTRA_SEED_ROOT, ADDITIONAL_SEED_ROOT, FOURTH_SEED_ROOT)], 'seed_self_pair_sources': seed_receipts,
      'seed_self_pair_unique_count': len(seed_pairs), 'source_files': pins,
      'solver_347': str(SOLVER), 'lp_method': args.lp_method, 'face_builder_292': str(FACE_BUILDER),
      'optional_first_face_target_cache': str(args.first_face_target_dir) if args.first_face_target_dir else None,
      'face_direction_mode': 'source-outward', 'global_backtracker_275': str(GLOBAL_HELPER),
      'local_backtracker_339': str(LOCAL_HELPER_339), 'backtracker_runner_306': str(BACKTRACKER),
      'gradient_safe_local_backtracker_340': str(LOCAL_BACKTRACKER),
      'area_reference_278': str(AREA_REFERENCE), 'area_reference_keys': [AREA_POINTS_KEY, AREA_FACES_KEY],
      'initial_prefix_validation_nodes': FIXED_PREFIX if args.resume_state_sha256 is None else 0,
      'resume_fit_fixed_prefix_nodes': FIT_FIXED_PREFIX,
      'initial_target_cache': str(INITIAL_TARGET_CACHE_DIR), 'initial_target_cache_pins': INITIAL_TARGET_CACHE_PINS,
      'rounds': args.rounds, 'attempts_per_round': args.attempts_per_round,
      'global_first': {'initial_scale': 1.0, 'zero_below': 1.0/64.0, 'maximum_exact_audits': 1, 'maximum_construction_passes': 16},
      'local_fallback_same_trial': {'initial_scale': 1.0, 'zero_below': 1.0/64.0, 'maximum_exact_audits': 8, 'maximum_construction_passes': 32},
      'point_builder':str(POINT_BUILDER), 'construction_clearance_m':CONSTRUCTION_CLEARANCE_M, 'all_seed_roots':[str(x) for x in configured_seed_roots()], 'fixed_boundary_owner':str(FIXED_OWNER), 'fixed_boundary_emitter':str(FIXED_EMITTER), 'early_pair_learning_maximum_LPs': 4, 'probe_exact_audits': 1, 'max_edge_gradient_proposal_mm_per_mm': 0.5, 'disk_floor_bytes': 200*1024**2, 'native_admission': False, 'accepted_physical_steps': 0,
    }
    write(OUT / 'configuration.json', configuration)
    write(OUT / 'initial-self-pair-seed.json', {'roots': [str(x) for x in configured_seed_roots()], 'unique_count': len(seed_pairs), 'sources': seed_receipts, 'archive_recovery': archive_receipt,
          'semantics': 'union of complete exact audits with candidate SHA and exact topology validation; includes all retained 337 audits; no older-topology/self-pair files loaded'})
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
        use_default_cache = round_index == 0 and args.resume_state_sha256 is None
        face_npz, face_report = run_face_targets(candidate, surface_path, rd, args.first_face_target_dir if use_default_cache else None, INITIAL_TARGET_CACHE_DIR if use_default_cache else None)
        point_npz, point_report = run_point_targets(m185, current_state, candidate, rd, INITIAL_TARGET_CACHE_DIR if use_default_cache else None)
        round_selected = None
        for attempt in range(args.attempts_per_round):
            attempt_dir = rd / f'attempt-{attempt+1:03d}'; attempt_dir.mkdir()
            pairs, receipts = accumulate_pairs(faces, archive_members)
            self_pairs = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
            pair_input = attempt_dir / 'self-pairs-input.npy'; np.save(pair_input, self_pairs)
            write(attempt_dir / 'self-pairs-input.json', {'array': str(pair_input), 'sha256': sha(pair_input),
                  'unique_pair_count': len(self_pairs), 'sources': receipts,
                  'seed_roots_included': [str(x) for x in configured_seed_roots()], 'no_older_topology_pairs': True})
            fixed_pairs_before = learned_fixed_bone_pairs(current_state)
            trial = run_fit_attempt(current_state, candidate, self_pairs, point_npz, point_report, face_npz, face_report, attempt_dir, args.lp_method)
            if trial is None:
                write(attempt_dir / 'result.json', {'status': 'NO_RAW_LP_PROPOSAL', 'native_admission': False, 'accepted_physical_steps': 0})
                break
            fit_support_rows = trial_support_rows(trial)
            global_selected = run_backtracker(current_state, trial, attempt_dir / 'global-275', GLOBAL_HELPER,
                    'reduce_global_step_scales', 1, 16, fit_support_rows)
            selected = global_selected
            selected_mode = 'global-275-probe'
            probe_pairs, probe_receipts = accumulate_pairs(faces, archive_members)
            new_probe_pairs = probe_pairs - pairs
            write(attempt_dir/'early-self-pair-learning.json', {'before_pair_count':len(pairs),
                'after_pair_count':len(probe_pairs),'new_pair_count':len(new_probe_pairs),'sources':probe_receipts,
                'same_base_sha256':sha(candidate),'same_face_target_sha256':sha(face_report),'same_point_target_sha256':sha(point_report),
                'construction_only':True,'native_admission':False,'accepted_physical_steps':0})
            new_fixed_pairs = learned_fixed_bone_pairs(current_state) - fixed_pairs_before
            write(attempt_dir/'early-fixed-bone-learning.json', {'before_count':len(fixed_pairs_before),'new_count':len(new_fixed_pairs),'new_typed_pairs':sorted(new_fixed_pairs),'base_sha256':sha(candidate),'native_admission':False})
            if selected is None and (new_probe_pairs or new_fixed_pairs) and attempt+1 < args.attempts_per_round:
                write(attempt_dir/'result.json', {'status':'EXACT_SELF_WITNESSES_LEARNED_BEFORE_LOCAL_DAMPING',
                    'new_pair_count':len(new_probe_pairs),'new_fixed_bone_pair_count':len(new_fixed_pairs),'native_admission':False,'accepted_physical_steps':0})
                continue
            if selected is None:
                selected = run_backtracker(current_state, trial, attempt_dir / 'local-339-fallback', LOCAL_HELPER_339,
                        'reduce_witness_node_scales_with_edge_gradient', 8, 32, fit_support_rows)
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
                if fresh_pairs == pairs and learned_fixed_bone_pairs(current_state) == fixed_pairs_before:
                    write(attempt_dir / 'stop-repeated-input.json', {
                        'status': 'STOP_IDENTICAL_NEXT_LP_INPUT', 'reason': 'no selected progress and no newly observed exact self-pairs',
                        'pair_input_sha256': sha(pair_input), 'unique_pair_count': len(pairs),
                        'native_admission': False, 'accepted_physical_steps': 0})
                    break
                continue
            result = read(selected)
            # Bind resumable state to the exact emitted candidate and both exact audit files.
            for field, hash_field in (('candidate','candidate_sha256'),('point_summary','point_summary_sha256'),('surface_report','surface_report_sha256')):
                path=Path(result.get(field,''))
                if not path.is_file():
                    raise ValueError(f'selected backtracker state is missing {field}: {path}')
                actual_sha=sha(path)
                prior=result.get(hash_field)
                if prior not in (None,'') and prior!=actual_sha:
                    raise ValueError(f'selected backtracker {hash_field} does not match bytes')
                result[hash_field]=actual_sha
            if result.get('status')!='CONSTRUCTION_PROGRESS_ONLY' or result.get('native_admission') is not False or result.get('accepted_physical_steps')!=0:
                raise ValueError('selected backtracker result is not unadmitted construction progress')
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
    write(OUT / 'completed.json', {'status': 'BOUNDED_COHERENT_TARGET_346_SUPPORT_SCOPED_CONTINUATION_FINISHED', 'final_state': str(current_state_path),
          'final_candidate': current_state.get('candidate'), 'final_candidate_sha256': current_state.get('candidate_sha256'),
          'native_admission': False, 'accepted_physical_steps': 0})

if __name__ == '__main__':
    main()

