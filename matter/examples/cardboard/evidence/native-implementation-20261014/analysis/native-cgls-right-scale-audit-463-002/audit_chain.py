#!/usr/bin/env python3
"""Strict CGLS audit via a complete 463 -> 455 -> 419 state join.

No same-run operator capture is assumed. Reuse of the 419 assembled operator is
permitted only after the native 463 state sidecar exactly matches the 455 state
sidecar at the selected root/Newton, and the sealed 455-vs-419 join inputs and
reported exact comparisons are revalidated. This is a descriptive action audit,
not a solver certificate or physical qualification.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, struct, sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path('/Users/home/cardboard-implementation-20261014')
V1 = Path(__file__).with_name('audit_v1_parser.py')
JOIN_EXPECTED = ROOT / 'analysis/base-cgls-state-reader-abi-repair-148-001/capture-455-vs-419.json'
STATE455 = ROOT / 'solver/cgls-state-sidecar-observer-455/base-cgls.bin.state.json'
OP419 = ROOT / 'solver/two-stage-first-failure-rich-fd-419/operator-capture'
SOURCE149 = ROOT / 'snapshots/source-cgls-block-right-preconditioned-149'
SOURCE143 = ROOT / 'snapshots/source-cgls-primary-policy-143'
STATE_READER = ROOT / 'analysis/base-cgls-state-reader-abi-repair-148-001/tools/read_state_capture.py'
REQUIRED_STATE_ARRAYS = (
    'accepted_nodes', 'candidate_nodes', 'accepted_tetrahedra', 'candidate_tetrahedra',
    'accepted_material_history', 'candidate_material_state_buffer', 'objects', 'materials',
    'scalar_programs', 'expression_instructions', 'environment_parameters', 'schedulers',
    'adaptive_state', 'original_newton_correction', 'base_residual_rhs',
    'algorithmic_tangent_cache', 'algorithmic_tangent_cache_valid',
)
CONTROL_STATUS = {
    0: 'inactive_or_not_selected',
    1: 'running',
    2: 'iteration_cap_exhausted',
    3: 'zero_normal_residual',
    4: 'breakdown',
    5: 'invalid_input',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    v = json.loads(Path(path).read_text())
    if not isinstance(v, dict): raise ValueError(f'expected JSON object at {path}')
    return v


def load_v1():
    return __import__('audit_v1_parser')


def array_cmp(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape or aa.dtype != bb.dtype:
        return {'equal': False, 'shape_equal': aa.shape == bb.shape,
                'dtype_equal': str(aa.dtype) == str(bb.dtype),
                'left_shape': list(aa.shape), 'right_shape': list(bb.shape),
                'left_dtype': str(aa.dtype), 'right_dtype': str(bb.dtype)}
    ab, bbts = aa.tobytes(order='C'), bb.tobytes(order='C')
    if ab == bbts:
        return {'equal': True, 'count': int(aa.size), 'bytes': len(ab), 'word_mismatches': 0}
    # For float and opaque record arrays, report uint32 word differences. This is
    # not used to waive differences in the 463->455 join.
    if len(ab) % 4 == 0:
        wa = np.frombuffer(ab, dtype='<u4'); wb = np.frombuffer(bbts, dtype='<u4')
        mism = wa != wb
        signedzero = (((wa & 0x7fffffff) == 0) & ((wb & 0x7fffffff) == 0) & mism)
        return {'equal': False, 'count': int(aa.size), 'bytes': len(ab),
                'word_mismatches': int(mism.sum()), 'signed_zero_only_words': int(signedzero.sum()),
                'first_mismatch_words': np.flatnonzero(mism)[:8].astype(int).tolist()}
    return {'equal': False, 'count': int(aa.size), 'bytes': len(ab), 'byte_mismatches': sum(x != y for x,y in zip(ab,bbts))}


def run_history_comparison(case: Path) -> dict[str, Any]:
    """Compare the complete four-row history to the retained disabled/baseline runs.

    Only the selected bent step-1 attempt is expected to differ. This is a
    failed-prefix comparison and deliberately does not call the changed attempt
    an accepted physical step.
    """
    solver = ROOT / 'solver'
    baselines = {
        '459': solver / 'post-release-default-zero-parity-459',
        '453': solver / 'cgls-primary-after48-control1-453',
        '455': solver / 'cgls-state-sidecar-observer-455',
    }
    stable_top_level = (
        'compiled.nmatterpack', 'initial.obj', 'accepted_step_000001_bent.obj',
        'accepted_step_000001_held_reference.obj', 'accepted_step_000002_held_reference.obj',
        'material_state_000002_bent.json', 'mesh.json',
        'rejected_step_diagnostic_000002_bent.obj', 'result.json',
    )
    current_output = case / 'output'
    with (current_output / 'observations.csv').open(newline='') as f:
        current_rows = list(csv.DictReader(f))
    if len(current_rows) != 4:
        raise ValueError(f'expected complete four-row attempted history, got {len(current_rows)}')
    if [(r['step'], r['environment'], r['arm']) for r in current_rows] != [
        ('0','0','bent'), ('0','1','held_reference'), ('1','0','bent'), ('1','1','held_reference')]:
        raise ValueError('463 attempted-history row identities/order differ from the retained protocol')
    if current_rows[0]['step_accepted'] != '1' or current_rows[1]['step_accepted'] != '1' or \
       current_rows[2]['step_accepted'] != '0' or current_rows[3]['step_accepted'] != '1':
        raise ValueError('463 does not have the expected one-accepted-step failed prefix')
    comparisons: dict[str, Any] = {}
    for label, base in baselines.items():
        receipt_path = base / 'receipt.json'
        output = base / 'output'
        if not receipt_path.is_file() or not (output / 'observations.csv').is_file():
            raise ValueError(f'missing completed baseline {label}')
        receipt = load_json(receipt_path)
        if receipt.get('execution_started') is not True or receipt.get('timed_out') is True:
            raise ValueError(f'baseline {label} did not complete as an executed run')
        with (output / 'observations.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 4 or [(r['step'],r['environment'],r['arm']) for r in rows] != [
            ('0','0','bent'), ('0','1','held_reference'), ('1','0','bent'), ('1','1','held_reference')]:
            raise ValueError(f'baseline {label} attempted-history shape differs')
        exact_rows = [i for i in (0,1,3) if rows[i] == current_rows[i]]
        if exact_rows != [0,1,3]:
            raise ValueError(f'463 physical-history prefix/held row differs from baseline {label}: {exact_rows}')
        row2_keys = sorted(set(current_rows[2]) | set(rows[2]))
        row2_diff = {k:{'native463':current_rows[2].get(k),'baseline':rows[2].get(k)}
                     for k in row2_keys if current_rows[2].get(k) != rows[2].get(k)}
        # The selected attempted step must remain rejected in both runs. Record
        # its changed metrics; do not require or imply nonlinear acceptance.
        if current_rows[2]['step_accepted'] != '0' or rows[2]['step_accepted'] != '0':
            raise ValueError(f'baseline {label} does not retain the failed attempted step')
        stable_files = {}
        for name in stable_top_level:
            a, b = current_output / name, output / name
            if not a.is_file() or not b.is_file():
                raise ValueError(f'expected physical/history output missing for baseline {label}: {name}')
            ha,hb=sha(a),sha(b)
            stable_files[name]={'native463_sha256':ha,'baseline_sha256':hb,'byte_exact':ha==hb,'bytes':a.stat().st_size}
        if not all(item['byte_exact'] for item in stable_files.values()):
            raise ValueError(f'physical assets or result differ from baseline {label}')
        # Report, but do not promote, diagnostic artifacts that change with the
        # solver path. These include Armijo conditioning snapshots.
        diagnostic_differences=[]
        diag_a=current_output/'rejected-conditioning-000002'
        diag_b=output/'rejected-conditioning-000002'
        if diag_a.is_dir() and diag_b.is_dir():
            for a in sorted(x for x in diag_a.rglob('*') if x.is_file()):
                rel=a.relative_to(diag_a)
                b=diag_b/rel
                if not b.is_file() or sha(a)!=sha(b):
                    diagnostic_differences.append(rel.as_posix())
        comparisons[label]={
            'baseline_case':str(base),
            'baseline_receipt_sha256':sha(receipt_path),
            'receipt_exit_code':receipt.get('exit_code'),
            'receipt_timed_out':receipt.get('timed_out'),
            'attempted_row_count':4,
            'exact_rows_vs_463':[0,1,3],
            'step1_bent_attempt':{
                'native463':{k:current_rows[2].get(k) for k in ('status_code','object_index','fgmres_iterations','certificate_residual','certificate_correction','step_accepted','free_force_imbalance_l2_N','free_force_imbalance_max_N')},
                'baseline':{k:rows[2].get(k) for k in ('status_code','object_index','fgmres_iterations','certificate_residual','certificate_correction','step_accepted','free_force_imbalance_l2_N','free_force_imbalance_max_N')},
                'changed_fields':row2_diff,
                'classification':'different solver attempt; rejected in both runs',
            },
            'physical_output_hashes':stable_files,
            'physical_output_files_byte_exact':all(item['byte_exact'] for item in stable_files.values()),
            'changed_rejected_conditioning_files':diagnostic_differences,
            'observations_sha256':sha(current_output/'observations.csv'),
            'baseline_observations_sha256':sha(output/'observations.csv'),
            'manifest_sha256':sha(current_output/'manifest.json'),
            'baseline_manifest_sha256':sha(output/'manifest.json'),
        }
    return {
        'schema':'numi.cardboard.native_cgls_failed_prefix_comparison.v1',
        'current_run':str(case),
        'current_receipt_exit_code':load_json(case/'receipt.json').get('exit_code'),
        'attempted_rows':4,
        'accepted_steps':1,
        'attempted_step1_bent_status':'rejected',
        'exact_shared_rows':[0,1,3],
        'classification':'selected failed-prefix comparison only; not a convergence or solver qualification',
        'baselines':comparisons,
    }


def validate_prior_join(prior_path: Path, prior_state_path: Path, op_path: Path) -> dict[str, Any]:
    report = load_json(prior_path)
    if report.get('schema') != 'numi.matter.base_cgls_state_capture_419_join.v1':
        raise ValueError('455-to-419 evidence has unexpected schema')
    inp = report.get('inputs', {})
    expected = {
        'operator_capture': (str(op_path), sha(op_path)),
        'state_capture_manifest': (str(prior_state_path), sha(prior_state_path)),
        'state_capture_payload': (str(prior_state_path.parent / 'base-cgls.bin.state.bin'),
                                  sha(prior_state_path.parent / 'base-cgls.bin.state.bin')),
    }
    for key,(p,h) in expected.items():
        if inp.get(key) != p and key != 'state_capture_payload':
            raise ValueError(f'prior 455->419 join path changed for {key}')
    if inp.get('operator_capture_sha256') != expected['operator_capture'][1]:
        raise ValueError('prior operator capture hash differs from retained 419 input')
    if inp.get('state_capture_manifest_sha256') != expected['state_capture_manifest'][1]:
        raise ValueError('prior 455 state manifest hash differs')
    if inp.get('state_capture_payload_sha256') != expected['state_capture_payload'][1]:
        raise ValueError('prior 455 state payload hash differs')
    if sha(prior_path) != '14cabbe692fcb4c83cc6581d965f257fb1698b7f713a13897ba3a85fbc8f9486':
        raise ValueError('the retained 455-to-419 join report is not the sealed expected artifact')
    if expected['operator_capture'][1] != 'f2d42314cfee0428ecd285ebb9dbd5da1556624e3e92af1d12240fd8106b309a':
        raise ValueError('the retained 419 operator capture differs from the sealed input')
    if expected['state_capture_manifest'][1] != '202dd136536cfc9293b882def510c3a4b6e0cbc1136916d5db67acf87b510948':
        raise ValueError('the retained 455 state manifest differs from the sealed input')
    if expected['state_capture_payload'][1] != '1a99fa72a4f45c68c67a53dcace37f0f01df6587d6db1a178606247a9fdef5b6':
        raise ValueError('the retained 455 state payload differs from the sealed input')
    if report.get('mismatches') not in ({}, None):
        raise ValueError('prior 455->419 report records mismatches')
    counts = report.get('exact_word_tallies', {})
    expected_tallies = report.get('expected_counts', {})
    for key, value in expected_tallies.items():
        if counts.get(key) != value:
            raise ValueError(f'prior 455->419 tally {key} does not meet its own exact expected count')
    tet = report.get('tetrahedron_snapshot_relation', {})
    if tet.get('419_after_signedzero_only_canonicalization') != '3840/3840 per arena; no other word mismatch':
        raise ValueError('prior tetrahedron signed-zero-only relation not as expected')
    if report.get('tetrahedron_signedzero_word_counts_by_field') is None:
        raise ValueError('prior report lacks signed-zero accounting')
    return {'path': str(prior_path), 'sha256': sha(prior_path), 'schema': report['schema'],
            'classification': report.get('classification'), 'exact_word_tallies': counts,
            'expected_counts': expected_tallies, 'tetrahedron_relation': tet,
            'inputs': inp, 'mismatches': report.get('mismatches', {})}


def load_states(v1, native_state_path: Path, prior_state_path: Path):
    nd, na, nv = v1.load_state_arrays(native_state_path, STATE_READER)
    pd, pa, pv = v1.load_state_arrays(prior_state_path, STATE_READER)
    for key in ('root','control_step','newton_iteration','environment_count','node_count',
                'tetrahedron_count','material_state_stride','unknown_count'):
        if nd.get(key) != pd.get(key): raise ValueError(f'463 and 455 state {key} differs: {nd.get(key)} vs {pd.get(key)}')
    for key in ('dt_seconds','world_fingerprint_u64_hex'):
        if nd.get(key) != pd.get(key): raise ValueError(f'463 and 455 state {key} differs')
    if nd.get('physical_state_changed') is not False or pd.get('physical_state_changed') is not False:
        raise ValueError('state sidecars must attest no physical-state mutation')
    missing = sorted(set(REQUIRED_STATE_ARRAYS) - set(na) | (set(REQUIRED_STATE_ARRAYS) - set(pa)))
    if missing: raise ValueError(f'required physical state arrays missing: {missing}')
    comparisons = {name: array_cmp(na[name], pa[name]) for name in REQUIRED_STATE_ARRAYS}
    failed = [name for name, r in comparisons.items() if not r.get('equal')]
    if failed: raise ValueError(f'463->455 exact physical-state join failed for {failed}')
    all_shared = sorted(set(na) & set(pa))
    secondary = {name: array_cmp(na[name], pa[name]) for name in all_shared if name not in REQUIRED_STATE_ARRAYS}
    return nd,na,nv,pd,pa,pv,{
        'root': nd['root'], 'newton_iteration': nd['newton_iteration'],
        'world_fingerprint_u64_hex': nd['world_fingerprint_u64_hex'],
        'native_execution_fingerprint_u64_hex': nd.get('device_program_fingerprint_u64_hex'),
        '455_execution_fingerprint_u64_hex': pd.get('device_program_fingerprint_u64_hex'),
        'execution_fingerprint_equal': nd.get('device_program_fingerprint_u64_hex') == pd.get('device_program_fingerprint_u64_hex'),
        'dimensions': {k: nd.get(k) for k in ('environment_count','node_count','tetrahedron_count','material_state_stride','unknown_count')},
        'dt_seconds': nd['dt_seconds'], 'required_exact_arrays': comparisons,
        'required_array_count': len(comparisons), 'required_mismatch_count': 0,
        'other_shared_array_comparisons': secondary,
        'other_shared_array_mismatch_count': sum(not x.get('equal') for x in secondary.values()),
        'operator_lineage_gate': 'passed' if not failed else 'failed',
        'execution_fingerprint_note': 'Different execution fingerprints are recorded, not equated; the join is over physical state arrays and world identity.'}


def parse_env(command: list[str]) -> dict[str,str]:
    out={}
    for i,x in enumerate(command[:-1]):
        if x=='--env':
            k,sep,v=command[i+1].partition('=')
            if sep: out[k]=v
    return out


def run(args):
    v1=load_v1()
    case=Path(args.case); command_path=Path(args.command)
    command=json.loads(command_path.read_text())
    if not isinstance(command,list): raise ValueError('command must be an argv array')
    receipt_path=Path(args.receipt); stderr_path=Path(args.stderr or (case/'stderr.log'))
    cgls_path=Path(args.cgls or (case/'base-cgls.bin'))
    native_state_path=Path(args.native_state or (case/'base-cgls.bin.state.json'))
    prior_state_path=Path(args.prior_state or STATE455)
    prior_report_path=Path(args.prior_join or JOIN_EXPECTED)
    op_path=Path(args.prior_operator or OP419)
    for p in (receipt_path,stderr_path,cgls_path,native_state_path,prior_state_path,prior_report_path,op_path):
        if not p.is_file(): raise ValueError(f'required completed-run/lineage file missing: {p}')
    receipt=load_json(receipt_path)
    if receipt.get('timed_out') is True or receipt.get('execution_started') is not True:
        raise ValueError('run timed out or did not execute')
    if receipt.get('source_inputs_unchanged') is not True or receipt.get('changed_inputs') not in ([],None):
        raise ValueError('native run does not attest unchanged inputs')
    # Bind exact source149 executable and metallib to its build receipt.
    build_path=Path(command[command.index('--build-provenance')+1])
    build=load_json(build_path); artifacts=build.get('artifacts_sha256',{})
    probe=Path(command[command.index('--')+1]); metallib=Path(command[command.index('--metallib')+1])
    if artifacts.get(str(probe))!=sha(probe) or artifacts.get(str(metallib))!=sha(metallib): raise ValueError('probe/metallib do not match source149 build receipt')
    if receipt.get('executable_sha256') and receipt['executable_sha256']!=sha(probe): raise ValueError('receipt executable mismatch')
    if receipt.get('metallib_sha256') and receipt['metallib_sha256']!=sha(metallib): raise ValueError('receipt metallib mismatch')
    env=parse_env(command)
    if env.get('NM_FEM_BASE_CGLS_RIGHT_SCALE')!='block3': raise ValueError('command did not opt into block3 right scaling')
    if env.get('NM_FEM_COUPLED_CAPTURE_SELECTOR')!='1:48': raise ValueError('command does not bind expected coupled capture selector')
    if env.get('NM_FEM_COUPLED_CAPTURE_PATH')!=str(case/'operator-capture'): raise ValueError('command coupled-capture path differs from case')
    if env.get('NM_FEM_BASE_CGLS_CAPTURE_NEWTON')!='48' or env.get('NM_FEM_BASE_CGLS_CAPTURE_PATH')!=str(case/'base-cgls.bin'):
        raise ValueError('command CGLS capture selector/path mismatch')
    stderr=stderr_path.read_text(errors='replace').splitlines()
    cap_mark=[]; dir_mark=[]; skip_mark=[]
    for line in stderr:
        if line.startswith('fem_base_cgls_capture='): cap_mark.append(json.loads(line.split('=',1)[1]))
        elif line.startswith('fem_base_cgls_direction='): dir_mark.append(json.loads(line.split('=',1)[1]))
        elif line.startswith('fem_coupled_capture_skipped='): skip_mark.append(json.loads(line.split('=',1)[1]))
    if len(cap_mark)!=1: raise ValueError('expected exactly one completed CGLS capture marker')
    # The runtime reports a direction summary on every Newton iteration. Bind
    # only the selected root/Newton record, never the last or first record by
    # file order.
    target_dirs=[m for m in dir_mark if m.get('root')==1 and m.get('newton_iteration')==48]
    if len(target_dirs)!=1: raise ValueError(f'expected one direction summary for root1/Newton48, got {len(target_dirs)}')
    cm,dm=cap_mark[0],target_dirs[0]
    if cm.get('complete') is not True or cm.get('bytes')!=cm.get('expected_bytes'): raise ValueError('CGLS payload marker incomplete')
    if dm.get('right_scale')!='block3' or dm.get('original_newton_certificate_applies_to_direction') is not False: raise ValueError('direction marker wrong mode/certificate semantics')
    if not any(m.get('reason')=='environment_count_out_of_scope' and m.get('observed',{}).get('environment_count')==2 for m in skip_mark):
        raise ValueError('expected same-run coupled operator capture to be explicitly skipped for E=2')
    cmeta,c=v1.parse_cgls(cgls_path)
    nd,na,nv,pd,pa,pv,join=load_states(v1,native_state_path,prior_state_path)
    if nd.get('cgls_right_scale')!='block3' or nd.get('cgls_iterations')!=cmeta['iteration_cap']: raise ValueError('native state sidecar mode/cap mismatch')
    if (nd['environment_count'],nd['node_count'],nd['unknown_count'])!=(cmeta['environment_count'],cmeta['nodes_per_environment'],cmeta['unknown_count']): raise ValueError('CGLS and state dimensions differ')
    prior=validate_prior_join(prior_report_path,prior_state_path,op_path)
    # The 455->419 report binds exact world/newton and the exact source capture bytes.
    if int(nd['root'])!=1 or int(nd['root'])!=int(prior_state_doc_root(prior_state_path)) or int(nd['newton_iteration'])!=48:
        raise ValueError('native selected state is not the root1/Newton48 lineage')
    if prior.get('classification')!='exact source-input join only; both source captures are from failed solver runs':
        raise ValueError('prior 455->419 evidence has unexpected classification')
    source143_shared=SOURCE143/'matter/include/numi/matter/shared.h'
    if sha(source143_shared)!=prior.get('inputs',{}).get('shared_header_sha256'):
        raise ValueError('source143 shared ABI header hash changed')
    # Parse 419 only after the full state chain is accepted.
    source_snapshot=Path(args.source_snapshot)
    if source_snapshot.resolve()!=SOURCE149.resolve():
        raise ValueError('unexpected operator source snapshot; this audit is bound to source149')
    mod= v1.load_module('source149_cached_operator_audit', source_snapshot/'matter/tools/cardboard_cached_operator_audit.py')
    header,onodes,otets,integrity=mod.capture_records(op_path)
    if (int(header.get('root',-1)),int(header.get('newton_iteration',-1)))!=(int(nd['root']),int(nd['newton_iteration'])):
        raise ValueError('419 operator root/Newton differs after state join')
    if header.get('world_fingerprint_hex')!=nd.get('world_fingerprint_u64_hex'):
        raise ValueError('419 world identity differs from exact joined state')
    if (int(header.get('environment_count',-1)), int(header.get('node_count_per_environment',-1)),
        int(header.get('tetrahedron_count_per_environment',-1))) != (int(nd['environment_count']),int(nd['node_count']),int(nd['tetrahedron_count'])):
        raise ValueError('419 operator dimensions differ from exact joined state')
    if header.get('dt_seconds') != nd.get('dt_seconds'):
        raise ValueError('419 operator timestep differs from exact joined state')
    mass,stiff,A,_,meta=mod.validate_and_assemble(header,onodes,otets,integrity,args.expected_environment)
    env_id=args.expected_environment; E=cmeta['environment_count']; N=cmeta['nodes_per_environment']
    if env_id<0 or env_id>=E: raise ValueError('expected environment outside capture')
    # Independent 455->419 array comparisons complement the retained exact join report.
    op_nodes=v1.raw_state_from_operator_nodes(onodes,env_id,N)
    state_nodes=np.frombuffer(pa['candidate_nodes'][env_id].tobytes(),dtype='<u4').reshape(N,20)
    node_words=op_nodes.view(np.uint32).reshape(N,20)
    node_mismatch=int(np.count_nonzero(state_nodes!=node_words))
    hist=pa['accepted_material_history'][env_id].astype(np.float32)
    op_hist_rows=[]
    for rec in sorted((r for r in otets if int(r['environment'])==env_id),key=lambda r:int(r['tet_id'])):
        vals=rec['center'].get('accepted_state')
        if vals in (None,[]):
            if int(rec.get('material_state_count',-1))!=0:
                raise ValueError(f"tet {rec['tet_id']} has no captured state but nonzero material state count")
            vals=[0.0]*15
        if len(vals)!=15: raise ValueError(f"tet {rec['tet_id']} has malformed captured accepted state")
        op_hist_rows.append(vals)
    op_hist=np.asarray(op_hist_rows,dtype=np.float32)
    history=v1.floats_equal_report(hist,op_hist)
    tangent=pa['algorithmic_tangent_cache'][env_id].astype(np.float32)
    state_valid=pa['algorithmic_tangent_cache_valid'][env_id].astype(np.uint32)
    op_tan_rows=[]; op_valid_rows=[]
    for rec in sorted((r for r in otets if int(r['environment'])==env_id),key=lambda r:int(r['tet_id'])):
        ccache=rec.get('cached_algorithmic_tangent_local') or {}
        valid=bool(ccache.get('valid',False)); op_valid_rows.append(1 if valid else 0)
        mat=ccache.get('matrix_row_major_local_dP_dF_Pa_per_dimensionless_F')
        if valid and (not isinstance(mat,list) or len(mat)!=81):
            raise ValueError(f"tet {rec['tet_id']} advertises valid cache without 81 matrix entries")
        op_tan_rows.append(mat if valid else [0.0]*81)
    op_valid=np.asarray(op_valid_rows,dtype=np.uint32)
    if not np.array_equal(state_valid,op_valid): raise ValueError('455 cache-valid mask differs from 419 operator capture')
    op_tan=np.asarray(op_tan_rows,dtype=np.float32).reshape(tangent.shape)
    tangent_report=v1.floats_equal_report(tangent,op_tan)
    params=pa['environment_parameters'][env_id].astype(np.float32)
    op_params=np.asarray(header['parameters_by_environment'][env_id],dtype=np.float32)
    param_report=v1.floats_equal_report(params,op_params)
    if node_mismatch or history['numeric_mismatch_count'] or tangent_report['numeric_mismatch_count'] or param_report['numeric_mismatch_count']:
        raise ValueError('independent 455-to-419 operator input arrays fail exact/numeric comparison')
    # The physical node records carry masses and fixed/free tags; the direct
    # candidate-node comparison above therefore binds these operator inputs.
    # Bind the stored Newton correction and RHS as well, rather than relying
    # solely on the earlier report's tally.
    op_correction=np.asarray([r['correction_delta_v_m_s'] for r in sorted((r for r in onodes if int(r['environment'])==env_id),key=lambda r:int(r['node_id']))],dtype=np.float32)
    state_correction=pa['original_newton_correction'][env_id*N:(env_id+1)*N,:3].astype(np.float32)
    correction_report=v1.floats_equal_report(state_correction,op_correction)
    if correction_report['fp32_word_mismatch_count']:
        raise ValueError('455 Newton correction does not match 419 captured correction words')
    # Now evaluate the captured native vectors with the joined 419 operator.
    selected=c['selected'].astype(np.uint32)
    if int(np.count_nonzero(selected))!=1 or np.any(selected>1): raise ValueError('expected exactly one selected environment')
    if int(np.flatnonzero(selected)[0])!=env_id: raise ValueError('selected environment mismatch')
    if int(c['errors'][env_id])!=0 or int(c['control'][env_id,3])!=1: raise ValueError('selected CGLS error/selection state invalid')
    free_nodes=np.asarray(meta['free_nodes'],dtype=np.int64); row={int(n):i for i,n in enumerate(free_nodes)}
    tags={int(r['node_id']):int(float(r['center']['constraint_tag'])) for r in onodes if int(r['environment'])==env_id}
    mech_base=env_id*N
    field_base=E*N+env_id*N
    expected_mask=np.asarray([[1,1,1,0] if tags[i]==0 else [0,0,0,0] for i in range(N)],dtype=np.float32)
    selected_mask=v1.floats_equal_report(c['mask'][mech_base:mech_base+N],expected_mask)
    if selected_mask['numeric_mismatch_count'] or np.any(c['mask'][field_base:field_base+N]!=0):
        raise ValueError('selected mechanical/field masks differ from operator node constraints')
    # Direct selected-environment RHS check against the already-bound 419
    # center residual.  The physical state sidecar additionally proves this
    # RHS came from the same 463/455 Newton candidate.
    op_residual=np.asarray([r['center_residual_impulse_Ns'] for r in sorted((r for r in onodes if int(r['environment'])==env_id),key=lambda r:int(r['node_id']))],dtype=np.float32)
    rhs_report=v1.floats_equal_report(c['b'][mech_base:mech_base+N,:],op_residual)
    if rhs_report['fp32_word_mismatch_count']:
        raise ValueError('CGLS selected RHS does not match 419 physical center-residual words')
    S=np.asarray(c['right_scale_rows'][env_id,:,:,:3],dtype=np.float64)
    inverrs=[]; identerrs=[]
    for node in range(N):
        if tags[node]==0:
            if node not in row: raise ValueError(f'free node {node} absent from operator map')
            sl=slice(3*row[node],3*row[node]+3)
            block=A[sl,:][:,sl].toarray().astype(np.float64)
            inv=np.linalg.inv(block)
            inverrs.append(float(np.linalg.norm(S[node]-inv)/max(np.linalg.norm(inv),1e-300)))
            identerrs.append(float(np.linalg.norm(block@S[node]-np.eye(3))))
            if c['right_scale_rows'][env_id,node,0,3]!=1 or np.any(c['right_scale_rows'][env_id,node,1:,3]!=0): raise ValueError('free-node scale sentinel invalid')
        else:
            if np.any(c['right_scale_rows'][env_id,node]!=0) or node in row: raise ValueError(f'constrained node scale/map mismatch at {node}')
    base=env_id*N
    x=c['x'][base:base+N,:3][free_nodes].astype(np.float64).reshape(-1)
    b=c['b'][base:base+N,:3][free_nodes].astype(np.float64).reshape(-1)
    r=c['r'][base:base+N,:3][free_nodes].astype(np.float64).reshape(-1)
    q=c['q'][base:base+N,:3][free_nodes].astype(np.float64).reshape(-1)
    s=c['s'][base:base+N,:3][free_nodes].astype(np.float64).reshape(-1)
    calcq=np.asarray(A@x).reshape(-1); calcr=b-calcq
    normal=np.asarray(A.T@r).reshape(-1); calcs=np.zeros_like(normal)
    for node in free_nodes:
        sl=slice(3*row[int(node)],3*row[int(node)]+3)
        calcs[sl]=S[int(node)].T@normal[sl]
    def action_error(native: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
      diff=native-reference
      ref_l2=float(np.linalg.norm(reference))
      diff_l2=float(np.linalg.norm(diff))
      return {'native_vs_reference':v1.floats_equal_report(native,reference),
        'native_l2':float(np.linalg.norm(native)), 'reference_l2':ref_l2,
        'difference_l2':diff_l2, 'difference_max_abs':float(np.max(np.abs(diff),initial=0.0)),
        'relative_l2_to_reference':diff_l2/ref_l2 if ref_l2>0 else None,
        'interpretation':'descriptive cross-build/source-capture action comparison; no tolerance or noise-floor gate'}
    action={
      'native_q_vs_419_Ax':action_error(q,calcq),
      'native_r_vs_b_minus_Ax':action_error(r,calcr),
      'native_s_vs_S_transpose_ATr':action_error(s,calcs),
      'native_x_l2':float(np.linalg.norm(x)), 'native_b_l2':float(np.linalg.norm(b)),
      'native_r_l2':float(np.linalg.norm(r)), 'native_s_l2':float(np.linalg.norm(s)),
      'native_q_l2':float(np.linalg.norm(q)), 'A_nnz':int(A.nnz),
      'A_relative_skew_norm':float(meta['relative_skew_norm'])}
    other=1-env_id
    quiet={k:bool(np.all(c[k][other*N:(other+1)*N]==0)) for k in ('x','r','s','p','q','b','mask')}
    quiet['field_rows_zero']=all(bool(np.all(c[k][E*N+other*N:E*N+(other+1)*N]==0)) for k in ('x','r','s','p','q','b','mask'))
    quiet['scale_zero']=bool(np.all(c['right_scale_rows'][other]==0))
    if not all(quiet.values()): raise ValueError(f'unselected environment is not zero: {quiet}')
    return {
      'schema':'numi.cardboard.native_cgls_block3_lineage_audit.v2',
      'verdict':'bounded native CGLS/operator consistency audit through exact physical-state lineage; not solver or physical qualification',
      'run':{'case':str(case),'receipt_sha256':sha(receipt_path),'stderr_sha256':sha(stderr_path),'command_sha256':sha(command_path),'probe_sha256':sha(probe),'metallib_sha256':sha(metallib),'source_build_receipt_sha256':sha(build_path)},
      'native_capture':{'cgls':cmeta,'cgls_sha256':sha(cgls_path),'state_manifest_sha256':sha(native_state_path),'state_payload_sha256':nv.get('payload_sha256'),'capture_marker':cm,'direction_marker':dm,'coupled_capture_skipped_marker':skip_mark},
      'lineage':{'native_463_to_state455':join,'state455_to_operator419':prior,
        'operator_419_path':str(op_path),'operator_419_sha256':sha(op_path),'operator_source_snapshot':str(source_snapshot),
        'native_source_snapshot':str(source_snapshot),'prior_455_source_snapshot':prior.get('inputs',{}).get('source_snapshot'),
        'same_run_operator_capture_used':False,
        'execution_fingerprints':{'native_463':nd.get('device_program_fingerprint_u64_hex'),'455':pd.get('device_program_fingerprint_u64_hex'),'419':header.get('execution_fingerprint_hex')},
        '455_to_419_independent_array_checks':{'candidate_node_word_mismatches':node_mismatch,'accepted_history':history,'cached_tangent':tangent_report,'parameters':param_report,'newton_correction':correction_report,'base_rhs_vs_center_residual':rhs_report}},
      'selected':{'environment':env_id,'selected_flags':selected.tolist(),'control':c['control'].tolist(),
        'control_status_labels':[CONTROL_STATUS.get(int(row[0]),'unknown') for row in c['control']],
        'control_contract':'(status, completed_iterations, iteration_cap, selected)',
        'errors':c['errors'].tolist(),'scalars':c['scalars'].tolist(),'quiet_environment':quiet},
      'right_scale':{'free_node_count':len(free_nodes),'fixed_node_count':N-len(free_nodes),'Aii_inverse_relative_error_max':max(inverrs,default=None),'Aii_inverse_relative_error_median':float(np.median(inverrs)) if inverrs else None,'Aii_times_S_identity_error_max':max(identerrs,default=None),'Aii_times_S_identity_error_median':float(np.median(identerrs)) if identerrs else None,'interpretation':'descriptive same-state local block comparison; no threshold/noise-floor inference'},
      'action_comparisons':action,
      'failed_prefix_history_comparison':run_history_comparison(case),
      'limits':['Same-run coupled operator capture is explicitly skipped at E=2; the 419 operator is reused only after the strict 463->455 state join and retained 455->419 join evidence pass.','Execution fingerprints are recorded and differ across builds; they are not treated as equivalent.','CGLS vectors and action checks do not inherit the original Newton certificate.','The existing 455->419 tetrahedron join includes only explicitly counted signed-zero JSON serialization differences; no other word mismatch was reported.','This audit does not establish native solver convergence, complete coupon success, speedup, material calibration, or physical qualification.']}


def prior_state_doc_root(p:Path)->int:
    return int(load_json(p).get('root',-1))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',required=True); ap.add_argument('--command',required=True); ap.add_argument('--receipt',required=True)
    ap.add_argument('--source-snapshot',required=True)
    ap.add_argument('--stderr'); ap.add_argument('--cgls'); ap.add_argument('--native-state')
    ap.add_argument('--prior-state',default=str(STATE455)); ap.add_argument('--prior-join',default=str(JOIN_EXPECTED)); ap.add_argument('--prior-operator',default=str(OP419))
    ap.add_argument('--expected-environment',type=int,default=1); ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args(); result=run(args); args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'valid':True,'output':str(args.output),'output_sha256':sha(args.output),'state_join':'463->455->419','newton_iteration':result['lineage']['native_463_to_state455']['newton_iteration']}))

if __name__=='__main__': main()
