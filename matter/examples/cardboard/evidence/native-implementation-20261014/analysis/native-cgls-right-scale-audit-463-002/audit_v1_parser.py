#!/usr/bin/env python3
"""Source-bound descriptive audit of an opt-in right-scaled CGLS snapshot.

The audit joins the CGLS v2 vector capture and its state sidecar to the same-run
operator capture. It reports operator-action consistency and the node-block
scale construction; it does not transfer the original Newton certificate to
CGLS, establish convergence, or qualify a physical/material model.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = 0x534C4743
VECTOR_NAMES = ("x", "r", "s", "p", "q", "b", "mask")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def align16(n: int) -> int:
    return (n + 15) & ~15


def parse_cgls(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw = path.read_bytes()
    if len(raw) < 32:
        raise ValueError('CGLS payload shorter than header')
    magic, version, e, n, u, cap, segments, reserved = struct.unpack_from('<8I', raw)
    if magic != MAGIC or version != 2 or segments != 13 or reserved != 0:
        raise ValueError(f'expected right-scale CGLS v2 header, got {(hex(magic),version,e,n,u,cap,segments,reserved)}')
    if e == 0 or n == 0 or u != 2 * e * n or cap not in (32, 64):
        raise ValueError('invalid CGLS dimensions or cap')
    specs: list[tuple[str, str, int, tuple[int, ...]]] = []
    for name in VECTOR_NAMES:
        specs.append((name, '<f4', u * 4, (u, 4)))
    specs.extend([
        ('row_environment', '<u4', u, (u,)),
        ('scalars', '<f4', e * 4, (e, 4)),
        ('control', '<u4', e * 4, (e, 4)),
        ('errors', '<u4', e, (e,)),
        ('selected', '<u4', e, (e,)),
        ('right_scale_rows', '<f4', e * n * 3 * 4, (e, n, 3, 4)),
    ])
    offset = 32
    arrays: dict[str, np.ndarray] = {}
    spans = []
    segment_rows = []
    for name, dtype, count, shape in specs:
        start = align16(offset)
        if start > len(raw) or any(raw[offset:start]):
            raise ValueError(f'bad/truncated alignment padding before {name}')
        end = start + count * 4
        if end > len(raw):
            raise ValueError(f'truncated segment {name}')
        if any(max(start, a) < min(end, b) for a, b in spans):
            raise ValueError(f'overlapping segment {name}')
        spans.append((start, end))
        arrays[name] = np.frombuffer(raw, dtype=np.dtype(dtype), count=count, offset=start).reshape(shape).copy()
        segment_rows.append({'name': name, 'offset': start, 'bytes': count * 4})
        offset = end
    if offset != len(raw):
        raise ValueError(f'unexpected trailing CGLS bytes: {len(raw)-offset}')
    return ({'magic_hex': f'0x{magic:08x}', 'version': version,
             'environment_count': e, 'nodes_per_environment': n,
             'unknown_count': u, 'iteration_cap': cap, 'segment_count': segments,
             'file_bytes': len(raw), 'sha256': sha(path), 'segments': segment_rows}, arrays)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'cannot import source-bound module {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def floats_equal_report(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.shape != bb.shape:
        return {'shape_match': False, 'left_shape': list(aa.shape), 'right_shape': list(bb.shape)}
    finite = np.isfinite(aa) & np.isfinite(bb)
    diff = aa.astype(np.float64) - bb.astype(np.float64)
    word_mismatch = aa.view(np.uint32) != bb.view(np.uint32)
    numeric_mismatch = (aa != bb) | ~finite
    return {'shape_match': True, 'count': int(aa.size), 'finite_pair_count': int(finite.sum()),
            'numeric_mismatch_count': int(numeric_mismatch.sum()),
            'fp32_word_mismatch_count': int(word_mismatch.sum()),
            'signed_zero_only_mismatch_count': int(np.count_nonzero(word_mismatch & (aa == bb) & finite)),
            'max_abs_difference': float(np.max(np.abs(diff[finite]))) if np.any(finite) else None,
            'difference_l2': float(np.linalg.norm(diff[finite])) if np.any(finite) else None}


def load_state_arrays(manifest_path: Path, reader_path: Path) -> tuple[dict, dict[str, np.ndarray], dict]:
    reader = load_module('base_cgls_state_reader_148', reader_path)
    valid = reader.validate_capture(manifest_path)
    doc = json.loads(manifest_path.read_text())
    payload = manifest_path.parent / doc['payload_file'] if isinstance(doc.get('payload_file'), str) else None
    if payload is None:
        # v1 sidecars use payload.file.
        payload = manifest_path.parent / doc['payload']['file']
    raw = payload.read_bytes()
    arrays: dict[str, np.ndarray] = {}
    for item in doc['arrays']:
        dtype = item['dtype']
        if dtype.startswith('record:'):
            dt = np.dtype(('u1', int(item['element_bytes'])))
        elif dtype == 'float32': dt = np.dtype('<f4')
        elif dtype == 'float32x4': dt = np.dtype('<f4')
        elif dtype == 'uint32': dt = np.dtype('<u4')
        else: raise ValueError(f'unsupported sidecar dtype {dtype}')
        count = math.prod(item['shape'])
        scalar_count = count * 4 if dtype == 'float32x4' else count
        a = np.frombuffer(raw, dtype=dt, count=scalar_count, offset=item['offset']).copy()
        if dtype == 'float32x4': a = a.reshape(item['shape'] + [4])
        elif dtype in ('float32','uint32'): a = a.reshape(item['shape'])
        else: a = a.reshape(item['shape'] + [int(item['element_bytes'])])
        arrays[item['name']] = a
    return doc, arrays, valid


def raw_state_from_operator_nodes(nodes: list[dict], env: int, n: int) -> np.ndarray:
    rows = sorted((r for r in nodes if int(r['environment']) == env), key=lambda r: int(r['node_id']))
    if len(rows) != n or [int(r['node_id']) for r in rows] != list(range(n)):
        raise ValueError('operator node rows are not complete dense IDs for selected environment')
    return np.asarray([r['center']['raw_state_float4s'] for r in rows], dtype=np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = Path(args.case)
    command_path = Path(args.command)
    command = json.loads(command_path.read_text())
    receipt_path = case / 'receipt.json'
    stderr_path = case / 'stderr.log'
    cgls_path = case / 'base-cgls.bin'
    state_path = case / 'base-cgls.bin.state.json'
    operator_path = case / 'operator-capture'
    if not all(p.is_file() for p in (receipt_path, stderr_path, cgls_path, state_path, operator_path)):
        raise ValueError('run is missing receipt, stderr, CGLS snapshot/state sidecar, or same-run operator capture')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('timed_out') is True or receipt.get('execution_started') is not True:
        raise ValueError('run timed out or did not start; retain but do not adjudicate as native capture')
    if receipt.get('source_inputs_unchanged') is not True or receipt.get('changed_inputs') not in ([], None):
        raise ValueError('native run does not attest unchanged source inputs')
    build_receipt = Path(command[command.index('--build-provenance') + 1])
    build = json.loads(build_receipt.read_text())
    artifacts = build.get('artifacts_sha256', {})
    probe = Path(command[command.index('--') + 1])
    metallib = Path(command[command.index('--metallib') + 1])
    if artifacts.get(str(probe)) != sha(probe) or artifacts.get(str(metallib)) != sha(metallib):
        raise ValueError('run executable/metallib do not match bound source-build receipt')
    if receipt.get('executable_sha256') and receipt['executable_sha256'] != sha(probe):
        raise ValueError('native receipt executable hash mismatch')
    if receipt.get('metallib_sha256') and receipt['metallib_sha256'] != sha(metallib):
        raise ValueError('native receipt metallib hash mismatch')
    # Require completed native capture markers, not just files left by a partial run.
    marker_prefixes = ('fem_base_cgls_capture=', 'fem_base_cgls_direction=')
    markers = {key: [] for key in marker_prefixes}
    for line in stderr_path.read_text(errors='replace').splitlines():
        for key in marker_prefixes:
            if line.startswith(key):
                markers[key].append(json.loads(line.split('=', 1)[1]))
    if len(markers['fem_base_cgls_capture=']) != 1 or len(markers['fem_base_cgls_direction=']) != 1:
        raise ValueError('expected one completed CGLS payload marker and one direction marker')
    capture_marker, direction_marker = markers['fem_base_cgls_capture='][0], markers['fem_base_cgls_direction='][0]
    if capture_marker.get('complete') is not True or capture_marker.get('bytes') != capture_marker.get('expected_bytes'):
        raise ValueError('native CGLS capture marker is incomplete')
    if direction_marker.get('right_scale') != 'block3' or direction_marker.get('original_newton_certificate_applies_to_direction') is not False:
        raise ValueError('native CGLS direction marker does not attest block3/non-certificate semantics')
    cgls_meta, c = parse_cgls(cgls_path)
    state_doc, st, state_validation = load_state_arrays(
        state_path, Path(args.state_reader))
    if state_doc.get('cgls_right_scale') != 'block3' or state_doc.get('cgls_iterations') not in (32, 64):
        raise ValueError('state sidecar does not attest the source149 block3 CGLS mode')
    if state_doc.get('root') is None or state_doc.get('newton_iteration') is None:
        raise ValueError('state sidecar lacks root/Newton selector')
    envs, nodes, unknowns = cgls_meta['environment_count'], cgls_meta['nodes_per_environment'], cgls_meta['unknown_count']
    if (state_doc.get('environment_count'), state_doc.get('node_count'), state_doc.get('unknown_count')) != (envs, nodes, unknowns):
        raise ValueError('CGLS vector capture and state sidecar dimensions differ')
    if cgls_meta['iteration_cap'] != state_doc.get('cgls_iterations'):
        raise ValueError('CGLS binary cap and sidecar cap differ')
    selected = c['selected'].astype(np.uint32)
    if int(np.count_nonzero(selected)) != 1 or np.any(selected > 1):
        raise ValueError('expected exactly one selected environment for this experiment')
    env = int(np.flatnonzero(selected)[0])
    if env != args.expected_environment:
        raise ValueError(f'selected environment {env} does not match expected {args.expected_environment}')
    for key in ('errors', 'control', 'scalars', 'x', 'r', 's', 'p', 'q', 'b', 'mask', 'row_environment'):
        if not np.isfinite(c[key]).all() if c[key].dtype.kind == 'f' else False:
            raise ValueError(f'CGLS segment {key} contains nonfinite values')
    if int(c['errors'][env]) != 0:
        raise ValueError(f'selected CGLS environment has nonzero error bits {int(c["errors"][env])}')
    if int(c['control'][env, 3]) != 1:
        raise ValueError('selected CGLS control word does not mark selection')
    if c['control'][env, 2] != cgls_meta['iteration_cap']:
        raise ValueError('selected CGLS state cap differs from header')

    # Load source149 operator assembly code and bind the selected operator capture.
    snapshot = Path(args.source_snapshot)
    audit_mod, audit_path = load_module('source149_cached_operator_audit', snapshot / 'matter/tools/cardboard_cached_operator_audit.py')
    header, op_nodes, op_tets, integrity = audit_mod.capture_records(operator_path)
    root, iteration = int(header.get('root', -1)), int(header.get('newton_iteration', -1))
    if (root, iteration) != (int(state_doc['root']), int(state_doc['newton_iteration'])):
        raise ValueError('same-run operator capture root/Newton does not match CGLS state sidecar')
    if header.get('schema') != 'numi.cardboard.fem-newton-capture.v1':
        raise ValueError('unsupported same-run operator capture schema')
    if header.get('world_fingerprint_hex') != state_doc.get('world_fingerprint_u64_hex'):
        raise ValueError('same-run operator/state world fingerprint mismatch')
    if header.get('execution_fingerprint_hex') != state_doc.get('device_program_fingerprint_u64_hex'):
        raise ValueError('same-run operator/state execution fingerprint mismatch')
    mass, stiffness, A, _, meta = audit_mod.validate_and_assemble(header, op_nodes, op_tets, integrity, env)
    free_nodes = np.asarray(meta['free_nodes'], dtype=np.int64)
    free_row = {int(node): i for i, node in enumerate(free_nodes)}
    cgls_raw = np.asarray(c['right_scale_rows'][env, :, :, :], dtype=np.float64)
    block_reports = []
    tag_by_node = {int(r['node_id']): int(float(r['center']['constraint_tag']))
                   for r in op_nodes if int(r['environment']) == env}
    inverse_errors, identity_errors = [], []
    for node in range(nodes):
        rows = cgls_raw[node, :, :3]
        if tag_by_node[node] == 0:
            if node not in free_row:
                raise ValueError(f'free candidate node {node} absent from assembled free map')
            ri = free_row[node]
            dofs = np.arange(3*ri, 3*ri+3)
            block = A[dofs, :][:, dofs].toarray().astype(np.float64)
            inv = np.linalg.inv(block)
            diff = rows - inv
            inverse_errors.append(float(np.linalg.norm(diff) / max(np.linalg.norm(inv), 1e-300)))
            identity_errors.append(float(np.linalg.norm(block @ rows - np.eye(3))))
            if c['right_scale_rows'][env,node,0,3] != 1.0 or np.any(c['right_scale_rows'][env,node,1:,3] != 0.0):
                raise ValueError(f'free node {node} right-scale sentinel lane invalid')
        else:
            if np.any(c['right_scale_rows'][env,node] != 0):
                raise ValueError(f'constrained node {node} has nonzero right-scale rows')
            if node in free_row:
                raise ValueError(f'constrained node {node} unexpectedly appears in free map')
    # Verify environment/field masks and the other environment stays quiet.
    mech = env * nodes
    field = envs * nodes + env * nodes
    selected_mask = c['mask'][mech:mech+nodes]
    expected_mask = np.asarray([[1,1,1,0] if tag_by_node[i] == 0 else [0,0,0,0] for i in range(nodes)], dtype=np.float32)
    mask_report = floats_equal_report(selected_mask, expected_mask)
    if mask_report['numeric_mismatch_count']:
        raise ValueError('selected mechanical mask disagrees with captured fixed/free node tags')
    if np.any(c['mask'][field:field+nodes] != 0):
        raise ValueError('selected field-backing mask is nonzero')
    other = 1 - env if envs == 2 else None
    quiet_report = None
    if other is not None:
        if selected[other] != 0:
            raise ValueError('quiet environment is marked selected')
        quiet_report = {k: bool(np.all(c[k][other*nodes:(other+1)*nodes] == 0)) for k in ('x','r','s','p','q','b','mask')}
        # Vectors are full mechanical+field blocks; also verify field environment explicitly.
        field_base = envs*nodes + other*nodes
        quiet_report['field_rows_zero'] = all(bool(np.all(c[k][field_base:field_base+nodes] == 0)) for k in ('x','r','s','p','q','b','mask'))
        if not all(quiet_report.values()):
            raise ValueError('unselected environment contains nonzero private CGLS vectors')

    # Bind candidate node state and accepted histories against same-run center data.
    candidate_node_bytes = st['candidate_nodes'][env]
    candidate_nodes_words = np.frombuffer(candidate_node_bytes.tobytes(), dtype='<u4').reshape(nodes,20)
    op_nodes_f32 = raw_state_from_operator_nodes(op_nodes, env, nodes)
    node_words = op_nodes_f32.view(np.uint32).reshape(nodes,20)
    node_word_mismatch = int(np.count_nonzero(candidate_nodes_words != node_words))
    node_numeric_equal = bool(np.array_equal(candidate_nodes_words.view('<f4'), op_nodes_f32.reshape(nodes,20)))
    histories = st['accepted_material_history'][env].astype(np.float32)
    op_history = np.asarray([r['center']['accepted_state'] for r in sorted((r for r in op_tets if int(r['environment']) == env), key=lambda r:int(r['tet_id']))],dtype=np.float32)
    history_report=floats_equal_report(histories,op_history)
    tangent = st['algorithmic_tangent_cache'][env].astype(np.float32)
    op_tangent=np.asarray([r['cached_algorithmic_tangent_local'] for r in sorted((r for r in op_tets if int(r['environment'])==env),key=lambda r:int(r['tet_id']))],dtype=np.float32)
    tangent_report=floats_equal_report(tangent,op_tangent)
    params = st['environment_parameters'][env].astype(np.float32)
    op_params=np.asarray(header['parameters_by_environment'][env],dtype=np.float32)
    parameter_report=floats_equal_report(params,op_params)
    candidate_binding={
        'state_sidecar_validation':state_validation,
        'candidate_node_word_mismatch_count_vs_same_run_operator_capture':node_word_mismatch,
        'candidate_node_numeric_equal_vs_same_run_operator_capture':node_numeric_equal,
        'accepted_history_fp32_comparison':history_report,
        'cached_tangent_fp32_comparison':tangent_report,
        'environment_parameters_fp32_comparison':parameter_report,
        'world_and_execution_fingerprints_match':True,
        'same_root_newton_capture':{'root':root,'newton_iteration':iteration}}

    # Native captured action vectors use full node-major ordering; reduce to free DOFs.
    ebase=env*nodes
    physical_x=c['x'][ebase:ebase+nodes,:3][free_nodes].reshape(-1).astype(np.float64)
    physical_b=c['b'][ebase:ebase+nodes,:3][free_nodes].reshape(-1).astype(np.float64)
    physical_r=c['r'][ebase:ebase+nodes,:3][free_nodes].reshape(-1).astype(np.float64)
    native_Ax=c['q'][ebase:ebase+nodes,:3][free_nodes].reshape(-1).astype(np.float64)
    native_s=c['s'][ebase:ebase+nodes,:3][free_nodes].reshape(-1).astype(np.float64)
    computed_Ax=np.asarray(A @ physical_x).reshape(-1)
    computed_r=physical_b-computed_Ax
    computed_s=np.zeros_like(computed_r)
    # S^T A^T r, with S stored as per-node inverse blocks.
    normal=np.asarray(A.T @ physical_r).reshape(-1)
    for node in free_nodes:
        dofs=slice(3*free_row[int(node)],3*free_row[int(node)]+3)
        S=cgls_raw[int(node),:,:3]
        computed_s[dofs]=S.T @ normal[dofs]
    action_reports={
        'q_vs_same_run_Ax':floats_equal_report(native_Ax,computed_Ax),
        'r_vs_b_minus_Ax':floats_equal_report(physical_r,computed_r),
        's_vs_S_transpose_ATr':floats_equal_report(native_s,computed_s),
        'captured_x_l2':float(np.linalg.norm(physical_x)),
        'captured_b_l2':float(np.linalg.norm(physical_b)),
        'captured_r_l2':float(np.linalg.norm(physical_r)),
        'captured_s_l2':float(np.linalg.norm(native_s)),
        'captured_q_l2':float(np.linalg.norm(native_Ax),),
        'A_nnz':int(A.nnz), 'mass_nnz':int(mass.nnz), 'stiffness_nnz':int(stiffness.nnz),
        'A_skew_relative_norm':float(meta['relative_skew_norm'])}
    return {
        'schema':'numi.cardboard.native-cgls-right-scale-audit.v1',
        'verdict':'bounded native capture/operator consistency audit; not a solver or physical qualification',
        'run':{'case':str(case),'receipt':receipt,'receipt_sha256':sha(receipt_path),'stderr_sha256':sha(stderr_path),'command_path':str(command_path),'command_sha256':sha(command_path),'source_build_receipt_sha256':sha(build_receipt),'probe_sha256':sha(probe),'metallib_sha256':sha(metallib)},
        'captures':{'cgls':cgls_meta,'cgls_sha256':sha(cgls_path),'state_manifest_sha256':sha(state_path),'operator_capture_sha256':sha(operator_path),'operator_audit_module':str(audit_path),'operator_audit_module_sha256':sha(audit_path)},
        'selection':{'selected_environment':env,'selected_flags':selected.tolist(),'control_words':c['control'].tolist(),'error_words':c['errors'].tolist(),'scalar_words':c['scalars'].tolist(),'native_capture_marker':capture_marker,'native_direction_marker':direction_marker,'quiet_environment':quiet_report},
        'layout_and_mask':{'mask_vs_captured_constraints':mask_report,'free_node_count':len(free_nodes),'fixed_node_count':nodes-len(free_nodes),'right_scale_inverse_Aii_relative_errors':{'max':max(inverse_errors,default=None),'median':float(np.median(inverse_errors)) if inverse_errors else None},'Aii_times_S_identity_frobenius_errors':{'max':max(identity_errors,default=None),'median':float(np.median(identity_errors)) if identity_errors else None},'classification':'descriptive comparison against same-run captured operator assembly; no tolerance/noise-floor inference'},
        'same_run_operator_binding':candidate_binding,
        'native_vector_action_checks':action_reports,
        'limits':['The CGLS direction is not certified by the original Newton certificate.','The assembled A is reconstructed from the same-run captured production tangent and geometry; comparisons are descriptive and do not establish whole-operator bit identity.','No absolute FP32 noise floor, performance benefit, complete-coupon success, or physical qualification is inferred.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--command',required=True)
    p.add_argument('--source-snapshot',required=True)
    p.add_argument('--state-reader',default='/Users/home/cardboard-implementation-20261014/analysis/base-cgls-state-reader-abi-repair-148-001/tools/read_state_capture.py')
    p.add_argument('--expected-environment',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=run(args)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'valid':True,'output':str(args.output),'output_sha256':sha(args.output),'selected_environment':result['selection']['selected_environment'],'newton_iteration':result['same_run_operator_binding']['same_root_newton_capture']['newton_iteration'],'free_nodes':result['layout_and_mask']['free_node_count']}))

if __name__=='__main__': main()
