#!/usr/bin/env python3
"""Strict offline comparison of full-contact native attempts 378 and 383.

No native process is launched. This auditor keeps source identity, device-program
identity, exact output parity, status selection, and physical verdict separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

BASE = Path('/Users/home/cardboard-implementation-20261014')
BASELINE_RUN = BASE / 'native/ccd-continuation-full-contact-378'
BASELINE_SOURCE = BASE / 'snapshots/source-ee-ccd-continuation-107'
CANDIDATE_SOURCE = BASE / 'snapshots/source-private-ccd-fracture-115'
EXPECTED_SOURCE_MANIFESTS = {
    str(BASELINE_SOURCE): '1def7ff09c9b2968d5a1fa699eaa4131a1bcb3974ded2dfb66d8a060975483ce',
    str(CANDIDATE_SOURCE): '12a5a9629afa6487f9326432fa16284c75cc074f9323cb773a0a5c9a1b0ec867',
}
EXPECTED_SOURCE_DELTA = {
    'matter/include/numi/matter/matter.hpp': ('3afd1dbe987b0cf3a386a69d335489ca38342984d9a5f4c52f2ddcd094ded633', '798e124cdf0b97370a7e1412526693b0f2469985df2b5e985ad3a8acbe00808a'),
    'matter/src/compiler_impl.cpp': ('9dc5cd30bc781704552936531120d28fd6bb95572d2793c7c4a0dd819143272c', '474e717759ded2ecf81a0590dd7f5cb51da8a1041eacddff33c971ccb96152df'),
    'matter/src/metal/contact.metalinc': ('1636c24f766abadd9187809aaaf9511842a56b68f9856cea2154bdc31bc1a603', '28bfbee02d2ea396e171f10e1b383980da245f2c91426089b2e867493a139e23'),
    'matter/src/regional_graph_schwarz.hpp': ('a8a6328d608ba7db4943025f15d35d750968755a8826de32b14b479752c9121d', '927ec6c22f713ece4e68909fa12cf60f8614a22accc1444b770a404c47d0e4cf'),
    'matter/src/runtime.mm': ('2e26d26a04326a3ce567cccb60d213da8f4a8ceaf178c98153939f1b68752fd7', '3c467b04e72512b53a5ba9e1f3ee4132f293b0d2e6ac22acc0c3936512319e74'),
    'matter/src/validation_impl.cpp': ('298c8cd8052309610ef0d85c22eb638aadb8cd5a589978b5599a5043e342c048', '853bafec74a952b3ded64f95558e563b869ba02c6ddcbd8dfb3bb6a25df3b808'),
    'matter/tools/cardboard_bilinear15_projection_policy_check.cpp': ('d7206a61f70b6c5392c97c31201683721aefb9db8c3cdaeb1f07467526d7d6df', '8e76c3435a18b2eeb2680c6b726dd4b6cb9bbd4e64dd48095d401120780b6c2d'),
    'matter/tools/cardboard_contact_ccd_native_check.mm': ('a6e913f735e356165106a937312d6b439ba4f31a7f8778fc2ceec02c2a106b5a', '676ff1f75b2c8634a8bbf5a78046037533c125f21622f4d5172533364a5feb4b'),
    'matter/tools/cardboard_damage_builder.hpp': ('8ff365b0b50bceb4113b58d72f29b7977d70159241c83ed1c4e7fd0a0501124a', '889a8d51fb47066a0a38e57e3cd2ffe50ca8e61100b749f6af336eb0fa7b581b'),
    'matter/tools/cardboard_damage_builder_check.cpp': ('1f25e96a2fc07a306e4c32ed1eea4ee98e5374a6de7753574881abbe611d8c3d', 'e420c7dc197ffecf32794172a68603f170141788a62a2c2fb12549a4d03cbe32'),
    'matter/tools/cardboard_probe.mm': ('af5196370b4c812bc6a71d603987b3ce3ae47d88769bfd54a23851e0c325e059', '6917e515993e1da26d09d946bf8a3f58c7066ad183b19131aa466e94e8d32e99'),
    'matter/tools/regional_graph_completed_tail_check.cpp': ('bfdc93f22dc12827af3620c36d675aef940f7b1a533bace9d4f8d9f44f2022d3', '496ecc347435624c9822905f6ba008241792bba9b381ea8ed967306315b07cca'),
}
MATERIAL_REL = Path('matter/materials/nagasawa2013_medium_orthotropic.nmatter')
MATERIAL_SHA = 'ce39e29918c0111ab1e25e9af8a69eb0e3f7f6f07d1227bfd443da735d9ea348'
CHECKER_REL = Path('matter/tools/cardboard_full_contact_armijo_check.mm')
ARM_PREFIX = 'fem_residual_armijo_iteration='
TRACE_ENABLED_PREFIX = 'fem_residual_armijo_trace_enabled='
CAPTURE_COMPLETE_PREFIX = 'fem_coupled_capture_complete='
CAPTURE_SCHEMA = 'numi.matter.coupled_newton_capture.v1'
CAPTURE_POINT = 'after_fgmres_measure_and_human_direction_map_before_armijo_trial'
EXPECTED_TRACE_SEMANTICS = {
    'base_l2': 'raw recorded free-node metric in impulse units; interpret as the complete assembled mechanical residual only when base_health is healthy_at_armijo_entry',
    'trial_l2': 'selected residual if accepted, otherwise last tested residual',
    'trial_alpha': 'selected alpha if accepted, otherwise last tested alpha',
    'environment_line_search_alpha': 'environment-scoped alpha read by rigid apply; residual Armijo may update the FEM object alpha independently',
    'base_status': 'captured immediately before this Armijo trial loop',
    'trial_status': 'last loop-pass scratch, not necessarily a tested private trial; after selected acceptance it may be an unused gated tail status',
}
EXPECTED_FLAGS = ['--sparse-free-body-parity', '--extended-budget', '--compensated-material-arithmetic']
EXPECTED_LOCAL_MATERIAL_ITERATIONS = 16

class AuditError(ValueError):
    pass

def need(ok, message):
    if not ok:
        raise AuditError(message)

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())

def strict_json_text(text: str, label: str):
    def reject(token):
        raise AuditError(f'{label}: non-standard JSON token {token}')
    return json.loads(text, parse_constant=reject)

def strict_json(path: Path):
    need(path.is_file(), f'missing JSON file: {path}')
    return strict_json_text(path.read_text(encoding='utf-8'), str(path))

def frozen_source(source: Path):
    manifest_path = source / 'frozen-source.json'
    digest = sha_file(manifest_path)
    expected = EXPECTED_SOURCE_MANIFESTS.get(str(source))
    need(expected is not None, f'unrecognized source snapshot path: {source}')
    need(digest == expected, f'{source.name}: frozen source manifest hash differs from reviewed binding')
    obj = strict_json(manifest_path)
    source_map = obj.get('source_sha256')
    need(isinstance(source_map, dict) and len(source_map) > 0, f'{source.name}: malformed frozen source map')
    for rel, expected_hash in source_map.items():
        p = source / rel
        need(p.is_file() and sha_file(p) == expected_hash, f'{source.name}: source file/hash mismatch: {rel}')
    return source_map, digest

def source_delta_report():
    old, old_manifest_hash = frozen_source(BASELINE_SOURCE)
    new, new_manifest_hash = frozen_source(CANDIDATE_SOURCE)
    need(set(old) == set(new), 'source snapshots have different file inventories')
    changed = {p: (old[p], new[p]) for p in old if old[p] != new[p]}
    need(changed == EXPECTED_SOURCE_DELTA, '107-to-115 source delta differs from the reviewed exact list')
    return {'baseline_manifest_sha256': old_manifest_hash,
            'candidate_manifest_sha256': new_manifest_hash,
            'file_count': len(old), 'changed_files': changed}

def locate_build_receipt(run: Path, prediction):
    nominated = Path(prediction.get('build_provenance', ''))
    candidates = [nominated, run / 'inputs/external/source-build.json',
                  run / 'diagnostic-state/inputs/external/source-build.json']
    for path in candidates:
        if path.is_file():
            return path
    raise AuditError(f'{run.name}: source-build receipt is not retained')

def validate_local_material_budget(source: Path, run: Path, prediction):
    checker = source / CHECKER_REL
    need(checker.is_file(), f'{run.name}: native checker source missing')
    text = checker.read_text(encoding='utf-8')
    values = [int(v) for v in re.findall(r'localMaterialNewtonIterations\s*=\s*(\d+)u', text)]
    need(values and set(values) == {EXPECTED_LOCAL_MATERIAL_ITERATIONS},
         f'{run.name}: checker source does not author local material Newton budget 16 consistently: {values}')
    argv = prediction.get('argv', [])
    forbidden = ('--material-iterations', '--newton-iterations', '--accurate-paper', '--accurate-paper-projection')
    need(not any(arg in forbidden for arg in argv), f'{run.name}: CLI overrides the compiled local material/accurate policy')
    stdout = (run / 'stdout.log').read_text(encoding='utf-8')
    settings = re.findall(r'^authored_newton_iterations=(\d+) accurate_paper=(true|false) compensated_material_arithmetic=(true|false)$', stdout, re.M)
    need(settings == [('32', 'false', 'true')], f'{run.name}: fixture-authored policy output differs: {settings}')
    need('selected_fem_preconditioner=scalar_diagonal selector=0' in stdout,
         f'{run.name}: unexpected FEM preconditioner mode')
    return {'checker_source_sha256': sha_file(checker),
            'local_material_newton_iterations': EXPECTED_LOCAL_MATERIAL_ITERATIONS,
            'checker_assignment_occurrences': len(values),
            'fixture_authored_newton_iterations': 32,
            'accurate_paper_override': False,
            'compensated_material_arithmetic': True,
            'override_absent': True}

def normalize_argv(argv, run: Path, source: Path):
    need(isinstance(argv, list) and len(argv) >= 4, 'malformed native checker argv')
    result = list(argv)
    need(Path(result[0]).name == 'numi-matter-cardboard-full-contact-armijo-check', 'unexpected checker executable identity')
    result[0] = '<full-contact-checker>'
    material = Path(result[1])
    expected_material = source / MATERIAL_REL
    need(material.resolve() == expected_material.resolve(), 'argv material is not the source-bound Nagasawa medium asset')
    need(sha_file(material) == MATERIAL_SHA, 'source-bound material asset hash mismatch')
    result[1] = '<source-bound-nagasawa-medium>'
    need('--diagnostic-state-output' in result, 'missing diagnostic-state output option')
    i = result.index('--diagnostic-state-output')
    need(i + 1 < len(result), 'missing diagnostic-state path')
    out = Path(result[i + 1])
    need(out.resolve() == (run / 'diagnostic-state').resolve(), 'diagnostic-state output path does not point into this run')
    result[i + 1] = '<diagnostic-state-dir>'
    need(all(flag in result for flag in EXPECTED_FLAGS), 'missing expected physical checker flags')
    need(not any(arg.startswith('--material-iterations') for arg in result), 'unexpected material iteration override')
    need(not any(arg.startswith('--accurate-paper') for arg in result), 'unexpected accurate-paper override')
    return result

def validate_bundle(run: Path, source: Path, label: str):
    run = run.resolve()
    receipt = strict_json(run / 'receipt.json')
    prediction = strict_json(run / 'prediction.json')
    need(receipt.get('schema') == 'numi.cardboard.native-run-receipt.v2', f'{label}: receipt schema mismatch')
    need(receipt.get('execution_started') is True, f'{label}: run never started')
    need(receipt.get('source_inputs_unchanged') is True and receipt.get('changed_inputs') == [], f'{label}: source inputs changed')
    need(receipt.get('timed_out') is False, f'{label}: run timed out')
    need(receipt.get('termination_errors') == [], f'{label}: run has termination errors')
    need(isinstance(receipt.get('exit_code'), int), f'{label}: missing native exit code')
    need(prediction.get('schema') == 'numi.cardboard.native-prediction.v1', f'{label}: prediction schema mismatch')
    need(Path(prediction.get('source_snapshot', '')).resolve() == source.resolve(), f'{label}: wrong source snapshot')
    source_map, snapshot_hash = frozen_source(source)
    hashes = prediction.get('inputs_sha256', {})
    need(isinstance(hashes, dict), f'{label}: malformed input hashes')
    source_keys = {k for k in hashes if k.startswith('source/')}
    need(source_keys == {'source/' + k for k in source_map}, f'{label}: source input inventory differs from frozen snapshot')
    for rel, digest in source_map.items():
        need(hashes.get('source/' + rel) == digest, f'{label}: prediction source hash mismatch: {rel}')
    build_path = locate_build_receipt(run, prediction)
    build = strict_json(build_path)
    need(sha_file(build_path) == hashes.get('external/source-build.json'), f'{label}: source-build receipt hash mismatch')
    need(build.get('success') is True and Path(build.get('source_snapshot', '')).resolve() == source.resolve(),
         f'{label}: source build is not a successful build of the selected snapshot')
    need(build.get('source_sha256') == source_map, f'{label}: build source map differs from snapshot')
    build_dir = Path(build.get('build_dir', ''))
    artifacts = build.get('artifacts_sha256', {})
    need(isinstance(artifacts, dict) and artifacts, f'{label}: build artifact map missing')
    artifact_by_relative = {}
    for path_text, digest in artifacts.items():
        path = Path(path_text)
        try:
            rel = path.resolve().relative_to(build_dir.resolve()).as_posix()
        except ValueError:
            continue
        artifact_by_relative['runtime/' + rel] = digest
    runtime_keys = {k for k in hashes if k.startswith('runtime/')}
    need(runtime_keys and runtime_keys <= set(artifact_by_relative), f'{label}: a runtime input has no source-build artifact binding')
    for key in runtime_keys:
        need(hashes[key] == artifact_by_relative[key], f'{label}: runtime input differs from source-build artifact: {key}')
        live = build_dir / key.removeprefix('runtime/')
        need(live.is_file() and sha_file(live) == hashes[key], f'{label}: live runtime artifact hash mismatch: {key}')
    need(set(hashes) == source_keys | runtime_keys | {'external/source-build.json'}, f'{label}: unclassified prediction input hashes')
    return {'run': run, 'receipt': receipt, 'prediction': prediction,
            'source_map': source_map, 'snapshot_manifest_sha256': snapshot_hash,
            'build_path': build_path, 'build': build, 'build_sha256': sha_file(build_path),
            'artifact_by_relative': artifact_by_relative}

def validate_run(run: Path, source: Path, label: str):
    common = validate_bundle(run, source, label)
    prediction = common['prediction']
    capture_path = prediction.get('environment', {}).get('NM_FEM_COUPLED_CAPTURE_PATH')
    need(prediction.get('environment', {}).get('NM_FEM_COUPLED_CAPTURE_SELECTOR') == '1201:7', f'{label}: capture selector differs')
    need(Path(capture_path or '').resolve() == (run / 'center-capture').resolve(), f'{label}: capture output prefix differs')
    argv = normalize_argv(prediction.get('argv'), run, source)
    need(prediction.get('timeout_seconds') == 700, f'{label}: timeout protocol differs')
    local_policy = validate_local_material_budget(source, run, prediction)
    return {**common, 'normalized_argv': argv,
            'local_material_policy': local_policy,
            'material_sha256': MATERIAL_SHA}

def parse_trace_file(path: Path):
    text = path.read_text(encoding='utf-8')
    records=[]; enabled=[]; complete=[]; full_contact=[]
    for n, line in enumerate(text.splitlines(), 1):
        if ARM_PREFIX in line:
            raw = line.split(ARM_PREFIX, 1)[1].strip()
            records.append((n, strict_json_text(raw, f'{path}:{n}')))
        if TRACE_ENABLED_PREFIX in line:
            raw = line.split(TRACE_ENABLED_PREFIX, 1)[1].strip()
            enabled.append((n, strict_json_text(raw, f'{path}:{n}')))
        if CAPTURE_COMPLETE_PREFIX in line:
            raw = line.split(CAPTURE_COMPLETE_PREFIX, 1)[1].strip()
            complete.append((n, strict_json_text(raw, f'{path}:{n}')))
        marker = 'fem_full_contact_residual_armijo_enabled='
        if marker in line:
            raw = line.split(marker, 1)[1].strip()
            full_contact.append((n, strict_json_text(raw, f'{path}:{n}')))
    return {'text': text, 'records': records, 'enabled': enabled, 'complete': complete, 'full_contact': full_contact}

def status_rows(parsed):
    rows=[]
    for line, record in parsed['records']:
        need(record.get('schema') == 3 and record.get('root') == 1201, f'Armijo record schema/root invalid at line {line}')
        need(record.get('semantics') == EXPECTED_TRACE_SEMANTICS, f'Armijo semantics mismatch at line {line}')
        envs = record.get('environments')
        need(isinstance(envs, list) and len(envs) == 1, f'expected one Armijo environment at line {line}')
        env = envs[0]
        rows.append({'newton_iteration': record.get('newton_iteration'),
                     'solver_status_code': env.get('solver_status_code'),
                     'solver_status': env.get('solver_status'),
                     'pretrial_status': env.get('pretrial_status'),
                     'trial_status': env.get('trial_status'),
                     'accepted': env.get('accepted'), 'trial_outcome': env.get('trial_outcome'),
                     'base_health': env.get('base_health'), 'selected_alpha': env.get('selected_alpha'),
                     'line_search_raw': env.get('line_search_raw')})
    return rows

def compare_trace(base_run: Path, candidate_run: Path):
    base = parse_trace_file(base_run / 'stderr.log')
    cand = parse_trace_file(candidate_run / 'stderr.log')
    bstatus = status_rows(base); cstatus = status_rows(cand)
    need(len(base['records']) == 32 and len(cand['records']) == 32, 'expected 32 authored Newton trace rows in both runs')
    need([x[1].get('newton_iteration') for x in base['records']] == list(range(32)), 'baseline trace iterations are not 0..31')
    need([x[1].get('newton_iteration') for x in cand['records']] == list(range(32)), 'candidate trace iterations are not 0..31')
    trace_exact = [r for (_,r) in base['records']] == [r for (_,r) in cand['records']]
    raw_trace_exact = [base['text'].splitlines()[line-1].split(ARM_PREFIX,1)[1]
                       for line,_ in base['records']] == [cand['text'].splitlines()[line-1].split(ARM_PREFIX,1)[1]
                       for line,_ in cand['records']]
    status_exact = bstatus == cstatus
    differences=[]; status_tuple_differences=[]
    for i, (b, c) in enumerate(zip(bstatus, cstatus)):
        if b != c:
            differences.append({'newton_iteration': i, 'baseline': b, 'candidate': c})
        keys=('solver_status_code','solver_status','pretrial_status','trial_status')
        if any(b[k] != c[k] for k in keys):
            status_tuple_differences.append({'newton_iteration':i,
                'baseline':{k:b[k] for k in keys},'candidate':{k:c[k] for k in keys}})
    first_fatal = lambda rows: next((r['newton_iteration'] for r in rows if r['solver_status_code'] not in (None, 0)), None)
    b_fatal_i=first_fatal(bstatus); c_fatal_i=first_fatal(cstatus)
    def fatal_tuple(rows, at):
        if at is None: return None
        row=rows[at]
        return {'solver_status_code':row['solver_status_code'],
                'environment':(row['solver_status'] or {}).get('environment'),
                'object_index':(row['solver_status'] or {}).get('object_index'),
                'failing_index':(row['solver_status'] or {}).get('failing_index'),
                'diagnostics':(row['solver_status'] or {}).get('diagnostics')}
    bcomplete=[v for _,v in base['complete']]; ccomplete=[v for _,v in cand['complete']]
    for run, entries, label in ((base_run,bcomplete,'baseline'),(candidate_run,ccomplete,'candidate')):
        for item in entries:
            need(Path(item.get('manifest','')).resolve() == (run/'center-capture.coupled.json').resolve(),
                 f'{label}: capture completion points at an unexpected manifest')
    bcomplete_norm=json.loads(json.dumps(bcomplete)); ccomplete_norm=json.loads(json.dumps(ccomplete))
    for item in bcomplete_norm+ccomplete_norm:
        item['manifest']='<run-specific-center-manifest>'
    bt=base['text'].replace(str(base_run/'center-capture.coupled.json'),'<run-specific-center-manifest>')
    ct=cand['text'].replace(str(candidate_run/'center-capture.coupled.json'),'<run-specific-center-manifest>')
    return {'trace_record_count': 32, 'all_armijo_records_exact': trace_exact,
            'armijo_record_json_bytes_exact': raw_trace_exact,
            'deterministic_status_rows_exact': status_exact, 'status_row_differences': differences,
            'baseline_first_nonzero_solver_status_iteration': b_fatal_i,
            'candidate_first_nonzero_solver_status_iteration': c_fatal_i,
            'baseline_first_failure_tuple':fatal_tuple(bstatus,b_fatal_i),
            'candidate_first_failure_tuple':fatal_tuple(cstatus,c_fatal_i),
            'failure_tuple_differences':status_tuple_differences,
            'baseline_trace_enabled': [v for _,v in base['enabled']],
            'candidate_trace_enabled': [v for _,v in cand['enabled']],
            'trace_enabled_exact': [v for _,v in base['enabled']] == [v for _,v in cand['enabled']],
            'baseline_full_contact_arms': [v for _,v in base['full_contact']],
            'candidate_full_contact_arms': [v for _,v in cand['full_contact']],
            'full_contact_armijo_config_exact': [v for _,v in base['full_contact']] == [v for _,v in cand['full_contact']],
            'baseline_capture_complete': bcomplete,
            'candidate_capture_complete': ccomplete,
            'capture_complete_exact_except_run_manifest_path': bcomplete_norm == ccomplete_norm,
            'capture_complete_path_delta': {'baseline': [x.get('manifest') for x in bcomplete],
                                            'candidate': [x.get('manifest') for x in ccomplete]},
            'baseline_stderr_sha256': sha_file(base_run/'stderr.log'),
            'candidate_stderr_sha256': sha_file(candidate_run/'stderr.log'),
            'stderr_bytes_exact': base['text'] == cand['text'],
            'stderr_exact_except_run_manifest_path': bt == ct,
            'status_tuple_adjudication': classify_tuple_change(status_tuple_differences, 7, b_fatal_i, c_fatal_i),
            'baseline_stdout_sha256':sha_file(base_run/'stdout.log'),
            'candidate_stdout_sha256':sha_file(candidate_run/'stdout.log'),
            'stdout_bytes_exact':(base_run/'stdout.log').read_bytes()==(candidate_run/'stdout.log').read_bytes(),
            'baseline_failure_summary':(base_run/'stderr.log').read_text(encoding='utf-8').splitlines()[0],
            'candidate_failure_summary':(candidate_run/'stderr.log').read_text(encoding='utf-8').splitlines()[0]}

def classify_tuple_change(differences, capture_iteration, base_fatal, candidate_fatal):
    if not differences:
        return {'different_tuple': False, 'candidate_order_fixed_at_failure': None, 'classification': 'no_status_row_difference'}
    same_fatal = base_fatal == candidate_fatal
    if same_fatal and base_fatal == capture_iteration:
        return {'different_tuple': True, 'candidate_order_fixed_at_failure': 'must_be_proved_from_paired_capture_arrays',
                'classification': 'failure_iteration_matches_selected_capture_but_order_evidence_pending'}
    return {'different_tuple': True, 'candidate_order_fixed_at_failure': False,
            'classification': 'unadjudicated_tuple_delta_capture_is_not_at_failure_iteration',
            'baseline_fatal_iteration': base_fatal, 'candidate_fatal_iteration': candidate_fatal,
            'selected_center_capture_iteration': capture_iteration}

def validate_capture(run: Path, label: str):
    manifest_path = run / 'center-capture.coupled.json'
    payload_path = run / 'center-capture.coupled.bin'
    manifest = strict_json(manifest_path)
    need(manifest.get('schema') == CAPTURE_SCHEMA, f'{label}: coupled capture schema mismatch')
    for key, expected in {'byte_order':'little_endian','capture_point':CAPTURE_POINT,'capture_kind':'raw_transaction_candidate_not_checkpoint',
                          'root':1201,'iteration':7,'environment_count':1}.items():
        need(manifest.get(key) == expected, f'{label}: capture {key} mismatch')
    need(Path(manifest.get('payload', {}).get('file', '')).name == payload_path.name, f'{label}: payload filename mismatch')
    need(payload_path.is_file(), f'{label}: missing adjacent capture payload')
    raw = payload_path.read_bytes()
    need(len(raw) == manifest['payload']['byte_count'], f'{label}: payload byte count mismatch')
    need(sha_bytes(raw) == manifest['payload']['sha256'], f'{label}: payload SHA mismatch')
    arrays=manifest.get('arrays')
    need(isinstance(arrays, list) and manifest.get('array_count') == 145 and len(arrays) == 145,
         f'{label}: expected all 145 capture arrays')
    names=[a.get('name') for a in arrays]
    need(len(set(names)) == len(names), f'{label}: duplicate array names')
    ranges=[]
    for a in arrays:
        start=a.get('offset'); size=a.get('byte_count'); elem=a.get('element_bytes'); count=a.get('count')
        need(all(isinstance(x,int) and x >= 0 for x in (start,size,elem,count)), f'{label}: malformed array extent {a.get("name")}')
        need(count * elem == size, f'{label}: count/element size mismatch for {a.get("name")}')
        need(start + size <= len(raw), f'{label}: array outside payload: {a.get("name")}')
        need(sha_bytes(raw[start:start+size]) == a.get('sha256'), f'{label}: array digest mismatch: {a.get("name")}')
        ranges.append((start,start+size,a['name']))
    ordered=sorted(ranges)
    need(all(ordered[i][1] <= ordered[i+1][0] for i in range(len(ordered)-1)), f'{label}: overlapping array ranges')
    return {'path':str(manifest_path),'manifest':manifest,'payload_path':payload_path,'payload':raw,
            'array_names':names,'arrays':arrays,'manifest_sha256':sha_file(manifest_path),
            'payload_sha256':sha_bytes(raw)}

def compare_capture(base_run: Path, cand_run: Path):
    b=validate_capture(base_run,'378'); c=validate_capture(cand_run,'383')
    bnames=b['array_names']; cnames=c['array_names']
    layout_exact = bnames == cnames and b['arrays'] == c['arrays']
    payload_exact = b['payload'] == c['payload']
    bm=b['manifest']; cm=c['manifest']
    # Do not normalize either effective fingerprint. Only the run-specific payload path is ignored.
    bm_norm=json.loads(json.dumps(bm)); cm_norm=json.loads(json.dumps(cm))
    bm_norm['payload']['file']=cm_norm['payload']['file']='<adjacent-payload>'
    metadata_diffs=[]
    all_keys=set(bm_norm)|set(cm_norm)
    for key in sorted(all_keys):
        if bm_norm.get(key) != cm_norm.get(key):
            metadata_diffs.append({'field':key,'baseline':bm_norm.get(key),'candidate':cm_norm.get(key)})
    arrays_by={name:(a['sha256'],a['offset'],a['byte_count'],a['dtype']) for name,a in zip(bnames,b['arrays'])}
    required_order_fields=('deformable_contact_candidates','deformable_contact_active_indices',
                           'deformable_contact_active_counts','deformable_contact_active_offsets',
                           'deformable_contact_global_active_indices')
    exact_order_fields=[]
    for name in required_order_fields:
        if name in bnames and name in cnames:
            ba=b['arrays'][bnames.index(name)]; ca=c['arrays'][cnames.index(name)]
            if ba == ca:
                exact_order_fields.append(name)
    order_equal = len(exact_order_fields) == len(required_order_fields)
    capture_order_proof = {'same_selected_capture_generation': bm.get('root') == cm.get('root') and bm.get('iteration') == cm.get('iteration') == 7,
                           'exact_order_array_names': exact_order_fields,
                           'required_order_arrays': list(required_order_fields),
                           'selected_center_candidate_order_exact': order_equal,
                           'scope_warning':'This proves candidate order only for the selected Newton-7 center capture; it does not bind a later Newton failure unless that failure occurs at iteration 7.'}
    return {'array_count_baseline':len(bnames),'array_count_candidate':len(cnames),
            'array_names_and_descriptors_exact':layout_exact,'payload_byte_count':len(b['payload']),
            'payload_sha256_baseline':b['payload_sha256'],'payload_sha256_candidate':c['payload_sha256'],
            'all_145_array_bytes_exact':payload_exact and layout_exact,
            'baseline_source_physics_fingerprint':bm.get('source_physics_fingerprint'),
            'candidate_source_physics_fingerprint':cm.get('source_physics_fingerprint'),
            'source_physics_fingerprint_equal':bm.get('source_physics_fingerprint') == cm.get('source_physics_fingerprint'),
            'baseline_device_program_fingerprint':bm.get('device_program_fingerprint'),
            'candidate_device_program_fingerprint':cm.get('device_program_fingerprint'),
            'device_program_fingerprint_equal':bm.get('device_program_fingerprint') == cm.get('device_program_fingerprint'),
            'manifest_metadata_equal_except_run_payload_path':not metadata_diffs,
            'manifest_metadata_differences':metadata_diffs,
            'candidate_order_evidence':capture_order_proof,
            'baseline_manifest_sha256':b['manifest_sha256'],'candidate_manifest_sha256':c['manifest_sha256']}

def compare_snapshot_binary(a: bytes, b: bytes, label: str):
    need(len(a) == len(b) and len(a) >= 20, f'{label}: binary snapshot size mismatch/malformed')
    la=int.from_bytes(a[:4],'little'); lb=int.from_bytes(b[:4],'little')
    need(la == lb and 4 + la + 16 <= len(a), f'{label}: snapshot fingerprint prefix malformed')
    source_start=4+la; device_start=source_start+8; end=device_start+8
    prefix_equal=a[:source_start] == b[:source_start]
    source_equal=a[source_start:device_start] == b[source_start:device_start]
    device_a=int.from_bytes(a[device_start:end],'little'); device_b=int.from_bytes(b[device_start:end],'little')
    suffix_equal=a[end:] == b[end:]
    return {'raw_bytes_exact':a == b,'message_prefix_exact':prefix_equal,
            'source_physics_fingerprint_equal':source_equal,
            'baseline_device_program_fingerprint':f'{device_a:016x}',
            'candidate_device_program_fingerprint':f'{device_b:016x}',
            'device_program_fingerprint_equal':device_a == device_b,
            'all_state_bytes_exact_after_fingerprint_fields':prefix_equal and suffix_equal,
            'all_bytes_exact_except_device_fingerprint':prefix_equal and source_equal and suffix_equal}

def compare_diag_state(base_run: Path, cand_run: Path):
    bdir=base_run/'diagnostic-state/dense'; cdir=cand_run/'diagnostic-state/dense'
    need(bdir.is_dir() and cdir.is_dir(), 'missing dense diagnostic-state directory')
    bf={p.name:p for p in bdir.iterdir() if p.is_file()}; cf={p.name:p for p in cdir.iterdir() if p.is_file()}
    inventory_equal=set(bf)==set(cf)
    report=[]; transactions=[]
    for name in sorted(set(bf)|set(cf)):
        if name not in bf or name not in cf:
            report.append({'file':name,'present_baseline':name in bf,'present_candidate':name in cf,'exact':False}); continue
        old=bf[name].read_bytes(); new=cf[name].read_bytes()
        if name.endswith('.bin'):
            details=compare_snapshot_binary(old,new,name)
            report.append({'file':name,**details})
        elif name.endswith('.json') and name.startswith('transaction-'):
            jo=strict_json(bf[name]); jn=strict_json(cf[name])
            transactions.append({'file':name,'attempt_index':jo.get('attempt_index'),
                'requested_control_step':jo.get('requested_control_step'),
                'baseline_completion_stage':jo.get('completion_stage'),
                'candidate_completion_stage':jn.get('completion_stage'),
                'baseline_accepted':jo.get('accepted'),'candidate_accepted':jn.get('accepted'),
                'metadata_exact':jo==jn,'before_state_hashes_equal':jo.get('before')==jn.get('before'),
                'after_state_hashes_equal':jo.get('after')==jn.get('after')})
            report.append({'file':name,'json_exact':jo==jn,'baseline_sha256':sha_bytes(old),'candidate_sha256':sha_bytes(new),
                           'attempt_index':jo.get('attempt_index'),'requested_control_step':jo.get('requested_control_step'),
                           'completion_stage_baseline':jo.get('completion_stage'),'completion_stage_candidate':jn.get('completion_stage'),
                           'accepted_baseline':jo.get('accepted'),'accepted_candidate':jn.get('accepted'),
                           'before_state_hashes_exact':jo.get('before') == jn.get('before'),
                           'after_state_hashes_exact':jo.get('after') == jn.get('after')})
        else:
            normalized_exact=False
            if name.endswith('.log'):
                old_norm=old.decode('utf-8').replace(str(base_run/'center-capture.coupled.json'),'<run-specific-center-manifest>')
                new_norm=new.decode('utf-8').replace(str(cand_run/'center-capture.coupled.json'),'<run-specific-center-manifest>')
                normalized_exact=old_norm==new_norm
            report.append({'file':name,'raw_bytes_exact':old==new,
                           'exact_except_run_manifest_path':normalized_exact,
                           'baseline_sha256':sha_bytes(old),'candidate_sha256':sha_bytes(new)})
    return {'file_inventory_exact':inventory_equal,'file_count_baseline':len(bf),'file_count_candidate':len(cf),
            'transactions':transactions,
            'files':report,
            'all_transaction_json_exact':all(x.get('json_exact',True) for x in report if x['file'].startswith('transaction-') and x['file'].endswith('.json')),
            'all_binary_payloads_exact_except_device_fingerprint':all(x.get('all_bytes_exact_except_device_fingerprint',False) for x in report if x['file'].endswith('.bin')),
            'all_log_bytes_exact_except_run_manifest_path':all(x.get('raw_bytes_exact') is True or x.get('exact_except_run_manifest_path') is True for x in report if x['file'].endswith('.log')),
            'raw_file_set_exact':inventory_equal and all(x.get('raw_bytes_exact',x.get('json_exact',False)) for x in report)}

def compare_protocol(b, c):
    bp=b['prediction']; cp=c['prediction']
    argv_equal=b['normalized_argv']==c['normalized_argv']
    be=dict(bp.get('environment',{})); ce=dict(cp.get('environment',{}))
    be['NM_FEM_COUPLED_CAPTURE_PATH']=ce['NM_FEM_COUPLED_CAPTURE_PATH']='<capture-output-prefix>'
    env_equal=be==ce
    narrative={k:{'baseline':bp.get(k),'candidate':cp.get(k),'equal':bp.get(k)==cp.get(k)} for k in ('question','prediction')}
    bh=bp.get('inputs_sha256',{}); ch=cp.get('inputs_sha256',{})
    need(set(bh)==set(ch),'prediction input hash path inventories differ')
    changed={k for k in bh if bh[k]!=ch[k]}
    expected_source_keys={'source/'+k for k in EXPECTED_SOURCE_DELTA}
    source_changed={k for k in changed if k.startswith('source/')}
    need(source_changed==expected_source_keys,'prediction source input deltas do not match expected 107-to-115 source map')
    runtime_changed={k for k in changed if k.startswith('runtime/')}
    need(changed <= source_changed | runtime_changed | {'external/source-build.json'}, 'unclassified prediction input hash changed')
    need('external/source-build.json' in changed, 'source build receipt was not rebound to candidate build')
    return {'normalized_physical_argv_equal':argv_equal,'baseline_normalized_argv':b['normalized_argv'],
            'candidate_normalized_argv':c['normalized_argv'],'environment_equal_except_capture_path':env_equal,
            'narrative_metadata':narrative,'effective_input_hash_inventory_equal':set(bh)==set(ch),
            'changed_input_hash_paths':sorted(changed),'reviewed_source_delta_paths':sorted(source_changed),
            'changed_runtime_artifact_paths':sorted(runtime_changed),
            'baseline_material_sha256':b['material_sha256'],'candidate_material_sha256':c['material_sha256'],
            'material_asset_equal':b['material_sha256']==c['material_sha256'],
            'baseline_effective_runtime_artifacts':{k:v for k,v in b['artifact_by_relative'].items() if k in b['prediction']['inputs_sha256']},
            'candidate_effective_runtime_artifacts':{k:v for k,v in c['artifact_by_relative'].items() if k in c['prediction']['inputs_sha256']}}

def validate_381_control(control_run: Path, source: Path):
    v=validate_bundle(control_run,source,'381')
    receipt=v['receipt']; pred=v['prediction']
    need(receipt['exit_code']==0,'381 routing-control native checker did not exit successfully')
    need(len(pred.get('argv',[]))>=1 and Path(pred['argv'][0]).name=='numi-matter-cardboard-contact-ccd-native-check',
         '381 is not the contact CCD native checker')
    lines=[x for x in (control_run/'stdout.log').read_text(encoding='utf-8').splitlines() if x.strip()]
    need(len(lines)==1,'381 stdout must contain one machine-readable result')
    result=strict_json_text(lines[0], '381 stdout')
    need(result.get('schema')=='numi.matter.contact-ccd-native-regression.v3' and result.get('status')=='pass' and result.get('checks')==169,
         '381 routing-control result schema/status/check count mismatch')
    need(result.get('physical_steps')==0, '381 control unexpectedly reports physical steps')
    need(result.get('geometric_failure_producer_control',{}).get('physical_qualification') is False and
         result.get('captured_ee_physical_qualification') is False,
         '381 control overstates physical qualification')
    return {'exit_code':receipt['exit_code'],'checks':result['checks'],'status':result['status'],
            'physical_steps':result['physical_steps'],'physical_qualification':False,
            'captured_ee_diagnostic_only':result.get('captured_ee_diagnostic_only'),
            'geometric_failure_producer_control':result.get('geometric_failure_producer_control'),
            'receipt_sha256':sha_file(control_run/'receipt.json'),'stdout_sha256':sha_file(control_run/'stdout.log'),
            'interpretation':'Native routing/control evidence only; zero FEM steps and no physical-contact or full-box qualification.'}

def audit(base_run: Path, candidate_run: Path, control_381: Path):
    source_binding=source_delta_report()
    b=validate_run(base_run,BASELINE_SOURCE,'378')
    c=validate_run(candidate_run,CANDIDATE_SOURCE,'383')
    protocol=compare_protocol(b,c)
    trace=compare_trace(base_run,candidate_run)
    cap=compare_capture(base_run,candidate_run)
    diagnostic=compare_diag_state(base_run,candidate_run)
    control=validate_381_control(control_381,CANDIDATE_SOURCE)
    run_facts={}
    for label,v in (('378',b),('383',c)):
        run_facts[label]={'exit_code':v['receipt']['exit_code'],'timed_out':v['receipt']['timed_out'],
                          'execution_started':v['receipt']['execution_started'],'source_inputs_unchanged':v['receipt']['source_inputs_unchanged'],
                          'wall_seconds':v['receipt']['wall_seconds'],'receipt_sha256':sha_file(v['run']/'receipt.json'),
                          'prediction_sha256':sha_file(v['run']/'prediction.json'),'source_build_sha256':v['build_sha256'],
                          'source_snapshot':str(v['run'] and (BASELINE_SOURCE if label=='378' else CANDIDATE_SOURCE)),
                          'checker_material_policy':v['local_material_policy']}
    source_fp_equal=cap['source_physics_fingerprint_equal']
    output_components=[protocol['normalized_physical_argv_equal'],protocol['environment_equal_except_capture_path'],
                       protocol['material_asset_equal'],trace['all_armijo_records_exact'],trace['deterministic_status_rows_exact'],
                       trace['armijo_record_json_bytes_exact'],trace['capture_complete_exact_except_run_manifest_path'],trace['stderr_exact_except_run_manifest_path'],
                       cap['array_names_and_descriptors_exact'],cap['all_145_array_bytes_exact'],
                       diagnostic['file_inventory_exact'],diagnostic['all_transaction_json_exact'],
                       diagnostic['all_binary_payloads_exact_except_device_fingerprint'],diagnostic['all_log_bytes_exact_except_run_manifest_path']]
    exact_output_components=all(output_components)
    physical_attempts={label:{'native_process_completed':True,'exit_code':v['receipt']['exit_code'],
                              'attempt_passed':v['receipt']['exit_code']==0,
                              'full_contact_qualified':False} for label,v in (('378',b),('383',c))}
    return {'schema':'numi.cardboard.full_contact_378_383_parity.v1',
            'overall_status':'COMPARISON_COMPLETE_OUTPUT_PARITY' if exact_output_components else 'COMPARISON_COMPLETE_WITH_DIFFERENCES',
            'native_runs':run_facts,'source_binding':source_binding,'protocol_comparison':protocol,
            'capture_comparison':cap,'diagnostic_transaction_comparison':diagnostic,
            'armijo_and_status_comparison':trace,'producer_reducer_381_control':control,
            'physical_attempt_verdicts':physical_attempts,
            'effective_fingerprint_comparison':{
              'source_physics_fingerprint_equal':source_fp_equal,
              'baseline':cap['baseline_source_physics_fingerprint'],'candidate':cap['candidate_source_physics_fingerprint'],
              'device_program_fingerprint_equal':cap['device_program_fingerprint_equal'],
              'baseline_device_program':cap['baseline_device_program_fingerprint'],
              'candidate_device_program':cap['candidate_device_program_fingerprint'],
              'note':'Both fingerprints are reported, never normalized away. Source and device changes are separately bound to the reviewed source snapshots/build receipts.'},
            'limits':['The runs are failed full-contact attempts if their native exit codes are nonzero; parity never promotes them to convergence or physical qualification.',
                      'The local material Newton budget is 16 in the checker source for both runs. The output authored_newton_iterations=32 is the separate world Newton budget; no material-iteration CLI override exists or is used.',
                      'The 381 checker passes 169 native routing checks but performs zero FEM steps; it is separate control evidence, not full-contact qualification.',
                      'The selected center capture is Newton 7. Its candidate-order arrays do not prove ordering at a later failure iteration; a changed failure tuple outside the capture generation remains unadjudicated.',
                      'No tolerance, residual, acceptance, or status differences are normalized by this auditor.'],
            'all_compared_output_components_exact':exact_output_components}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-run',type=Path,default=BASELINE_RUN)
    ap.add_argument('--candidate-run',type=Path,required=True)
    ap.add_argument('--control-381',type=Path,default=BASE/'native/deterministic-private-ccd-producer-reducer-381')
    ap.add_argument('--output',type=Path,required=True)
    ns=ap.parse_args()
    try:
        report=audit(ns.baseline_run.resolve(),ns.candidate_run.resolve(),ns.control_381.resolve())
        ns.output.parent.mkdir(parents=True,exist_ok=True)
        ns.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print(json.dumps({'status':report['overall_status'],'all_components_exact':report['all_compared_output_components_exact'],
                          'armijo_exact':report['armijo_and_status_comparison']['all_armijo_records_exact'],
                          'capture_exact':report['capture_comparison']['all_145_array_bytes_exact'],
                          'control381_checks':report['producer_reducer_381_control']['checks'],'output':str(ns.output)},sort_keys=True))
        return 0 if report['all_compared_output_components_exact'] else 1
    except Exception as exc:
        ns.output.parent.mkdir(parents=True,exist_ok=True)
        fail={'schema':'numi.matter.full_contact_378_383_parity.v1','overall_status':'FAIL_CLOSED',
              'baseline_run':str(ns.baseline_run.resolve()),'candidate_run':str(ns.candidate_run.resolve()),
              'error':str(exc),'physical_qualification':False}
        ns.output.write_text(json.dumps(fail,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print(json.dumps({'status':'FAIL_CLOSED','error':str(exc),'output':str(ns.output)},sort_keys=True),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
