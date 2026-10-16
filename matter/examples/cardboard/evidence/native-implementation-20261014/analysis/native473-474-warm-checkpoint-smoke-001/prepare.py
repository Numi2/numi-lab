#!/usr/bin/env python3
"""Source173 warm-checkpoint smoke preflight; never launches a native process."""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

PKG = Path(__file__).resolve().parent
SPEC_PATH = PKG / 'run-spec.json'
MANIFEST_PATH = PKG / 'package-manifest.json'
B = Path('/Users/home/cardboard-implementation-20261014')
SOLVER = B / 'solver'
GIB = 1024 ** 3
MIB = 1024 ** 2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def regular_file(path, cap):
    path = Path(path)
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError(f'parent path contains a symlink: {parent}')
    st = path.lstat()
    if path.is_symlink() or not path.is_file() or st.st_nlink != 1:
        raise ValueError(f'expected a fresh unlinked regular file: {path}')
    if st.st_size <= 0 or st.st_size > cap:
        raise ValueError(f'file size outside bounded interval for {path}: {st.st_size}')
    if path.resolve(strict=True) != path.absolute():
        raise ValueError(f'file path is aliased or noncanonical: {path}')
    return st


def parse_kv(path):
    values = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        key, sep, value = line.partition('=')
        if not sep or not key or key in values:
            raise ValueError(f'malformed or duplicate property in {path}: {key!r}')
        values[key] = value
    return values


def parse_prefix(path, expected_control, expected_output):
    fields = {}
    entries = []
    seen = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('file='):
            parts = line[5:].split('|')
            if len(parts) != 4:
                raise ValueError('malformed prefix file record')
            rel, raw_bytes, digest, observation = parts
            relpath = Path(rel)
            if (not rel or relpath.is_absolute() or '..' in relpath.parts or
                    any(c in rel for c in ('\\', '|', '\n', '\r')) or
                    rel in seen or not re.fullmatch(r'[0-9a-f]{64}', digest) or
                    observation not in ('0', '1')):
                raise ValueError(f'unsafe/duplicate prefix member: {rel!r}')
            try:
                size = int(raw_bytes, 10)
            except ValueError as exc:
                raise ValueError(f'invalid prefix byte count: {raw_bytes!r}') from exc
            if size <= 0 or size > 2 * GIB:
                raise ValueError(f'prefix member outside bounded bytes: {rel}')
            seen.add(rel)
            entries.append((rel, size, digest, observation == '1'))
        else:
            key, sep, value = line.partition('=')
            if not sep or not key or key in fields:
                raise ValueError(f'malformed/duplicate prefix header {key!r}')
            fields[key] = value
    if fields.get('schema') != 'numi.cardboard.accepted-coupon-prefix.v1':
        raise ValueError('unexpected prefix manifest schema')
    if fields.get('source_run_directory') != str(Path(expected_output).resolve()):
        raise ValueError('prefix source directory differs from this output')
    if fields.get('accepted_control_index') != str(expected_control):
        raise ValueError('prefix accepted control differs from expected cursor')
    obs = [entry for entry in entries if entry[0] == 'observations.csv' and entry[3]]
    if len(obs) != 1 or sum(entry[3] for entry in entries) != 1:
        raise ValueError('prefix manifest must bind exactly one observations.csv prefix')
    if fields.get('observations_prefix_bytes') != str(obs[0][1]) or fields.get('observations_prefix_sha256') != obs[0][2]:
        raise ValueError('prefix observation header differs from file entry')
    if not entries:
        raise ValueError('prefix manifest has no retained output files')
    total = 0
    for rel, size, digest, _is_observations in entries:
        total += size
        if total > 2 * GIB:
            raise ValueError('prefix inventory exceeds the native 2 GiB total bound')
        file_path = Path(expected_output) / rel
        st = regular_file(file_path, 2 * GIB)
        if st.st_size != size or sha(file_path) != digest:
            raise ValueError(f'prefix retained-file binding mismatch: {rel}')
    return {'accepted_control_index': expected_control, 'entries': len(entries),
            'observations_prefix_bytes': obs[0][1], 'total_bytes': total,
            'source_run_directory': fields['source_run_directory']}


def _expected_env(spec):
    return dict(spec['environment_overrides'])


def _import_environment_module(source):
    module_path = Path(source) / 'matter/tools/cardboard_solver_environment.py'
    module_spec = importlib.util.spec_from_file_location('cardboard_solver_environment_173', module_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError('cannot load source173 environment contract module')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _manifest_inventory_ok(directory, manifest_file):
    directory = Path(directory)
    manifest = load_json(manifest_file)
    listed = manifest.get('files_sha256')
    if not isinstance(listed, dict):
        raise ValueError('validation manifest lacks files_sha256')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p != Path(manifest_file) and '__pycache__' not in p.parts}
    if set(listed) != actual:
        raise ValueError(f'validation report inventory mismatch: missing={sorted(actual-set(listed))}, extra={sorted(set(listed)-actual)}')
    for rel, digest in listed.items():
        p = directory / rel
        if p.is_symlink() or sha(p) != digest:
            raise ValueError(f'validation evidence changed: {rel}')
    return len(listed)


def verify_command_contract(spec):
    env = spec['environment_overrides']
    for key in ('NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR',
                'NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH', 'NM_FEM_OPERATOR_FD_SELECTOR',
                'NM_FEM_OPERATOR_FD_CAPTURE_PATH'):
        if key in env:
            raise ValueError(f'unrequested capture control is set: {key}')
    if env.get('NM_FEM_BASE_CGLS_WARM_START_REFINE') != '1':
        raise ValueError('warm-start refinement must be enabled in both runs')
    a = spec['cases']['473']['native_argv']
    b = spec['cases']['474']['native_argv']
    def physical(argv):
        excluded = {'--output', '--coupon-checkpoint-after-control',
                    '--coupon-resume-checkpoint', '--coupon-resume-metadata'}
        values=[]; i=1
        while i < len(argv):
            if argv[i] in excluded:
                i += 2
            else:
                values.append(argv[i]); i += 1
        return values
    if physical(a) != physical(b):
        raise ValueError('473 and 474 physical/material/protocol arguments differ')
    if a[a.index('--coupon-checkpoint-after-control') + 1] != '0':
        raise ValueError('473 must stop at the accepted zero-based control 0')
    if '--coupon-checkpoint-after-control' in b:
        raise ValueError('474 resume cannot request another checkpoint in this source')
    if b[b.index('--coupon-resume-checkpoint') + 1] != spec['cases']['473']['expected_checkpoint']:
        raise ValueError('474 does not restore the exact 473 control-0 checkpoint')
    if b[b.index('--coupon-resume-metadata') + 1] != spec['cases']['473']['expected_metadata']:
        raise ValueError('474 does not restore the exact 473 control-0 metadata')
    if spec['protocol']['total_controls'] != sum(spec['protocol']['phase_controls'].values()):
        raise ValueError('protocol phase counts do not sum to the declared full schedule')
    return {'same_physical_argv': True, 'warm_start_enabled_both': True,
            'prefix_control': 0, 'resume_start_control': 1,
            'resume_runs_remaining_schedule': True, 'checkpoint_creation_and_resume_combined': False}


def verify_protocol_manifest(path, spec):
    return verify_protocol_manifest_from_mapping(load_json(path), spec)


def verify_protocol_manifest_from_mapping(manifest, spec):
    protocol = spec['protocol']
    solver = manifest.get('solver', {})
    mesh = manifest.get('mesh', {})
    materials = manifest.get('materials', {})
    phases = protocol['phase_controls']
    checks = {
        'schema': manifest.get('schema') == 'numi.cardboard.explicit-strip.v1',
        'preset': manifest.get('preset') == 'literature2009',
        'loading_steps': solver.get('steps') == phases['loading'],
        'hold_steps': solver.get('hold_steps') == phases['hold'],
        'unload_steps': solver.get('unload_steps') == phases['unloading'],
        'relax_steps': solver.get('relax_steps') == phases['relaxation'],
        'release_steps': solver.get('release_steps') == phases['release'],
        'post_release_relaxation_steps': solver.get('post_release_relaxation_steps') == phases['post_release_relaxation'],
        'total_controls': sum(solver.get(name, 0) for name in (
            'steps', 'hold_steps', 'unload_steps', 'relax_steps', 'release_steps',
            'post_release_relaxation_steps')) == protocol['total_controls'],
        'dt_seconds': solver.get('dt_s') == 0.00005,
        'newton_budget': solver.get('newton_iteration_budget') == 64,
        'fgmres_restart': solver.get('fgmres_restart') == 16,
        'fgmres_budget': solver.get('fgmres_iteration_budget') == 1024,
        'line_search_steps': solver.get('line_search_steps') == 24,
        'preconditioner': solver.get('fgmres_preconditioner') == 'regional_galerkin_graph_schwarz_conditioning_only',
        'tangent_cache': solver.get('fgmres_algorithmic_tangent_cache') is True,
        'self_contact_disabled': solver.get('deformable_self_contact') is False,
        'reference_displacement_gradient': solver.get('deformation_gradient') == 'persistent_reference_displacement',
        'compensated_paper_arithmetic': solver.get('paper_compensated_arithmetic') is True,
        'bend_angle': solver.get('bend_angle_deg') == 0.375,
        'nx': mesh.get('nx_per_pitch') == 8,
        'ny': mesh.get('ny') == 2,
        'thickness_slices': mesh.get('thickness_slices') == 1,
        'liner_path': materials.get('liner_path') == protocol['materials']['liner']['path'],
        'liner_sha256': materials.get('liner_sha256') == protocol['materials']['liner']['sha256'],
        'medium_path': materials.get('medium_path') == protocol['materials']['medium']['path'],
        'medium_sha256': materials.get('medium_sha256') == protocol['materials']['medium']['sha256'],
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise ValueError(f'protocol manifest differs from the declared full schedule: {failures}')
    return checks


def verify_static(spec):
    verify_command_contract(spec)
    source = Path(spec['source']['snapshot']).resolve(strict=True)
    frozen_path = Path(spec['source']['frozen_source_file']).resolve(strict=True)
    if source != Path('/Users/home/cardboard-implementation-20261014/snapshots/source-checkpoint-warm-breakdown-173').resolve():
        raise ValueError('unexpected source173 snapshot path')
    if sha(frozen_path) != spec['source']['frozen_source_sha256']:
        raise ValueError('source173 frozen manifest hash changed')
    frozen = load_json(frozen_path)
    if frozen.get('schema') != 'numi.cardboard.frozen-source.v1' or len(frozen.get('source_sha256', {})) != 804:
        raise ValueError('source173 frozen file inventory differs')
    for rel, digest in frozen['source_sha256'].items():
        path = (source / rel).resolve(strict=True)
        if not path.is_relative_to(source) or sha(path) != digest:
            raise ValueError(f'frozen source changed: {rel}')

    build_dir = Path(spec['source']['build_dir']).resolve(strict=True)
    provenance = Path(spec['source']['source_build']).resolve(strict=True)
    if sha(provenance) != spec['source']['source_build_sha256']:
        raise ValueError('source173 build receipt changed')
    built = load_json(provenance)
    if (built.get('schema') != 'numi.cardboard.source-build.v1' or built.get('success') is not True or
            built.get('source_snapshot') != str(source) or built.get('source_sha256') != frozen['source_sha256'] or
            built.get('targets') != spec['source']['build_targets'] or
            built.get('artifacts_sha256') != spec['source']['artifact_sha256']):
        raise ValueError('source173 successful build no longer binds this snapshot/target/artifact inventory')
    for filename, digest in built['artifacts_sha256'].items():
        path = Path(filename).resolve(strict=True)
        if not path.is_relative_to(build_dir) or sha(path) != digest:
            raise ValueError(f'source173 built artifact changed: {filename}')
    for rel, digest in built['build_metadata_sha256'].items():
        path = build_dir / rel
        if sha(path) != digest:
            raise ValueError(f'source173 build metadata changed: {rel}')

    cpu_dir = Path(spec['source']['cpu_validation_manifest']).parent
    if sha(spec['source']['cpu_validation_report']) != spec['source']['cpu_validation_report_sha256']:
        raise ValueError('source173 CPU validation report changed')
    if sha(spec['source']['cpu_validation_manifest']) != spec['source']['cpu_validation_manifest_sha256']:
        raise ValueError('source173 CPU validation manifest changed')
    count = _manifest_inventory_ok(cpu_dir, spec['source']['cpu_validation_manifest'])
    cpu = load_json(spec['source']['cpu_validation_report'])
    if (cpu.get('schema') != 'numi.cardboard.source173.cpu-validation.v1' or
            cpu.get('source_files_verified') != 804 or cpu.get('build_record_sha256') != sha(provenance) or
            cpu.get('build_targets') != spec['source']['build_targets'] or
            cpu.get('artifact_sha256') != spec['source']['artifact_sha256'] or
            cpu.get('gpu_executions') != 0 or cpu.get('physical_transactions') != 0 or
            cpu.get('cpu_test_cases_reported') != 29):
        raise ValueError('source173 CPU validation does not match the bound successful build or scope')
    integration_path = Path(spec['source']['integration_report'])
    if sha(integration_path) != spec['source']['integration_report_sha256']:
        raise ValueError('source173 integration report changed')
    integration = load_json(integration_path)
    if integration.get('frozen_source_sha256') != sha(frozen_path) or integration.get('selected_source_count') != 804:
        raise ValueError('source173 integration report does not bind the frozen source')

    env_module = _import_environment_module(source)
    env_module.validate_source_registry(source)
    parsed = env_module.parse_overrides([f'{k}={v}' for k, v in spec['environment_overrides'].items()])
    if parsed != spec['environment_overrides']:
        raise ValueError('runtime environment overrides failed source173 parser')
    if spec['environment_overrides'].get('NM_FEM_BASE_CGLS_WARM_START_REFINE') != '1':
        raise ValueError('warm-start refinement was not enabled from control zero')
    if any(key.endswith('_PATH') or key.startswith('NM_FEM_OPERATOR_FD') or key == 'NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR'
           for key in spec['environment_overrides']):
        raise ValueError('this checkpoint smoke must not request diagnostic capture paths/selectors')

    for material, item in spec['protocol']['materials'].items():
        path = Path(item['path']).resolve(strict=True)
        if not path.is_relative_to(source) or sha(path) != item['sha256']:
            raise ValueError(f'{material} material changed or escaped source173')
    for alias, digest in spec['package_input_sha256'].items():
        path = PKG / 'inputs' / alias
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError(f'packaged source173 input changed: {alias}')
    return {'source_files_verified': 804, 'build_artifacts_verified': len(built['artifacts_sha256']),
            'cpu_report_files_verified': count, 'cpu_test_cases_reported': 29,
            'source_registry_keys': len(env_module.NATIVE_ENV_KEYS),
            'environment_keys_set': len(parsed), 'gpu_execution_by_preflight': False}


def _expected_case_wrapper(spec, case_key, dynamic_inputs=()):
    record = spec['cases'][case_key]
    wrapper = list(record['wrapper_argv_template_static_inputs_only'] if case_key == '474'
                   else record['wrapper_argv'])
    if dynamic_inputs:
        split = wrapper.index('--')
        native = wrapper[split + 1:]
        pre = wrapper[:split]
        # Dynamic resume artifacts are appended after the sealed static inputs,
        # before the native argv delimiter.
        for path in dynamic_inputs:
            pre += ['--input-file', str(path)]
        wrapper = pre + ['--', *native]
    return wrapper


def verify_warm_checker(spec):
    directory = Path(spec['source']['warm_kernel_check_report_dir'])
    receipt_path = Path(spec['source']['warm_kernel_check_receipt'])
    manifest_path = Path(spec['source']['warm_kernel_check_manifest'])
    if not receipt_path.is_file() or not manifest_path.is_file():
        raise ValueError('required source173 warm-start kernel checker receipt is not present yet')
    manifest = load_json(manifest_path)
    listed = manifest.get('files_sha256')
    actual = {p.name for p in directory.iterdir() if p.is_file() and p != manifest_path}
    if not isinstance(listed, dict) or set(listed) != actual:
        raise ValueError('warm-kernel checker manifest file inventory mismatch')
    for name, digest in listed.items():
        p = directory / name
        if p.is_symlink() or sha(p) != digest:
            raise ValueError(f'warm-kernel check artifact changed: {name}')
    receipt = load_json(receipt_path)
    req = spec['source']['warm_kernel_check_required']
    if (receipt.get('schema') != req['receipt_schema'] or receipt.get('returncode') != 0 or
            receipt.get('timed_out') is not False or receipt.get('gpu_execution') is not True or
            receipt.get('physical_transactions') != 0 or
            receipt.get('production_material_operator_or_full_cgls_recurrence_exercised') is not False or
            receipt.get('frozen_source_sha256') != spec['source']['frozen_source_sha256'] or
            receipt.get('source_build_sha256') != spec['source']['source_build_sha256'] or
            receipt.get('cpu_manifest_sha256') != spec['source']['cpu_validation_manifest_sha256']):
        raise ValueError('source173 warm checker did not pass the exact default-off native checker contract')
    return {'checker_receipt_sha256': sha(receipt_path), 'checker_manifest_sha256': sha(manifest_path),
            'checks_reported': receipt.get('checks_reported'), 'physical_transactions': 0,
            'qualification': 'isolated Metal checker only; no production trajectory/physical qualification'}


def verify_package_inventory():
    manifest = load_json(MANIFEST_PATH)
    if manifest.get('schema') != 'numi.cardboard.native473-474-package.v1':
        raise ValueError('unsupported runner package manifest')
    listed = manifest.get('files_sha256')
    actual = {str(path.relative_to(PKG)) for path in PKG.rglob('*')
              if path.is_file() and path != MANIFEST_PATH and '__pycache__' not in path.parts}
    if set(listed or {}) != actual:
        raise ValueError(f'package inventory mismatch: missing={sorted(actual-set(listed or {}))}, extra={sorted(set(listed or {})-actual)}')
    for rel, digest in listed.items():
        path = PKG / rel
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError(f'runner package changed: {rel}')
    return len(listed)


def verify_fresh_case(spec, key):
    case = Path(spec['cases'][key]['case'])
    if case.parent.resolve(strict=True) != SOLVER.resolve(strict=True) or case.is_symlink() or case.exists():
        raise ValueError(f'{key} case path must be a new, direct child of the solver directory')
    return True


def verify_process_and_capacity(spec):
    active = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,stat=,command='], text=True).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, stat, command = fields
        executable = command.split(None, 1)[0]
        if Path(executable).name != 'numi-matter-cardboard-probe' or stat.startswith(('T', 'Z')):
            continue
        active.append({'pid': int(pid), 'stat': stat, 'command': command})
    if active:
        raise ValueError(f'one-worker guard found active native probe(s): {active}')
    stats = os.statvfs(B)
    free = stats.f_bavail * stats.f_frsize
    if free < spec['launch_policy']['minimum_free_bytes_before_manual_wrapper_launch']:
        raise ValueError(f'launch preflight needs 2 GiB free; observed {free} bytes')
    return {'active_probe_processes': active, 'available_bytes': free}


def _input_paths_from_wrapper(wrapper):
    split = wrapper.index('--')
    paths = []
    index = 0
    while index < split:
        if wrapper[index] == '--input-file':
            if index + 1 >= split:
                raise ValueError('wrapper has an incomplete --input-file')
            paths.append(Path(wrapper[index + 1]))
            index += 2
        else:
            index += 1
    return paths


def _expected_external_input_hashes(input_paths, source_build_digest):
    names = [Path(path).name for path in input_paths] + ['source-build.json']
    if any(not name or name in ('.', '..') for name in names) or len(names) != len(set(names)):
        raise ValueError('explicit external inputs have empty or colliding basenames')
    result = {f'external/{Path(path).name}': sha(path) for path in input_paths}
    result['external/source-build.json'] = source_build_digest
    return result


def verify_effective_source_run(spec, case_key, expected_argv, expected_wrapper):
    case = Path(spec['cases'][case_key]['case'])
    receipt = load_json(case / 'receipt.json')
    if (receipt.get('schema') != 'numi.cardboard.native-run-receipt.v2' or
            receipt.get('run_mode') != 'diagnostic' or receipt.get('execution_started') is not True or
            receipt.get('exit_code') != 0 or receipt.get('timed_out') is not False or
            receipt.get('source_inputs_unchanged') is not True or receipt.get('changed_inputs') != []):
        raise ValueError(f'{case_key} source-bound diagnostic run receipt is not successful and unchanged')
    prediction_path = case / 'prediction.json'
    prediction_raw = prediction_path.read_bytes()
    prediction = json.loads(prediction_raw)
    if (prediction.get('schema') != 'numi.cardboard.native-diagnostic-prediction.v1' or
            prediction.get('run_mode') != 'diagnostic' or prediction.get('argv') != expected_argv or
            prediction.get('environment') != spec['environment_overrides'] or
            prediction.get('source_snapshot') != spec['source']['snapshot'] or
            prediction.get('runtime_dir') != spec['source']['build_dir'] or
            prediction.get('build_provenance') != spec['source']['source_build'] or
            prediction.get('executed_build_verified') is not True):
        raise ValueError(f'{case_key} prediction/source/control binding differs')
    # The source-bound runner records every explicit external input under a
    # basename key. Check these exact source/build/material bindings rather than
    # trusting argv text alone.
    input_paths = _input_paths_from_wrapper(expected_wrapper)
    try:
        expected_external = _expected_external_input_hashes(
            input_paths, spec['source']['source_build_sha256'])
    except ValueError as error:
        raise ValueError(f'{case_key} {error}') from error
    input_map = prediction.get('inputs_sha256')
    if not isinstance(input_map, dict):
        raise ValueError(f'{case_key} prediction lacks retained input hashes')
    for relative, digest in expected_external.items():
        if input_map.get(relative) != digest:
            raise ValueError(f'{case_key} retained input binding differs: {relative}')
    actual_external = {key: value for key, value in input_map.items()
                       if key.startswith('external/')}
    if actual_external != expected_external:
        raise ValueError(f'{case_key} external input inventory contains missing or unexpected members')
    env_path = case / 'native-environment.json'
    env_record = load_json(env_path)
    env_module = _import_environment_module(Path(spec['source']['snapshot']))
    env_module.validate_run_record(env_record)
    env_module.validate_prediction_binding(env_record, prediction_raw)
    expected_contract = env_module.make_diagnostic_contract(spec['environment_overrides'], case, {})
    if env_record.get('contract') != expected_contract:
        raise ValueError(f'{case_key} effective environment differs from the no-capture contract')
    diagnostic = load_json(case / 'diagnostic-captures.json')
    if (diagnostic.get('status') != 'no_capture_paths_requested' or diagnostic.get('files') != [] or
            diagnostic.get('total_bytes') != 0 or
            diagnostic.get('environment_contract_sha256') != expected_contract['sha256']):
        raise ValueError(f'{case_key} unexpectedly requested or retained diagnostic capture payload')
    return {'receipt_sha256': sha(case / 'receipt.json'),
            'prediction_sha256': sha(prediction_path), 'environment_record_sha256': sha(env_path),
            'contract_sha256': expected_contract['sha256']}


def verify_prefix_case(spec, key='473'):
    run = spec['cases'][key]
    case = Path(run['case'])
    output = case / 'output'
    if (case.is_symlink() or not case.is_dir() or case.resolve(strict=True) != case.absolute() or
            output.is_symlink() or not output.is_dir() or output.resolve(strict=True) != output.absolute()):
        raise ValueError(f'{key} case/output directories are missing, aliased, or symlinked')
    receipt_bindings = verify_effective_source_run(spec, key, run['native_argv'], run['wrapper_argv'])
    receipt = load_json(case / 'receipt.json')
    output_dir = output.resolve(strict=True)
    result_path = output / 'result.json'
    result = load_json(result_path)
    protocol_checks = verify_protocol_manifest(output / 'manifest.json', spec)
    control = run['checkpoint_after_control']
    accepted_count = run['accepted_prefix_count']
    if (result.get('schema') != 'numi.cardboard.probe-result.v1' or
            result.get('status') != 'accepted_checkpoint_prefix_saved' or
            result.get('accepted_steps') != accepted_count or result.get('physical_validation') is not False):
        raise ValueError(f'{key} output is not the expected incomplete accepted prefix')
    report = load_json(output / 'coupon-checkpoint-prefix.json')
    if (report.get('schema') != 'numi.cardboard.coupon-checkpoint-prefix.v1' or
            report.get('status') != 'accepted_checkpoint_prefix_saved' or
            report.get('accepted_control_index') != control or report.get('next_control_index') != control + 1 or
            report.get('full_protocol_complete') is not False or report.get('trajectory_parity_established') is not False or
            report.get('physical_validation') is not False):
        raise ValueError(f'{key} checkpoint prefix report has a wrong cursor or overstates completion')
    expected_cp = 'checkpoints/coupon-control-000000.nmcb'
    expected_meta = 'checkpoints/coupon-control-000000.resume'
    expected_prefix = 'checkpoints/coupon-control-000000.prefix'
    if (report.get('checkpoint_file') != expected_cp or report.get('metadata_file') != expected_meta or
            report.get('checkpoint_sha256') != sha(output / expected_cp) or
            report.get('metadata_sha256') != sha(output / expected_meta)):
        raise ValueError('control-0 checkpoint/report binding mismatch')
    cp = output / expected_cp
    meta_path = output / expected_meta
    prefix_path = output / expected_prefix
    cpst = regular_file(cp, 2 * GIB)
    metast = regular_file(meta_path, 64 * 1024)
    prefixst = regular_file(prefix_path, 4 * MIB)
    meta = parse_kv(meta_path)
    expected_meta_pairs = {
      'schema': 'numi.cardboard.accepted-coupon-resume.v2',
      'accepted_control_index': '0', 'next_control_index': '1',
      'source_prefix_control_count': '1', 'environment_count': '2',
      'both_environments_accepted': 'true', 'checkpoint_sha256': sha(cp),
      'checkpoint_bytes': str(cpst.st_size), 'protocol_manifest_sha256': sha(output / 'manifest.json'),
      'prefix_manifest_sha256': sha(prefix_path), 'prefix_manifest_file': 'coupon-control-000000.prefix',
      'source_run_directory': str(output_dir),
    }
    for name, expected in expected_meta_pairs.items():
        if meta.get(name) != expected:
            raise ValueError(f'control-0 resume metadata mismatch for {name}')
    if Path(meta.get('source_run_directory', '')).resolve() != output_dir:
        raise ValueError('checkpoint metadata source-run directory is not the 473 output directory')
    prefix = parse_prefix(prefix_path, control, output_dir)
    if not any(entry[0] == 'manifest.json' for entry in _prefix_entries(prefix_path)):
        raise ValueError('prefix inventory does not retain protocol manifest')
    saved = parse_kv(output / 'coupon-checkpoint-saved.txt')
    if (saved.get('schema') != 'numi.cardboard.accepted-coupon-checkpoint-reference.v1' or
            saved.get('checkpoint') != expected_cp or saved.get('control_index') != '0' or
            saved.get('next_control_index') != '1' or saved.get('continuation_parity_established') != 'false'):
        raise ValueError('checkpoint-saved reference does not identify control0')
    return {'status': 'accepted_incomplete_prefix_verified', 'control': control,
            'accepted_prefix_count': accepted_count, 'checkpoint_bytes': cpst.st_size,
            'checkpoint_sha256': sha(cp), 'metadata_sha256': sha(meta_path),
            'prefix_manifest_sha256': sha(prefix_path), 'prefix_entries': prefix['entries'],
            'result_sha256': sha(result_path), 'source_run_bindings': receipt_bindings,
            'execution_started': receipt['execution_started'], 'full_protocol_complete': False,
            'physical_validation': False, 'protocol_checks': protocol_checks}


def _prefix_entries(path):
    entries = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('file='):
            parts = line[5:].split('|')
            if len(parts) == 4:
                entries.append((parts[0], int(parts[1]), parts[2], parts[3] == '1'))
    return entries


def build_dynamic_resume_wrapper(spec):
    run = spec['cases']['474']
    source_case = Path(spec['cases']['473']['case'])
    extra = [Path(p) for p in run['dynamic_source_inputs']]
    for p in extra:
        regular_file(p, 2 * GIB if p.suffix == '.nmcb' else 64 * MIB)
    return _expected_case_wrapper(spec, '474', extra)


def verify_resumed_output(spec):
    run = spec['cases']['474']
    case = Path(run['case'])
    output = case / 'output'
    if (case.is_symlink() or not case.is_dir() or case.resolve(strict=True) != case.absolute() or
            output.is_symlink() or not output.is_dir() or output.resolve(strict=True) != output.absolute()):
        raise ValueError('474 case/output directories are missing, aliased, or symlinked')
    expected_argv = run['native_argv']
    wrapper = build_dynamic_resume_wrapper(spec)
    binding = verify_effective_source_run(spec, '474', expected_argv, wrapper)
    result = load_json(output / 'result.json')
    segment = load_json(output / 'coupon-resume-segment.json')
    receipt = load_json(case / 'receipt.json')
    source_meta = parse_kv(spec['cases']['473']['expected_metadata'])
    protocol_checks = verify_protocol_manifest(output / 'manifest.json', spec)
    if (result.get('schema') != 'numi.cardboard.probe-result.v1' or
            result.get('status') != 'resumed_diagnostic_segment_reached_schedule_end' or
            result.get('accepted_steps') != run['scheduled_controls_remaining'] or
            result.get('physical_validation') is not False):
        raise ValueError('474 did not reach the scheduled end as a nonqualified diagnostic segment')
    if (segment.get('schema') != 'numi.cardboard.coupon-resume-segment.v1' or
            segment.get('evidence_class') != 'diagnostic_resumed_segment' or
            segment.get('status') != result.get('status') or
            segment.get('source_prefix_accepted_control_count') != 1 or
            segment.get('segment_start_control_index') != 1 or
            segment.get('segment_accepted_controls') != spec['cases']['474']['scheduled_controls_remaining'] or
            segment.get('source_run_id') != source_meta.get('source_run_id') or
            segment.get('checkpoint_sha256') != source_meta.get('checkpoint_sha256') or
            segment.get('protocol_manifest_sha256') != source_meta.get('protocol_manifest_sha256') or
            segment.get('accepted_state_restored_from_native_snapshot') is not True or
            segment.get('accepted_prefix_reconstructed_from_obj_or_json') is not False or
            segment.get('full_protocol_complete') is not False or
            segment.get('trajectory_parity_established') is not False or
            segment.get('physical_validation') is not False):
        raise ValueError('474 resume segment report has a wrong start/count or overstates completion')
    return {'status': 'resumed_segment_reached_scheduled_end', 'segment_controls': segment['segment_accepted_controls'],
            'receipt_exit_code': receipt['exit_code'], 'source_run_bindings': binding,
            'full_uninterrupted_trajectory_parity': False, 'physical_validation': False,
            'protocol_checks': protocol_checks}


def preflight(spec, phase):
    package_count = verify_package_inventory()
    static = verify_static(spec)
    verify_fresh_case(spec, '473' if phase == '473' else '474')
    warm = verify_warm_checker(spec)
    process = verify_process_and_capacity(spec)
    if phase == '474':
        prefix = verify_prefix_case(spec, '473')
        wrapper = build_dynamic_resume_wrapper(spec)
    else:
        prefix = None
        wrapper = spec['cases']['473']['wrapper_argv']
    return {'status': 'preflight_passed_wrapper_not_executed', 'execution_performed': False,
            'phase': phase, 'package_files_verified': package_count, 'static_bindings': static, 'command_contract': verify_command_contract(spec),
            'warm_checker': warm, 'capacity_and_worker_guard': process, 'source473_prefix': prefix,
            'wrapper_argv': wrapper,
            'qualification': 'diagnostic checkpoint/resume smoke only; no complete physical or convergence qualification'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static-audit', action='store_true', help='verify package/source/build/CPU receipts only; no launch checks')
    parser.add_argument('--prepare', choices=('473', '474'), help='validate launch prerequisites and print wrapper argv; never execute')
    parser.add_argument('--verify-473-prefix', action='store_true', help='verify an already-terminal 473 accepted checkpoint prefix')
    parser.add_argument('--verify-474-result', action='store_true', help='verify an already-terminal 474 resumed segment')
    args = parser.parse_args()
    if sum((args.static_audit, args.prepare is not None, args.verify_473_prefix, args.verify_474_result)) != 1:
        parser.error('choose exactly one of --static-audit, --prepare, --verify-473-prefix, --verify-474-result')
    spec = load_json(SPEC_PATH)
    if args.static_audit:
        result = {'status': 'static_source_build_input_audit_passed', 'execution_performed': False,
                  'package_files_verified': verify_package_inventory(), 'static_bindings': verify_static(spec),
                  'command_contract': verify_command_contract(spec),
                  'qualification': 'CPU/source/binary provenance only; warm Metal checker and trajectory remain separate'}
    elif args.prepare:
        result = preflight(spec, args.prepare)
    elif args.verify_473_prefix:
        verify_static(spec)
        result = verify_prefix_case(spec, '473')
    else:
        verify_static(spec)
        verify_prefix_case(spec, '473')
        result = verify_resumed_output(spec)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if 'wrapper_argv' in result:
        print('WRAPPER_ARGV_JSON=' + json.dumps(result['wrapper_argv']))

if __name__ == '__main__':
    main()
