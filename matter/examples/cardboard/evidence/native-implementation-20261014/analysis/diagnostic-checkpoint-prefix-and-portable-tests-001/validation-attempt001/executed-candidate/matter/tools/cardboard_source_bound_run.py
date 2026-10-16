#!/usr/bin/env python3
"""Seal and retain a source-bound native cardboard run and any failure/timeout."""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from cardboard_source_bound_run_support import wait_with_timeout, write_failure_receipt
from cardboard_calibration_case_binding import (
    CalibrationBindingError, validate_executed_manifest, validate_pre_run,
)
import cardboard_solver_environment as solver_environment


def _single_argv_value(argv, flag):
    matches = [index for index, value in enumerate(argv) if value == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(argv):
        raise ValueError('native argv must contain exactly one '+flag+' value')
    return argv[matches[0] + 1]

parser = argparse.ArgumentParser()
parser.add_argument('--case', required=True)
parser.add_argument('--question', required=True)
parser.add_argument('--prediction', required=True)
parser.add_argument('--timeout', type=int, default=1800)
parser.add_argument('--env', action='append', default=[])
parser.add_argument('--diagnostic-run', action='store_true',
                    help='mark this run diagnostic-only; it cannot establish convergence qualification')
parser.add_argument('--diagnostic-capture', action='append', default=[],
                    choices=sorted(solver_environment.CAPTURE_PATH_KEYS),
                    help='request a registered capture path; the runner creates its fresh case-local prefix')
parser.add_argument('--source-snapshot')
parser.add_argument('--runtime-dir')
parser.add_argument('--build-provenance', help='successful fresh-build receipt; checked before native launch')
parser.add_argument('--input-file', action='append', default=[])
parser.add_argument('--ensemble-plan', help='sealed BCT calibration plan; required for --shipping-box')
parser.add_argument('--case-binding', help='prepared candidate/condition binding; required for --shipping-box')
parser.add_argument('argv', nargs=argparse.REMAINDER)
args = parser.parse_args()
environment_overrides = solver_environment.parse_overrides(args.env)
if args.diagnostic_capture and not args.diagnostic_run:
    parser.error('--diagnostic-capture requires --diagnostic-run')
if len(args.diagnostic_capture) != len(set(args.diagnostic_capture)):
    parser.error('duplicate --diagnostic-capture key')
effective_environment_contract = None
diagnostic_prefixes = {}
repo = Path(__file__).resolve().parents[2]
case = Path(args.case).resolve()
# One native GPU worker per checkout. Compilation does not acquire this lock.
worker_lock_directory = repo / 'build-cardboard-native-worker'
worker_lock_directory.mkdir(exist_ok=True)
worker_lock_path = worker_lock_directory / 'worker.lock'
worker_lock = worker_lock_path.open('a')
try:
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError('another source-bound native worker owns this checkout')
case.mkdir(parents=True, exist_ok=False)
execution_started = False
run_state = {'timed_out': False, 'termination_errors': []}
seal_started = time.monotonic()
def retain_unhandled_failure(kind, error, traceback):
    write_failure_receipt(
        case/'receipt.json', error, state=run_state,
        execution_started=execution_started,
        wall_seconds=time.monotonic()-seal_started,
        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    sys.__excepthook__(kind,error,traceback)
sys.excepthook = retain_unhandled_failure
if not args.argv:
    parser.error('an executable command is required after --')
argv = args.argv[1:] if args.argv[0] == '--' else args.argv
if not argv:
    parser.error('an executable command is required after --')
argv[0] = str(Path(argv[0]).resolve())
if args.diagnostic_run:
    diagnostic_prefixes = solver_environment.create_diagnostic_capture_prefixes(
        case, args.diagnostic_capture)
    effective_environment_contract = solver_environment.make_diagnostic_contract(
        environment_overrides, case, diagnostic_prefixes)
else:
    effective_environment_contract = solver_environment.make_contract(environment_overrides)
calibration_identity = None
calibration_binding_path = None
required_calibration_inputs = set()
sealed_external_inputs = [str(Path(value).resolve()) for value in args.input_file]
if '--shipping-box' in argv:
    if not args.ensemble_plan or not args.case_binding:
        raise ValueError('shipping-box runs require --ensemble-plan and --case-binding before execution')
    calibration_binding_path = Path(args.case_binding).resolve()
    calibration_identity, auto_inputs = validate_pre_run(
        Path(args.ensemble_plan), calibration_binding_path, argv, case/'output')
    required_calibration_inputs = {path.resolve() for path in auto_inputs}
    existing = {Path(value).resolve() for value in sealed_external_inputs}
    for path in auto_inputs:
        if path.resolve() not in existing:
            sealed_external_inputs.append(str(path.resolve()))
            existing.add(path.resolve())
elif args.ensemble_plan or args.case_binding:
    raise ValueError('BCT plan/binding options require --shipping-box in the native argv')
source_root = Path(args.source_snapshot).resolve() if args.source_snapshot else repo
runtime_dir = Path(args.runtime_dir).resolve() if args.runtime_dir else repo/'build-cardboard'
build_provenance = None
if args.build_provenance:
    provenance_path = Path(args.build_provenance).resolve()
    build_provenance = json.loads(provenance_path.read_text())
    if build_provenance.get('schema') != 'numi.cardboard.source-build.v1' or not build_provenance.get('success'):
        raise ValueError('build provenance is not a successful fresh source build')
    if Path(build_provenance['source_snapshot']).resolve() != source_root or Path(build_provenance['build_dir']).resolve() != runtime_dir:
        raise ValueError('build provenance source/runtime paths do not match this run')
    for relative, expected in build_provenance['source_sha256'].items():
        path = (source_root/relative).resolve()
        if not path.is_relative_to(source_root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('source no longer matches its executed build: '+relative)
    if argv[0] not in build_provenance['artifacts_sha256']:
        raise ValueError('requested executable is not identified by this build')
    for filename, expected in build_provenance['artifacts_sha256'].items():
        path = Path(filename).resolve()
        if not path.is_relative_to(runtime_dir) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('artifact no longer matches its executed build: '+filename)
    if str(provenance_path) not in sealed_external_inputs:
        sealed_external_inputs.append(str(provenance_path))
files = []
for directory in ('matter/include', 'matter/src', 'include', 'src', 'numi', 'cmake', 'schemas',
                  'matter/examples/cardboard/source-data'):
    files.extend(p for p in (source_root/directory).rglob('*') if p.is_file())
files.extend(p for p in (source_root/'matter/tools').iterdir() if p.is_file())
files.extend(p for p in (source_root/'matter/materials').iterdir() if p.is_file())
files.append(source_root/'matter/CMakeLists.txt')
if (source_root/'CMakeLists.txt').is_file(): files.append(source_root/'CMakeLists.txt')
protocol_path = source_root/'matter/examples/cardboard/cardboard_shipping_300x200_protocol.json'
if protocol_path.is_file(): files.append(protocol_path)
files.extend(runtime_dir.glob('*.dylib'))
files.extend((runtime_dir/'lib').glob('*.dylib'))
files.append(Path(argv[0]))
files.extend((runtime_dir/'shaders').glob('*.metallib'))
files.extend((runtime_dir/'matter/shaders').glob('*.metallib'))
bindings = {}
original_bindings = {}
for path in sorted(set(files)):
    path = path.resolve()
    if path.is_relative_to(source_root):
        relative = Path('source') / path.relative_to(source_root)
    elif path.is_relative_to(runtime_dir):
        relative = Path('runtime') / path.relative_to(runtime_dir)
    else:
        relative = Path('executable') / path.name
    target = case/'inputs'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    bindings[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    original_bindings[str(path)] = bindings[str(relative)]
for external in sealed_external_inputs:
    path = Path(external).resolve()
    relative = 'external/' + path.name
    if relative in bindings: raise ValueError('duplicate external input basename')
    target = case/'inputs'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    copied_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    if calibration_identity is not None:
        expected_calibration_hash = calibration_identity['external_input_sha256'].get(str(path))
        if expected_calibration_hash is not None and copied_sha != expected_calibration_hash:
            raise ValueError('calibration input changed between pre-run validation and sealing: '+path.name)
        if expected_calibration_hash is None and path in required_calibration_inputs:
            raise ValueError('required calibration input is absent from the pre-run hash identity: '+path.name)
    bindings[relative] = copied_sha
    original_bindings[str(path.resolve())] = bindings[relative]
env_delta = dict(environment_overrides)
if args.diagnostic_run:
    env_delta.update({key: effective_environment_contract['effective'][key]['value']
                      for key in diagnostic_prefixes})
seal = {'schema':('numi.cardboard.native-diagnostic-prediction.v1' if args.diagnostic_run
                  else 'numi.cardboard.native-prediction.v1'),
        'sealed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'question':args.question,'prediction':args.prediction,
        'argv':argv,'environment':env_delta,
        'run_mode':'diagnostic' if args.diagnostic_run else 'qualification',
        'effective_native_environment_contract': effective_environment_contract,
        'effective_native_environment_contract_sha256': effective_environment_contract['sha256'],
        'source_snapshot':str(source_root),'runtime_dir':str(runtime_dir),
        'external_inputs':sealed_external_inputs,'inputs_sha256':bindings,
        'native_protocol_id': (calibration_identity['native_protocol_id'] if calibration_identity else None),
        'calibration_case_binding': calibration_identity,
        'executed_build_verified':build_provenance is not None,
        'build_provenance':args.build_provenance,
        'source_binding_limitation':'Listed source, executable and Metal libraries retained; compiler system headers and OS/driver not archived.',
        'timeout_seconds':args.timeout}
(case/'prediction.json').write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n')
prediction_sha256 = hashlib.sha256((case/'prediction.json').read_bytes()).hexdigest()
environment, removed_environment_keys = solver_environment.child_environment(
    os.environ, effective_environment_contract)
if args.diagnostic_run:
    environment_record = solver_environment.make_diagnostic_run_record(
        effective_environment_contract, removed_inherited_keys=removed_environment_keys,
        prediction_sha256=prediction_sha256)
else:
    environment_record = solver_environment.make_run_record(
        effective_environment_contract, producer='source-bound-native-runner',
        removed_inherited_keys=removed_environment_keys,
        prediction_sha256=prediction_sha256)
solver_environment.write_run_record_new(case/'native-environment.json', environment_record)
started = time.monotonic()
with (case/'stdout.log').open('w') as out, (case/'stderr.log').open('w') as err:
    process = subprocess.Popen(argv, stdout=out, stderr=err, env=environment, start_new_session=True)
    execution_started = True
    code = wait_with_timeout(
        process, args.timeout, state=run_state, kill_process_group=os.killpg,
        term_grace_seconds=20,
    )
changed_inputs = []
for path, digest in original_bindings.items():
    try: current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError: current = None
    if current != digest: changed_inputs.append(path)
calibration_validation = None
calibration_validation_error = None
if calibration_identity is not None and code == 0 and not run_state['timed_out']:
    try:
        manifest_path = case/'output'/'manifest.json'
        calibration_validation = validate_executed_manifest(
            manifest_path, calibration_identity, calibration_binding_path)
    except (CalibrationBindingError, OSError, KeyError, TypeError, ValueError) as error:
        calibration_validation_error = str(error)
        calibration_validation = {
            'status': 'failed',
            'manifest_sha256': (hashlib.sha256((case/'output'/'manifest.json').read_bytes()).hexdigest()
                                if (case/'output'/'manifest.json').is_file() else None),
            'error': calibration_validation_error,
        }
elif calibration_identity is not None:
    calibration_validation = {'status': 'not_run', 'reason': 'native process did not complete successfully'}
diagnostic_capture_validation = None
diagnostic_capture_validation_error = None
if args.diagnostic_run:
    try:
        diagnostic_capture_validation = solver_environment.validate_diagnostic_outputs(
            case, diagnostic_prefixes, native_argv=argv, native_exit_code=code,
            effective_contract=effective_environment_contract)
        output_record = {
            'schema': 'numi.cardboard.diagnostic-capture-files.v1',
            'environment_contract_sha256': effective_environment_contract['sha256'],
            **diagnostic_capture_validation,
        }
        with (case/'diagnostic-captures.json').open('x', encoding='utf-8') as stream:
            json.dump(output_record, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError) as error:
        diagnostic_capture_validation_error = str(error)
        diagnostic_capture_validation = {'status': 'failed', 'reason': diagnostic_capture_validation_error}
receipt = {'schema':'numi.cardboard.native-run-receipt.v2',
           'execution_started':execution_started,
           'source_inputs_unchanged':not changed_inputs,
           'changed_inputs':changed_inputs,
           'exit_code':code,'timed_out':run_state['timed_out'],
           'termination_errors':run_state['termination_errors'],
           'wall_seconds':time.monotonic()-started,
           'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if args.diagnostic_run:
    receipt['run_mode'] = 'diagnostic'
    receipt['diagnostic_capture_validation'] = diagnostic_capture_validation
if calibration_identity is not None:
    receipt.update({
        'native_protocol_id': calibration_identity['native_protocol_id'],
        'calibration_plan_sha256': calibration_identity['plan_sha256'],
        'calibration_case_binding_sha256': calibration_identity['case_binding_sha256'],
        'calibration_candidate_id': calibration_identity['candidate_id'],
        'calibration_candidate_sha256': calibration_identity['candidate_sha256'],
        'calibration_condition_id': calibration_identity['condition_id'],
        'calibration_parameter_validation': calibration_validation,
    })
(case/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(124 if run_state['timed_out'] else
                 (3 if changed_inputs or calibration_validation_error or diagnostic_capture_validation_error else
                  (code if code >= 0 else 128-code)))
