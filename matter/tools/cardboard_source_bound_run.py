#!/usr/bin/env python3
"""Seal and retain a source-bound native cardboard run and any failure/timeout."""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time

parser = argparse.ArgumentParser()
parser.add_argument('--case', required=True)
parser.add_argument('--question', required=True)
parser.add_argument('--prediction', required=True)
parser.add_argument('--timeout', type=int, default=1800)
parser.add_argument('--env', action='append', default=[])
parser.add_argument('--source-snapshot')
parser.add_argument('--runtime-dir')
parser.add_argument('--build-provenance', help='successful fresh-build receipt; checked before native launch')
parser.add_argument('--input-file', action='append', default=[])
parser.add_argument('argv', nargs=argparse.REMAINDER)
args = parser.parse_args()
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
seal_started = time.monotonic()
def retain_unhandled_failure(kind, error, traceback):
    (case/'receipt.json').write_text(json.dumps({
        'exit_code':1,'timed_out':False,
        'execution_started':execution_started,
        'stage':'execution' if execution_started else 'input-seal',
        'error_type':kind.__name__,'reason':str(error),
        'wall_seconds':time.monotonic()-seal_started,
        'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},
        indent=2)+'\n')
    sys.__excepthook__(kind,error,traceback)
sys.excepthook = retain_unhandled_failure
if not args.argv:
    parser.error('an executable command is required after --')
argv = args.argv[1:] if args.argv[0] == '--' else args.argv
if not argv:
    parser.error('an executable command is required after --')
argv[0] = str(Path(argv[0]).resolve())
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
    args.input_file.append(str(provenance_path))
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
for external in args.input_file:
    path = Path(external)
    relative = 'external/' + path.name
    if relative in bindings: raise ValueError('duplicate external input basename')
    target = case/'inputs'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    bindings[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    original_bindings[str(path.resolve())] = bindings[relative]
env_delta = dict(value.split('=', 1) for value in args.env)
seal = {'schema':'numi.cardboard.native-prediction.v1',
        'sealed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'question':args.question,'prediction':args.prediction,
        'argv':argv,'environment':env_delta,
        'source_snapshot':str(source_root),'runtime_dir':str(runtime_dir),
        'external_inputs':args.input_file,'inputs_sha256':bindings,
        'executed_build_verified':build_provenance is not None,
        'build_provenance':args.build_provenance,
        'source_binding_limitation':'Listed source, executable and Metal libraries retained; compiler system headers and OS/driver not archived.',
        'timeout_seconds':args.timeout}
(case/'prediction.json').write_text(json.dumps(seal,indent=2)+'\n')
environment = os.environ.copy()
environment.update(env_delta)
started = time.monotonic()
with (case/'stdout.log').open('w') as out, (case/'stderr.log').open('w') as err:
    process = subprocess.Popen(argv, stdout=out, stderr=err, env=environment, start_new_session=True)
    execution_started = True
    timed_out = False
    try:
        code = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try: code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
changed_inputs = []
for path, digest in original_bindings.items():
    try: current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError: current = None
    if current != digest: changed_inputs.append(path)
receipt = {'schema':'numi.cardboard.native-run-receipt.v2',
           'execution_started':execution_started,
           'source_inputs_unchanged':not changed_inputs,
           'changed_inputs':changed_inputs,
           'exit_code':code,'timed_out':timed_out,
           'wall_seconds':time.monotonic()-started,
           'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(case/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(124 if timed_out else (3 if changed_inputs else (code if code >= 0 else 128-code)))
