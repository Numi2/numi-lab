from pathlib import Path
import datetime, hashlib, json, subprocess, time

B = Path('/Users/home/cardboard-implementation-20261014')
P = B / 'analysis/native-base-cgls-diagnostic-443-001'
O = B / 'native/base-cgls-stage-diagnostic-446/attempt-002'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not O.exists()
O.mkdir()
checker = P / 'native/base_cgls_native_check.mm'
library = P / 'native/build/BaseCGLSCheck.metallib'
helper = B / 'snapshots/source-bounded-native-cgls-141/matter/src/metal/base_cgls.metalinc'
bound = [checker, library, helper, P / 'source-overlay.json',
         P / 'native/base_cgls_test_actions.metal']
before = {str(p): sha(p) for p in bound}
compile_command = ['clang++', '-std=c++20', '-fobjc-arc', '-O2',
    '-Wall', '-Wextra', '-Wpedantic', '-Werror',
    '-DNUMI_BASE_CGLS_CHECK_METALLIB="' + str(library) + '"',
    str(checker), '-framework', 'Foundation', '-framework', 'Metal',
    '-o', str(O / 'base-cgls-native-check')]
(O / 'compile-command.json').write_text(json.dumps(compile_command, indent=2) + '\n')
with (O / 'compile.log').open('x') as out:
    c = subprocess.run(compile_command, stdout=out, stderr=subprocess.STDOUT)
assert c.returncode == 0
command = [str(O / 'base-cgls-native-check')]
(O / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
start = time.monotonic()
timeout = False
with (O / 'stdout.log').open('x') as out, (O / 'stderr.log').open('x') as err:
    try:
        p = subprocess.run(command, stdout=out, stderr=err, timeout=240)
        code = p.returncode
    except subprocess.TimeoutExpired:
        timeout = True
        code = None
after = {str(p): sha(p) for p in bound}
receipt = dict(schema='numi.cardboard.direct_base_cgls_kernel_check.v1',
    source_inputs_unchanged=before == after, input_hashes=before,
    binary_sha256=sha(O / 'base-cgls-native-check'), execution_started=True,
    exit_code=code, timed_out=timeout, wall_seconds=time.monotonic()-start,
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    evidence_scope='Diagnostic-only stage trace of unchanged native443 helper recurrence and synthetic CSR actions. Does not qualify production FEM action parity or a nonlinear physical step.',
    physical_steps=0)
(O / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(O / 'runner.py').write_bytes(Path(__file__).read_bytes())
(O / 'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(O)):
    dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(O.iterdir()) if p.is_file()}), indent=2) + '\n')
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if code == 0 and before == after and not timeout else 2)
