from pathlib import Path
import hashlib, json, os, subprocess, sys, time

root = Path(__file__).resolve().parent.resolve()
invocation = json.loads((root / 'invocation.json').read_text())

def resolved(path):
    return Path(path).resolve()

def sha(path):
    return hashlib.sha256(resolved(path).read_bytes()).hexdigest()

argv = [str(resolved(x)) for x in invocation['argv']]
launcher_argv = [str(resolved(x)) for x in invocation['launcher_argv']]
cwd = resolved(invocation['cwd'])
msl = resolved(invocation['msl_path'])
library = resolved(invocation['library_path'])
output = (root / invocation['output_relative']).resolve()

assert launcher_argv == argv, 'launcher argv diverges from recorded argv'
assert len(argv) == 2 and Path(argv[1]) == msl, 'MSL argv mismatch'
assert cwd == root, 'resolved cwd mismatch'
assert output != root and output.is_relative_to(root), 'resolved output escapes runner root'
assert sha(argv[0]) == invocation['pins']['executable']
assert sha(msl) == invocation['pins']['msl']
assert sha(library) == invocation['pins']['library']
env = invocation['environment']
assert resolved(env['DYLD_LIBRARY_PATH']) == library.parent, 'DYLD library directory mismatch'
assert resolved(env['NUMI_MATTER_TEST_SOURCE']) == msl, 'source override path mismatch'
assert env['NUMI_MATTER_TEST_SOURCE_SHA256'] == invocation['pins']['msl']

if sys.argv[1:] == ['--preflight']:
    print('PREFLIGHT PASS: resolved argv, cwd, output fence, executable, MSL and library pins match')
    raise SystemExit(0)
assert not sys.argv[1:], 'only --preflight is accepted'
output.mkdir(exist_ok=False)
start = time.monotonic()
with (output / 'stdout.log').open('wb') as stdout_file, (output / 'stderr.log').open('wb') as stderr_file:
    process = subprocess.run(
        launcher_argv, cwd=cwd,
        env={**os.environ, **env}, stdout=stdout_file, stderr=stderr_file,
        timeout=invocation.get('timeout_s', 180), check=False,
    )
result = {
    'exit_code': process.returncode,
    'elapsed_s': time.monotonic() - start,
    'executable_sha256': sha(argv[0]),
    'library_sha256': sha(library),
    'msl_sha256': sha(msl),
    'stdout_sha256': sha(output / 'stdout.log'),
    'stderr_sha256': sha(output / 'stderr.log'),
    'output_directory': str(output),
}
(output / 'result.json').write_text(json.dumps(result, indent=2) + chr(10))
print(json.dumps(result, indent=2))
