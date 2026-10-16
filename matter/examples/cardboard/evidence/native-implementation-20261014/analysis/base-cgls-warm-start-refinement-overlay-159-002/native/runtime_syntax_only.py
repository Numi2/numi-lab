#!/usr/bin/env python3
"""Syntax-check the candidate runtime.mm with frozen-157 build flags; no linking."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

root = Path('/Users/home/cardboard-implementation-20261014')
package = root / 'analysis/base-cgls-warm-start-refinement-overlay-159-002'
snapshot = root / 'snapshots/source-right-grip-telemetry-157'
build = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-157')
db_path = build / 'compile_commands.json'
db = json.loads(db_path.read_text())
entry = next(x for x in db if x.get('file', '').endswith('/matter/src/runtime.mm'))
args = shlex.split(entry['command'])
args = [x for x in args if x != '-c']
if '-o' in args:
    i = args.index('-o')
    del args[i:i + 2]
candidate = package / 'candidate/matter/src'
args += [
    '-I' + str(snapshot / 'matter/src'),
    '-I' + str(candidate),
    '-I' + str(candidate / 'metal'),
    '-fsyntax-only',
    str(candidate / 'runtime.mm'),
]
proc = subprocess.run(args, cwd=entry['directory'], text=True,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
record = {
    'schema': 'numi.cardboard.candidate_runtime_syntax_check.v1',
    'base_snapshot': str(snapshot),
    'base_frozen_source_sha256': hashlib.sha256((snapshot / 'frozen-source.json').read_bytes()).hexdigest(),
    'compile_commands': str(db_path),
    'compile_commands_sha256': hashlib.sha256(db_path.read_bytes()).hexdigest(),
    'directory': entry['directory'],
    'command_argv': args,
    'candidate_runtime_sha256': hashlib.sha256((candidate / 'runtime.mm').read_bytes()).hexdigest(),
    'exit_code': proc.returncode,
    'output': proc.stdout,
    'native_execution': False,
}
(package / 'native/runtime-syntax-only.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: record[k] for k in ('exit_code', 'candidate_runtime_sha256', 'native_execution')}, sort_keys=True))
if proc.returncode:
    print(proc.stdout, end='')
raise SystemExit(proc.returncode)
