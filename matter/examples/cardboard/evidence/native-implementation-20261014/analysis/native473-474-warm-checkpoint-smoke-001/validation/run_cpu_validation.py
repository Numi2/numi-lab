#!/usr/bin/env python3
"""Run bounded package/contract checks; never launches the native probe."""
from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time


PACKAGE = Path(__file__).resolve().parents[1]
VALIDATION = PACKAGE / 'validation'
SPEC = json.loads((PACKAGE / 'run-spec.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


module_spec = importlib.util.spec_from_file_location('warm_smoke_prepare_validation', PACKAGE / 'prepare.py')
prepare = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(prepare)
static = prepare.verify_static(SPEC)
command_contract = prepare.verify_command_contract(SPEC)

env = dict(os.environ)
env['PYTHONDONTWRITEBYTECODE'] = '1'
started = time.time()
argv = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
        str(PACKAGE / 'tests'), '-v']
test = subprocess.run(argv, cwd=PACKAGE, env=env, text=True, capture_output=True)
(VALIDATION / 'unittest.stdout.txt').write_text(test.stdout)
(VALIDATION / 'unittest.stderr.txt').write_text(test.stderr)
if test.returncode != 0:
    raise SystemExit(test.returncode)
elapsed = time.time() - started
receipt = {
    'schema': 'numi.cardboard.native473-474.cpu-validation.v1',
    'argv': argv,
    'working_directory': str(PACKAGE),
    'python_version': platform.python_version(),
    'returncode': test.returncode,
    'test_count': 8,
    'stdout_sha256': sha(VALIDATION / 'unittest.stdout.txt'),
    'stderr_sha256': sha(VALIDATION / 'unittest.stderr.txt'),
    'elapsed_seconds': elapsed,
    'source_static_audit': static,
    'command_contract': command_contract,
    'native_probe_executed': False,
    'gpu_execution': False,
    'physical_transactions': 0,
    'warm_kernel_checker_receipt_present_at_validation': Path(SPEC['source']['warm_kernel_check_receipt']).is_file(),
    'qualification': 'CPU verifier/contract checks only; no checkpoint or trajectory execution',
}
(VALIDATION / 'cpu-validation-receipt.json').write_text(
    json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
