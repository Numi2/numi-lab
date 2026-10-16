#!/usr/bin/env python3
"""Exercise native argument admission without file output or a Metal runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', required=True)
    args = parser.parse_args()
    probe = str(Path(args.probe).resolve(strict=True))
    cases = [
        ('checkpoint0', ['--coupon-checkpoint-after-control', '0'], 0, None),
        ('checkpoint19', ['--coupon-checkpoint-after-control', '19'], 0, None),
        ('resume', ['--coupon-resume-checkpoint', 'missing.nmcb', '--coupon-resume-metadata', 'missing.resume'], 0, None),
        ('combined-resume-save', ['--coupon-checkpoint-after-control', '19', '--coupon-resume-checkpoint', 'missing.nmcb', '--coupon-resume-metadata', 'missing.resume'], 1, 'coupon checkpoint creation and resume cannot be combined'),
        ('invalid-dt', ['--dt', '0'], 1, None),
        ('restart-over-ceiling', ['--fgmres-restart', '17'], 1, None),
        ('invalid-newton', ['--newton-iterations', '0'], 1, None),
        ('invalid-material-budget', ['--material-iterations', '33'], 1, None),
    ]
    with tempfile.TemporaryDirectory(prefix='cardboard-argument-admission-') as temporary:
        for name, options, code, message in cases:
            output = Path(temporary) / name
            command = [probe, '--preset', 'literature2009', '--output', str(output),
                       '--steps', '24', '--validate-only'] + options
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            if result.returncode != code:
                raise AssertionError((name, result.returncode, result.stdout, result.stderr))
            if output.exists():
                raise AssertionError((name, 'argument admission created a simulation output'))
            if code == 0:
                expected = {'schema': 'numi.cardboard.probe-argument-validation.v1',
                            'arguments_valid': True, 'file_inputs_validated': False,
                            'mesh_generated': False, 'native_runtime_created': False}
                if json.loads(result.stdout) != expected:
                    raise AssertionError((name, result.stdout))
            elif message and message not in result.stderr:
                raise AssertionError((name, result.stderr))
    print(json.dumps({'schema': 'numi.cardboard.argument-admission-check.v1',
                      'cases': len(cases), 'passed': True, 'native_gpu_dispatch_requested': False}))


if __name__ == '__main__':
    main()
