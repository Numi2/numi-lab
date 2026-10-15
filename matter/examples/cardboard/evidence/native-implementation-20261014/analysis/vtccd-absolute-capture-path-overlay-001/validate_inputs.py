#!/usr/bin/env python3
"""CPU-only exact-capture path-resolution regression for source125 checker."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
NATIVE = Path('/Users/home/cardboard-implementation-20261014/native')
CHECKER = ROOT / 'numi-matter-cardboard-vtccd-budget-replay-check'
CAPTURES = [
    NATIVE / 'full-contact-trial21-vtccd-call-405' / 'trial-capture.coupled.json',
    NATIVE / 'full-contact-trial22-vtccd-call-406' / 'trial-capture.coupled.json',
]
EXPECTED_PAYLOADS = [
    'eb6286c09081577e226eec6deb11021b4e483e104583214216b6b77c441190b5',
    'd1c3e4d3c74d3bf75a72ade703557b4cfa9a70d62f1f51683fffbcff8546bd74',
]

def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(CHECKER), *args], text=True,
                          capture_output=True, check=False)

def strict_load(text: str):
    def reject(token: str):
        raise ValueError(f'nonstandard JSON token {token}')
    return json.loads(text, parse_constant=reject)

def main() -> None:
    positive = run('--validate-inputs-only', *(str(p) for p in CAPTURES))
    assert positive.returncode == 0, positive.stderr
    result = strict_load(positive.stdout)
    assert result['schema'] == 'numi.matter.vtccd_capture_input_validation.v1'
    assert result['status'] == 'PASS_CAPTURE_INPUTS'
    assert result['metal_initialized'] is False
    assert [row['payload_sha256'] for row in result['captures']] == EXPECTED_PAYLOADS
    assert [row['observer_sha256'] for row in result['captures']] == [
        'ed119deb040fcc91e33e4eafe73e948cdd34946960c4e1ecee2decc6d600d097',
        '981fb3efe37dd2463b86e756b5a312ce33c619c8a93110c7ea4580a16d5c2277',
    ]

    # A modified manifest that points across capture directories must fail at
    # the resolved-parent guard, before payload/hash admission.
    bad_root = json.loads(CAPTURES[0].read_text())
    bad_root['payload']['file'] = str(CAPTURES[1].with_suffix('.bin'))
    with tempfile.TemporaryDirectory(prefix='vtccd-path-negative-') as td:
        bad_manifest = Path(td) / CAPTURES[0].name
        bad_manifest.write_text(json.dumps(bad_root))
        negative = run('--validate-inputs-only', str(bad_manifest), str(CAPTURES[1]))
    assert negative.returncode != 0
    assert 'capture payload must resolve to a sibling of its manifest' in negative.stderr

    print(json.dumps({
        'status': 'PASS_CPU_EXACT_CAPTURE_PATH_CONTRACT',
        'positive': '405/406 manifests, payload SHA/extents, observer arrays and return words loaded before Metal initialization',
        'negative': 'cross-directory absolute payload rejected before payload hash admission',
        'metal_initialized': False,
    }, sort_keys=True))

if __name__ == '__main__':
    main()
