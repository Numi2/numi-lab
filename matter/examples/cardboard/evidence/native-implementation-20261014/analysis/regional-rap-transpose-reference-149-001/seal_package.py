#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source_snapshot = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-block-right-preconditioned-149')
source_files = [
    source_snapshot / 'matter/src/metal/fgmres.metalinc',
    source_snapshot / 'matter/src/runtime.mm',
    source_snapshot / 'matter/src/metal/base_cgls.metalinc',
]
package_files = [
    'README.md', 'audit_source_contract.py', 'regional_rap_reference.py',
    'test_regional_rap_reference.py', 'source-contract.json',
    'source-contract.stdout.json', 'test-receipt.stdout.txt', 'seal_package.py',
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_contract = json.loads((root / 'source-contract.json').read_text())
report = {
  'schema': 'numi.matter.regional_rap_transpose_reference_report.v1',
  'package': str(root),
  'source_snapshot': str(source_snapshot),
  'source_snapshot_status': 'immutable source149 snapshot; source hashes verified by audit_source_contract.py',
  'verdict': {
    'selectors_3_4': 'CPU mathematical forward and algebraic-transpose reference implemented; 7/7 bounded manufactured tests pass.',
    'selector_5': 'UNSUPPORTED in this package. Source equations are audited, but owner confirms no actual same-generation selector-5 capture exists yet. No generic-map test is evidence for selector 5.',
    'native_parity': 'No native arbitrary-vector output was captured or tested. No GPU was run.',
  },
  'selector5_source_derivation': {
    'capture_point_required': 'after graph patch matrix assembly/factorization and coarse RAP construction for the exact selected Newton linearization, before any later generation can overwrite the arrays',
    'forward_real_arithmetic': 'c=R^-1 P^T r; q=r-AP c; patch_rhs=E q; patch_solution=D patch_rhs; S r=P c+G patch_solution',
    'operator_names': {'R': 'P^T A P', 'Q': 'A P', 'E': 'per-patch extraction', 'G': 'per-patch scatter averaged by nodeCoverage'},
    'refined_patch_map': 'For exact real arithmetic, with F denoting the captured pivoted-LU solve and A_p the captured original patch matrix, D=F+F(I-A_p F)=2F-F A_p F. The production implementation is FP32 and uses a two-word compensated residual plus a conditional exact-zero refinement skip; the formula is not a bitwise FP32 emulation.',
    'transpose_real_arithmetic': 'H=G D E; S=H+(P-HQ)R^-1P^T; S^T=H^T+P R^-T(P^T-Q^T H^T), H^T=E^T D^T G^T, D^T=2F^T-F^T A_p^T F^T.',
  },
  'selector5_same_generation_capture_required': source_contract['findings']['selector5_required_capture'],
  'selector5_capture_status': 'Owner has confirmed prospective source151 schema coverage for graph state, but no selector-5 capture payload exists at this report cut.',
  'source_anchors': source_contract['anchors'],
  'source_file_sha256': {str(p): sha(p) for p in source_files},
  'cpu_verification': {
    'command': '/Users/home/.pyenv/versions/3.10.12/bin/python3 -m unittest -v test_regional_rap_reference.py',
    'result': 'PASS 7 tests',
    'receipt_sha256': sha(root / 'test-receipt.stdout.txt'),
    'source_contract_command': '/Users/home/.pyenv/versions/3.10.12/bin/python3 audit_source_contract.py',
    'source_contract_result': 'PASS; all three source hashes and graph/RAP anchors matched',
    'source_contract_sha256': sha(root / 'source-contract.json'),
  },
  'limits': [
    'Reference tests use manufactured small matrices; they do not validate production candidate values or a native captured generation.',
    'The transpose is an independent algebraic CPU implementation because source149 contains no regional RAP or selector-5 transpose kernel.',
    'Selector-5 exact same-generation forward-map reconstruction additionally requires graph arrays and candidate/factor state from one capture generation; the arbitrary-vector native action still needs a native checker to establish forward parity.',
    'The scope is not a CGLS trajectory, convergence, or physical qualification result.',
  ],
}
(root / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
manifest = {
  'schema': 'numi.matter.regional_rap_transpose_reference_manifest.v1',
  'package': str(root),
  'source_snapshot': str(source_snapshot),
  'source_files': {str(p): {'sha256': sha(p)} for p in source_files},
  'artifacts': {name: {'sha256': sha(root / name), 'bytes': (root / name).stat().st_size}
                for name in package_files + ['report.json']},
  'manifest_excludes_itself': True,
  'native_execution': False,
  'gpu_execution': False,
}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({'report_sha256': sha(root/'report.json'), 'manifest_sha256': sha(root/'manifest.json'), 'file_count': len(manifest['artifacts'])}, sort_keys=True))
