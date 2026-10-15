from pathlib import Path
import hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-repeated-cauchy-fallback-137'
Old=B/'snapshots/source-cauchy-positive-mass-admission-135'
O=B/'analysis/source137-root-integration-preflight-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((S/'frozen-source.json').read_text())['source_sha256']
old=json.loads((Old/'frozen-source.json').read_text())['source_sha256']
expected={'matter/src/metal/common.metalinc','matter/src/metal/fem.metalinc','matter/src/runtime.mm',
    'matter/tools/cardboard_coupled_free_box_operator_check.mm','matter/tools/cardboard_gpu_check.mm',
    'matter/tools/fem_material_frame_check.mm','matter/tools/fem_reference_check.mm','matter/tools/fem_regional_material_check.mm'}
different={p for p,h in f.items() if not(R/p).is_file() or sha(R/p)!=h}
assert different==expected,different
assert json.loads((R/'build-cardboard-provenance-137/source-build.json').read_text())['success']
assert json.loads((B/'cpu-checks-137/receipt.json').read_text())['exit_code']==0
suite=json.loads((B/'native/source137-direct-operator-abi-439/receipt.json').read_text())
assert len(suite['results'])==5 and all(v['exit_code']==0 for v in suite['results'].values())
O.mkdir();rows=[]
for p in sorted(different):
    assert sha(R/p)==old[p] and sha(S/p)==f[p],p
    rows.append(dict(path=p,before_sha256=sha(R/p),candidate_sha256=f[p]))
(O/'preflight.json').write_text(json.dumps(dict(files=rows,
    complete_selected_sources=len(f),build_sha256=sha(R/'build-cardboard-provenance-137/source-build.json'),
    cpu_receipt_sha256=sha(B/'cpu-checks-137/receipt.json'),
    abi_receipt_sha256=sha(B/'native/source137-direct-operator-abi-439/receipt.json')),indent=2)+'\n')
for p in sorted(different):shutil.copy2(S/p,R/p)
for p,h in f.items():assert sha(R/p)==h,p
(O/'integration.json').write_text(json.dumps(dict(files=rows,source_snapshot=str(S),
    complete_selected_sources_verified=len(f),
    scope='Exact source137 selected source integration after69-target build,74CPU checks and5native directoperator regressions. Default fallback/cache policies remain off. Repeated-fallback experiment442 is in progress; source136 opt-in cache full coupon437 rejected initialization and remains an explicitly experimental path.',
    known_trace_label_erratum='Two source137 observer strings say64 while selected all-Newton actual cap is512; source139 snapshot corrects these labels and the opt-in cache admission, awaiting separate native parity.'),indent=2)+'\n')
(O/'integrate.py').write_bytes(Path(__file__).read_bytes())
print('root137',len(rows),'changed paths;',len(f),'selected sources verified')
