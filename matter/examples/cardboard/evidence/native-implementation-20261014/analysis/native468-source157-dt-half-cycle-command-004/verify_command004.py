#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
BASE=Path('/Users/home/cardboard-implementation-20261014/analysis/native468-source157-dt-half-cycle-command-003')
REF=Path('/Users/home/cardboard-implementation-20261014/analysis/native468-executable-preflight-refusal-001/report.json')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((BASE/'native-468-command.json').read_text()); b=json.loads((P/'native-468-command.json').read_text())
assert sha(BASE/'native-468-command.json')=='a866723fba64c9e82c027ca017f9e9fd2eaf2e2023e3cd0af10b6d34105572e0'
assert a[:a.index('--')+1]==b[:b.index('--')+1] and a[a.index('--')+2:]==b[b.index('--')+2:]
assert a.index('--')==b.index('--') and sum(x!=y for x,y in zip(a,b))==1
exe=b[b.index('--')+1]; bindings=json.loads((P/'source-build-bindings.json').read_text())
assert exe=='/Users/home/numi-cardboard-20261006/build-cardboard-provenance-157/numi-matter-cardboard-probe'
assert sha(Path(exe))==bindings['artifacts'][exe]['expected_sha256']
assert sha(Path(bindings['source_build_provenance_path']))==bindings['source_build_provenance_sha256']
assert sha(REF)==json.loads((P/'command-binding.json').read_text())['refusal_sha256']
assert json.loads(REF.read_text())['native_launched'] is False
inputs=json.loads((P/'external-input-bindings.json').read_text())['inputs']
assert len(inputs)==103 and len({Path(x['copied_path']).name for x in inputs})==103
for x in inputs: assert sha(Path(x['copied_path']))==x['sha256']
print('PASS command004 exact executable rebind; source/build/input/refusal bindings valid; no native launch')
