#!/usr/bin/env python3
import copy,json,math,struct,sys
from pathlib import Path
BASE=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
INPUT=BASE/'analysis/native547-critical23-sweep-input-002/input.json'
SOURCE=BASE/'analysis/source-local-projection-state-certification-220-003/cardboard_bilinear15_capture_diagnostic_check.mm'
def f32(x): return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def bits(x): return struct.pack('<f',float(x))
def verify(center,endpoint,power,sign):
    basis=2
    eps=math.ldexp(1.0,-power)
    dt=float(center[45])
    if not math.isfinite(dt) or dt<=0 or sign not in (-1,1): return False
    target=float(center[basis])+float(center[9+basis])+sign*eps
    hi=f32(target)
    low=f32((float(center[basis])-float(hi))+float(center[9+basis])+sign*eps)
    rate=f32(float(center[18+basis])+sign*eps/dt)
    if bits(endpoint[basis])!=bits(hi) or bits(endpoint[9+basis])!=bits(low) or bits(endpoint[18+basis])!=bits(rate): return False
    allowed={basis,9+basis,18+basis}
    return all(i in allowed or bits(endpoint[i])==bits(center[i]) for i in range(47))
d=json.loads(INPUT.read_text()); rows=d['rows']; center=rows[256]['packed_input_f32']
eps_rows=[(282,22,1),(283,22,-1),(284,23,1),(285,23,-1),(286,24,1),(287,24,-1),(288,25,1),(289,25,-1)]
tests=[]
for idx,power,sign in eps_rows:
    assert verify(center,rows[idx]['packed_input_f32'],power,sign)
    tests.append({'name':f'actual row {idx} p{power} sign{sign}','pass':True})
    assert not verify(center,rows[idx]['packed_input_f32'],power,-sign)
    tests.append({'name':f'wrong sign rejected row {idx}','pass':True})
mut=copy.deepcopy(rows[284]['packed_input_f32']);mut[0]=f32(mut[0]+2**-23)
assert not verify(center,mut,23,1);tests.append({'name':'legacy wrong F[0] perturbation rejected','pass':True})
mut=copy.deepcopy(rows[284]['packed_input_f32']);mut[2]=center[2]
assert not verify(center,mut,23,1);tests.append({'name':'missing declared F[2] change rejected','pass':True})
mut=copy.deepcopy(rows[284]['packed_input_f32']);mut[20]=f32(mut[20]+1e-5)
assert not verify(center,mut,23,1);tests.append({'name':'incorrect Fdot[2] rejected','pass':True})
mut=copy.deepcopy(rows[284]['packed_input_f32']);mut[46]=f32(mut[46]+1)
assert not verify(center,mut,23,1);tests.append({'name':'unrelated packed input mutation rejected','pass':True})
source=SOURCE.read_text()
assert 'verifySource547EndpointInput(out.selected.rows[0],out.selected.rows[i]' in source
assert 'endpointIt->input[2u]!=centerIt->input[2u]' in source
assert 'endpointIt->input[0u]!=centerIt->input[0u]' not in source
tests.append({'name':'C++ source contains CPU admission and corrected p23 identity predicate','pass':True})
out={'schema':'numi.cardboard.source220-endpoint-identity-cpu-test.v1','input_sha256':__import__('hashlib').sha256(INPUT.read_bytes()).hexdigest(),'tests':tests,'passed':len(tests),'failed':0,'note':'Python contract mirror over exact source547 input; compiled C++ CPU admission still required before native dispatch.'}
(OUT:=SOURCE.parent/'cpu-test.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'passed':len(tests),'failed':0,'input_sha256':out['input_sha256']}))
