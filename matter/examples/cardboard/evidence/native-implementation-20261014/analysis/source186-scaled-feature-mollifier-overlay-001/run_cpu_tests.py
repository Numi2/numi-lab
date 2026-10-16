#!/usr/bin/env python3
import hashlib,json,subprocess,tempfile
from pathlib import Path
D=Path(__file__).resolve().parent
BASE=Path('/Users/home/cardboard-implementation-20261014/snapshots/source-portable-paper-reference-185/matter/src/metal/contact.metalinc')
CAND=D/'candidate/matter/src/metal/contact.metalinc'
def extract(path):
 text=path.read_text(); start=text.index('inline float ipcFeatureMollifierDifferential('); brace=text.index('{',start); depth=0
 for i in range(brace,len(text)):
  if text[i]=='{': depth+=1
  elif text[i]=='}':
   depth-=1
   if depth==0: return text[start:i+1]
 raise RuntimeError('unbalanced target function')
def sha(b): return hashlib.sha256(b).hexdigest()
cf=extract(CAND); bf=extract(BASE)
assert 'kGeometryFloor' not in cf and 'cross(firstUnit, secondUnit)' in cf
assert 'relativeThreshold = 1.0e-3f' in cf and 'kGeometryFloor' in bf
with tempfile.TemporaryDirectory(prefix='source186-mollifier-') as tmp:
 td=Path(tmp)
 (td/'production_mollifier_extracted.hpp').write_text('// Candidate function extracted from source; only Metal reference qualifiers removed.\n'+cf.replace('thread ','')+'\n')
 (td/'baseline_mollifier_extracted.hpp').write_text('// Baseline function extracted from frozen source185; only Metal reference qualifiers removed.\n'+bf.replace('thread ','')+'\n')
 outputs=[]
 for source,name,header in [(D/'tests/test_driver.cpp','candidate_scale_fd','production_mollifier_extracted.hpp'),(D/'tests/test_legacy_scale_failure.cpp','baseline_unit_dependence','baseline_mollifier_extracted.hpp')]:
  exe=td/name
  cmd=['clang++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(td),str(source),'-o',str(exe)]
  cp=subprocess.run(cmd,text=True,capture_output=True)
  if cp.returncode: raise SystemExit('compile failed: '+name+'\n'+cp.stdout+cp.stderr)
  rp=subprocess.run([str(exe)],text=True,capture_output=True)
  outputs.append({'name':name,'command':cmd,'returncode':rp.returncode,'stdout':rp.stdout,'stderr':rp.stderr,'binary_sha256':sha(exe.read_bytes())})
  if rp.returncode: raise SystemExit('test failed: '+name+'\n'+rp.stdout+rp.stderr)
 result={'schema':'numi.cardboard.source186_mollifier_cpu_test.v1','production_function_sha256':sha(cf.encode()),'baseline_function_sha256':sha(bf.encode()),'results':outputs}
 (D/'cpu-test-receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps(result,sort_keys=True))
