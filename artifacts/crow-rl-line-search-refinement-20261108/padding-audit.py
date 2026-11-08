from pathlib import Path
import hashlib,json,struct
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-refinement-20261108';B=R/'build/crow-line-search-refinement-build-01'
inputs=json.loads((A/'inputs.json').read_text());cases=[]
for name,f in inputs.items():
 raw=(R/f['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==f['sha256']
 d=struct.unpack_from('<16I8f',raw,32);nv,rows,nb=d[2:5];covered=np.zeros(rows,bool)
 for i in range(nb):
  at,n,kind,flags=struct.unpack_from('<4I',raw,256+i*128);assert n>0 and not covered[at:at+n].any();covered[at:at+n]=True
 offset=256+128*nb+4*nv*nv
 J=np.frombuffer(raw,dtype='<f4',count=rows*nv,offset=offset).reshape(rows,nv)
 bias=np.frombuffer(raw,dtype='<f4',count=rows,offset=offset+4*rows*nv)
 assert np.all(J[~covered]==0) and np.all(bias[~covered]==0)
 output=B/'observations'/(name+'-candidate-0');status=struct.unpack('<12I12f',(output/'status.bin').read_bytes())
 case=dict(name=name,unused_rows=int(np.sum(~covered)),zero_unused_input_operator_and_bias=True,status=status[0])
 if status[0]==0:
  p=np.fromfile(output/'impulse.f32',dtype='<f4')[:rows];unused=p[~covered];assert np.all(unused==0)
  case.update(unused_output_impulses_zero=True,unused_output_bit_patterns=sorted(int(v) for v in np.unique(unused.view('<u4'))))
 else:
  assert name=='substep7' and status[0]==7
  case['rejected_output_poison_preserved']=(output/'impulse.f32').read_bytes()==b'\xCD'*(4*max(nv,rows));assert case['rejected_output_poison_preserved']
 cases.append(case)
report=dict(schema='crow-line-search-padding-audit-v1',cases=cases,passed=True,scope='Six actual retained operators and outputs. No new native commands, no proof about uncaptured future problems.',physical_or_policy_changed=False)
(A/'padding-audit-result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
