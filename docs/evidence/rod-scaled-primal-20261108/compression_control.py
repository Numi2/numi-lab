#!/usr/bin/env python3
import json,struct,sys
from pathlib import Path
import numpy as np
b=Path('docs/evidence/rod-scaled-primal-20261108')
if sys.argv[1]=='generate':
 base=(b/'metal-input.bin').read_bytes()[:576];compressed=bytearray(base)
 struct.pack_into('<f',compressed,16,.025);struct.pack_into('<f',compressed,32,.05)
 struct.pack_into('<f',compressed,272+32,.05)
 (b/'metal-compression-input.bin').write_bytes(base+compressed)
elif sys.argv[1]=='analyze':
 data=(b/'metal-compression-output.bin').read_bytes();inputs=(b/'metal-compression-input.bin').read_bytes()
 if len(data)!=1120:raise ValueError('wrong output size')
 control=data[:560];compressed=data[560:];status=struct.unpack_from('<f',compressed,552)[0]
 values=np.frombuffer(inputs[576:1152],dtype='<f4')
 nodes=values[:68].reshape((17,4))[:3];target=values[68:136].reshape((17,4))[:3];params=values[136:140]
 x=nodes[:,:3].astype(np.float64);mass=nodes[:,3].astype(np.float64);rest=float(params[0]);k=float(params[1]);h=float(params[2])
 ns={};exec((b/'vector_full_newton.py').read_text().rsplit('\nspec=[',1)[0],ns)
 _,g,diag,_=ns['evaluate'](x,x,[],rest,k,mass,True)
 eigen=np.linalg.eigvalsh(diag[1]);norm=float(np.linalg.norm(g[1]))
 untouched=compressed[:544]==bytes([0xa5])*544
 control_status=struct.unpack_from('<f',control,552)[0]
 passed=bool(control_status==0 and float(eigen[0])<0 and norm<1e-5 and status!=0 and untouched)
 result=dict(control_status=control_status,compressed_status=status,output_unchanged=untouched,
     transverse_eigenvalues_N_per_m=list(map(float,eigen)),free_node_force_N=norm,passed=passed)
 print(json.dumps(result,indent=2,sort_keys=True));sys.exit(0 if passed else 1)
else:raise ValueError('generate or analyze')
