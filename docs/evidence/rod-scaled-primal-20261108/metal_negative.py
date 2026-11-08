#!/usr/bin/env python3
import json,struct,sys
from pathlib import Path
b=Path('docs/evidence/rod-scaled-primal-20261108')
if sys.argv[1]=='generate':
    base=(b/'metal-input.bin').read_bytes()[:576]
    cases=[bytearray(base) for _ in range(5)]
    struct.pack_into('<I',cases[1],560,1)
    cases[2][16:28]=cases[2][0:12]
    struct.pack_into('<f',cases[3],16,float('nan'))
    struct.pack_into('<f',cases[4],28,-1.)
    (b/'metal-negative-input.bin').write_bytes(b''.join(cases))
elif sys.argv[1]=='analyze':
    data=(b/'metal-negative-output.bin').read_bytes(); baseline=(b/'metal-output.bin').read_bytes()
    if len(data)!=5*560:raise ValueError('wrong output size')
    rows=[]
    for i,name in enumerate(('control','invalid_count','degenerate_edge','nonfinite_position','nonpositive_mass')):
        sample=data[i*560:(i+1)*560]
        status=struct.unpack_from('<f',sample,552)[0]
        body=sample[:544]
        valid=(status==0 and body==baseline[:544]) if i==0 else (status!=0 and body==bytes([0xa5])*544)
        rows.append(dict(case=name,status=status,output_unchanged=(body==bytes([0xa5])*544),passed=valid))
    result=dict(cases=rows,all_pass=all(r['passed'] for r in rows))
    print(json.dumps(result,indent=2,sort_keys=True));sys.exit(0 if result['all_pass'] else 1)
else:raise ValueError('expected generate or analyze')
