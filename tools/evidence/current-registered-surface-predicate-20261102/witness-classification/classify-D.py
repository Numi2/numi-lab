from pathlib import Path
import sys,json,re,time,numpy as np,hashlib
sys.path.insert(0,'/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001')
import ast,types,math,struct
from fractions import Fraction
from numbers import Integral
owner=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/tools/cardiac_cavity_intersections.py')
tree=ast.parse(owner.read_text());defs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]);ex=types.ModuleType('exact_owner');ex.__dict__.update(Fraction=Fraction,Integral=Integral,math=math,struct=struct,HumanImportError=ValueError);exec(compile(defs,str(owner),'exec'),ex.__dict__)
d=Path(__file__).parent;results=[]
for role in 'D':
 with (d/f'{role}.bin').open('rb') as f:
  n,nf=map(int,np.fromfile(f,dtype='<u8',count=2));p=np.fromfile(f,dtype='<f8',count=n*3).reshape(-1,3);t=np.fromfile(f,dtype='<u4',count=nf*3).reshape(-1,3)
 pairs=[tuple(map(int,m)) for m in re.findall(r'^pair (\d+) (\d+)',(d/'D-component-diagnostic.stdout').read_text(),re.M)]
 used=set(x for pair in pairs for x in pair);points={int(i):tuple(float(v).as_integer_ratio() for v in p[i]) for i in np.unique(t[list(used)])};den=max(q[1] for row in points.values() for q in row);points={i:tuple(v*(den//q) for v,q in row) for i,row in points.items()};records={i:(tuple(points[int(v)] for v in t[i]),None,None,i,tuple(map(int,t[i]))) for i in used}
 start=time.time();r=ex._audit_record_pairs(((records[a],records[b]) for a,b in pairs),same_surface=True);r.update(role=role,seconds=time.time()-start,source_binary_sha256=hashlib.sha256((d/f'{role}.bin').read_bytes()).hexdigest());results.append(r);print(role,json.dumps({k:v for k,v in r.items() if k!='triangle_pairs'}),r['triangle_pairs'][:30],flush=True)
(d/'D-exact-witness-classification.json').write_text(json.dumps(results,indent=2)+'\n')
