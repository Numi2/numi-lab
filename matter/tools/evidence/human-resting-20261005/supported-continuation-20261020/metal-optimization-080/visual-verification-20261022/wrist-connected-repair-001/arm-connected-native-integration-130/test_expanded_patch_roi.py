import ast,copy,hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).parent
src=root/'audit_existing_patch_v4.py'
tree=ast.parse(src.read_text())
names={'same_layout','changed_rows'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(nodes)==2
base={'nv':8,'nb':2,'fingerprint':7,'source_sha256':b'a'*32,'binds':np.zeros(2,dtype='<f4'),'inds':np.array([[0,1,2],[2,3,4],[5,6,7]],dtype='<u4'),'weights':np.tile([1.,0.],(8,1)),'verts':np.zeros((8,14))}
env={'np':np,'base':base}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),env)
same=env['same_layout'];changed=env['changed_rows']
assert same(base)
for key,value in [('nv',9),('fingerprint',8),('source_sha256',b'b'*32)]:
 q=copy.deepcopy(base);q[key]=value;assert not same(q)
q=copy.deepcopy(base);q['inds'][0]=[1,0,2];assert not same(q)
q=copy.deepcopy(base);q['binds'][0]=1;assert not same(q)
assert len(changed(base))==0
q=copy.deepcopy(base);q['verts'][2,0]=.001
assert changed(q).tolist()==[2]
r=copy.deepcopy(base);r['weights'][7]=[.5,.5]
assert changed(r).tolist()==[7]
F=base['inds'].astype(np.int64);a={'face_origin_codes':np.array([1,0,0])}
# Run the actual source's ROI assignments, with the two precomputed changed-row sets.
selected=[]
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'changed','roi_changed','affected','incident','patch','other'} for t in n.targets):selected.append(n)
assert len(selected)==6
code=compile(ast.Module(body=selected,type_ignores=[]),str(src),'exec')
def roi(actual,requested):
 e=dict(env,p=actual,rp=requested,F=F,a=a)
 exec(code,e)
 return e['patch'].tolist(),e['other'].tolist()
assert roi(q,q)==([0,1],[2])
assert roi(base,q)==roi(q,q)
assert roi(r,r)==([0,2],[1])
assert roi(q,r)==([0,1,2],[])
assert roi(base,base)==([0],[1,2])
print(json.dumps({'pass':True,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'covered':['changed original position','changed original weights','matched baseline ROI','union of candidate and requested changes','no-change ancestry patch','registered layout mismatch']},indent=2))
