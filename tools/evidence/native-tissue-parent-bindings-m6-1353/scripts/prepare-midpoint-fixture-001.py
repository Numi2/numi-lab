from pathlib import Path
import json,sys,importlib.util,hashlib,numpy as np,struct
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
O=A/'parent-bindings-midpoint-regression-1353-001';O.mkdir(exist_ok=False);(O/'assets').mkdir()
sp=importlib.util.spec_from_file_location('h',A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py');h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=A/'temporalis163-native-m6-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
t=h.load_tissue(T);row=h.row_data(t,132);p=row['positions'];f=row['faces'];local=row['local'];weights=row['weights']
receipt=json.loads((A/'temporalis163-native-m6-001/baseline/native-run/accepted-geometry/step-4767.receipt.json').read_text())
pm={int(x['body_index']):(np.asarray(x['position_m'],np.float32),np.asarray(x['quaternion_xyzw'],np.float32)) for x in receipt['accepted_body_poses']}
edges=sorted({tuple(sorted((int(a),int(b)))) for tri in f for a,b in zip(tri,np.roll(tri,-1))})
eligible=[];newlocal=[];newweights=[]
for a,b in edges:
 d={}
 for v in (a,b):
  for k,w in zip(local[v],weights[v]):
   if w>0:d[int(k)]=d.get(int(k),0)+float(w)*.5
 if len(d)>4:continue
 pairs=sorted(d.items());eligible.append((a,b));newlocal.append([k for k,w in pairs]+[4294967295]*(4-len(pairs)));newweights.append([w for k,w in pairs]+[0]*(4-len(pairs)))
edges=np.array(eligible);mid=(p[edges[:,0]].astype(float)+p[edges[:,1]].astype(float))*.5
newlocal=np.array(newlocal,np.uint32);newweights=np.array(newweights,np.float32)
world=h.forward(p,local,weights,row,t,pm);averaged=h.forward(mid.astype(np.float32),newlocal,newweights,row,t,pm);expected=world[edges].astype(float).mean(1)
errors=np.linalg.norm(averaged.astype(float)-expected,axis=1);i=int(errors.argmax());a,b=map(int,edges[i])
assert errors[i]>1e-5
raw=bytearray(t['raw']);offset=t['vo']+(row['fv']+a)*56
struct.pack_into('<3f',raw,offset,*mid[i].astype(np.float32));struct.pack_into('<4I',raw,offset+24,*newlocal[i]);struct.pack_into('<4f',raw,offset+40,*newweights[i])
target=O/'assets'/T.name;target.write_bytes(raw);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads(T.with_suffix('.manifest.json').read_text());manifest['payload']['sha256']=sha(target)
manifest['status']='native_parent_interpolation_regression_fixture_not_anatomically_admitted'
manifest['source']['parent_interpolation_regression']=dict(source_payload=str(T),source_payload_sha256=sha(T),changed_row=132,changed_vertex=a,source_edge=[a,b],coefficient=[.5,.5],purpose='Witness noncommutation of averaged bindings and source-parent interpolation in the integrated native scene; not a repaired anatomical candidate.')
target.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
report=dict(source=str(T),source_sha256=sha(T),child=str(target),child_sha256=sha(target),stable_id=132,vertex=a,other_vertex=b,eligible_edges=len(edges),helper_noncommutation_m=float(errors[i]),rest_local_displacement_m=float(np.linalg.norm(mid[i]-p[a])),admitted=False)
(O/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
