#!/usr/bin/env python3
"""Partition the validated common-field PLC using the existing TetGen tool.
No source geometry, cut-ring coordinates, or material acceptance gate changes.
"""
from pathlib import Path
from collections import Counter
import ast,json,hashlib,time,subprocess,sys
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');A=E/'ankle-parent-tet-refinement-026';B=E/'right-commonfield-layer-reconstruction-202';S=B/(sys.argv[1] if len(sys.argv)>1 else 'attempt-003');O=E/(sys.argv[2] if len(sys.argv)>2 else 'right-commonfield-material-volume-203');O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads((S/'source-geometry-receipt-202.json').read_text());assert receipt['source_geometry_pass']
ns={'__name__':'frozen_volume_functions','MAX_TET_VOL':1.5e-7};p=A/'build_common_volume_126.py';tree=ast.parse(p.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[]),str(p),'exec'),ns)
z=np.load(S/'right-commonfield-calibrated-202.npz');X=z['points_world_f32_m'];T=z['tetrahedra'];phi=z['phi'];levels=z['levels']
sourcefaces=np.vstack([z['outer_faces'],z['construction_faces']]);used=np.unique(sourcefaces);index={int(n):i for i,n in enumerate(used)};vertices=X[used].tolist();parts=[];markers=[]
parts.append(np.asarray([[index[int(n)] for n in f] for f in z['outer_faces']],np.int64));markers.extend([1]*len(parts[-1]))
parts.append(np.asarray([[index[int(n)] for n in f] for f in z['construction_faces']],np.int64));markers.extend([2]*len(parts[-1]))
ancestry=[(int(n),int(n),0.) for n in used]
for role,marker in [('I',3),('D',4)]:
 q=np.load(S/f'right-{role}-interface-202.npz');v=q['vertices_world_f32_m'];f=q['faces'];edges=q['source_volume_edge_nodes'];alpha=q['source_edge_endpoint_one_weight'];mp=[]
 for point,(a,b),w in zip(v,edges,alpha):
  if a==b:
   if int(a) not in index:
    index[int(a)]=len(vertices);vertices.append(X[int(a)].tolist());ancestry.append((int(a),int(a),0.))
   j=index[int(a)];assert np.array_equal(np.asarray(vertices[j],np.float32),point)
  else:
   j=len(vertices);vertices.append(point.tolist());ancestry.append((int(a),int(b),float(w)))
  mp.append(j)
 parts.append(np.asarray(mp,np.int64)[f]);markers.extend([marker]*len(f))
V=np.asarray(vertices,np.float32);F=np.vstack(parts);M=np.asarray(markers,np.int64)
if len(np.unique(V,axis=0))!=len(V):raise ValueError('coincident distinct PLC nodes require identity diagnosis; no silent weld')
mid=phi[T].mean(1);anchors=[]
for attr,lo,hi in [(1,0.,levels[0]),(2,levels[0],levels[1]),(3,levels[1],1.)]:
 ids=np.flatnonzero((mid>lo)&(mid<hi));margin=np.minimum(mid[ids]-lo,hi-mid[ids]);row=int(ids[np.argmax(margin)]);anchor=X[T[row]].astype(float).mean(0);anchors.append((attr,anchor,row,float(mid[row])))
poly=O/'right-material-plc.poly'
with poly.open('x') as f:
 f.write(f'{len(V)} 3 0 1\n')
 for i,v in enumerate(V,1):f.write(f'{i} {v[0]:.17g} {v[1]:.17g} {v[2]:.17g} 1\n')
 f.write(f'{len(F)} 1\n')
 for tri,m in zip(F,M):f.write(f'1 0 {m}\n3 {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')
 f.write('0\n3\n')
 for attr,point,row,field in anchors:f.write(f'{attr} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g} {attr} 1.5e-7\n')
np.savez_compressed(O/'plc-source-ancestry.npz',vertices_world_f32_m=V,faces=F,facet_markers=M,source_volume_edge_nodes=np.asarray(ancestry)[:,:2].astype(np.int64),source_endpoint_one_weight=np.asarray(ancestry)[:,2])
exe=E/'arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen';cmd=[str(exe),'-pYq1.2/5AS50000',str(poly)];start=time.monotonic();report={'physical_admission':False,'source_receipt_sha256':sha(S/'source-geometry-receipt-202.json'),'command':cmd,'tetgen_sha256':sha(exe),'script_sha256':sha(__file__),'plc_nodes':len(V),'plc_faces':len(F),'marker_counts':{str(m):int((M==m).sum()) for m in np.unique(M)},'anchors':[{'attribute':a,'point':v.tolist(),'source_tet':t,'P1_field':p} for a,v,t,p in anchors]}
try:
 with (O/'tetgen.log').open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=90)
 report['exit_code']=result.returncode
 if result.returncode:raise RuntimeError('TetGen rejected PLC')
 Vnew,Tnew=ns['parse_tetgen'](O/'right-material-plc');ele=Path(str(O/'right-material-plc')+'.1.ele');lines=[s.split('#')[0].strip() for s in ele.read_text().splitlines() if s.split('#')[0].strip()];assert int(lines[0].split()[2])==1;attrs=np.array([int(float(s.split()[5])) for s in lines[1:]],np.int64)
 vf,q=ns['tet_metrics'](Vnew.astype(np.float32).astype(float),Tnew); report.update(nodes=len(Vnew),tetrahedra=len(Tnew),nonpositive_F32_cells=int((vf<=0).sum()),below_quality_floor=int((q<1e-5).sum()),minimum_normalized_determinant=float(q.min()),material_volumes_m3={str(a):float(vf[attrs==a].sum()) for a in np.unique(attrs)})
 active=np.isin(attrs,[1,2]); report['material_quality']={str(a):{'cells':int((attrs==a).sum()),'nonpositive':int(((attrs==a)&(vf<=0)).sum()),'below_floor':int(((attrs==a)&(q<1e-5)).sum()),'minimum_q':float(q[attrs==a].min())} for a in np.unique(attrs)}
 report['construction_material_volume_relative_error']={str(a):float((vf[attrs==a].sum()-target)/target) for a,target in [(1,.0003491),(2,.0011945)]}
 report['geometric_volume_pass']=bool(np.all(vf[active]>0) and np.all(q[active]>=1e-5) and max(abs(v) for v in report['construction_material_volume_relative_error'].values())<=.01)
 selected=np.flatnonzero(active);used=np.unique(Tnew[active]);mp=np.full(len(Vnew),-1,np.int64);mp[used]=np.arange(len(used));np.savez_compressed(O/'right-skin-fat-volume-unadmitted.npz',points_world_f64_m=Vnew[used],tetrahedra=mp[Tnew[active]],material_codes=attrs[active],source_material_plc_node_ids=used,source_material_plc_tet_ids=selected)
 np.savez_compressed(O/'right-material-volume-203.npz',points_world_f64_m=Vnew,tetrahedra=Tnew,material_codes=attrs,quality=q,signed_volume_m3=vf)
except Exception as exc:report['failure']=str(exc);report['geometric_volume_pass']=False
report['wall_seconds']=time.monotonic()-start;report['remaining']=['exact material interface incidence and boundary ownership','old/new conservative overlap ancestry and regional mass mapping','retained-pose/native mechanical qualification']
(O/'report-203.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if not report['geometric_volume_pass']:raise SystemExit('material volume gate failed')
