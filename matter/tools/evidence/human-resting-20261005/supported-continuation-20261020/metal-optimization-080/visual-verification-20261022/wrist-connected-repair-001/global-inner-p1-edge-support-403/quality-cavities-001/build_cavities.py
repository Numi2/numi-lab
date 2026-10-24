from pathlib import Path
import ast,collections,hashlib,json,subprocess,time,numpy as np
from scipy.spatial import ConvexHull,QhullError
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=E/'global-inner-p1-edge-support-403'
IN=D/'build-002/candidate.npz'
OUT=D/'quality-cavities-001'
BASE=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse'
src=BASE/'build_014-replay.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
wanted={'dets','quality','volume','face_rows','edge_hist','read_nodes','read_ele'}
tree=ast.parse(src.read_text()); ns=dict(np=np,collections=collections,ConvexHull=ConvexHull,QhullError=QhullError)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[]),str(src),'exec'),ns)
globals().update({k:ns[k] for k in wanted})
def main():
 start=time.monotonic()
 OUT.mkdir(exist_ok=False)
 z=np.load(IN,allow_pickle=False);X=z['points_world_m'];T=z['tetrahedra'].astype(np.int64);A=z['region_attributes'].astype(np.int64)
 assert X.dtype==np.float32 and np.all(dets(X,T)>0)
 bad=np.flatnonzero(quality(X,T)<1e-5)
 assert 0<len(bad)<=50
 seednodes=np.unique(T[bad]);sel=set(map(int,np.flatnonzero(np.isin(T,seednodes).any(1))))
 closure=[]
 for iteration in range(4):
  bf,_=face_rows(X,T,sel);problems=[e for e,n in edge_hist(bf).items() if n!=2]
  if not problems:break
  for a,b in problems:sel.update(map(int,np.flatnonzero((T==a).any(1)&(T==b).any(1))))
  closure.append({'iteration':iteration,'problem_edges':len(problems),'selected_tets':len(sel)})
 else:raise RuntimeError('cavity closure failed')
 assert len(sel)<=2500,'cavity work budget'
 rows=np.asarray(sorted(sel),np.int64);LT=T[rows];nodes=np.unique(LT)
 bf,_=face_rows(X,T,sel);assert all(n==2 for n in edge_hist(bf).values())
 # Retain EVERY input cavity point, including free support midpoints/centroids.
 localnodes=nodes; lid={int(g):i+1 for i,g in enumerate(localnodes)}
 assert len({X[g].tobytes() for g in localnodes})==len(localnodes)
 adj=collections.defaultdict(list)
 for ti in rows:
  for k in range(4):adj[tuple(sorted(np.delete(T[ti],k)))].append(int(ti))
 neighbors=collections.defaultdict(set)
 for ids in adj.values():
  if len(ids)==2:
   neighbors[ids[0]].add(ids[1]);neighbors[ids[1]].add(ids[0])
 unseen=set(sel);components=[]
 while unseen:
  i=min(unseen);unseen.remove(i);stack=[i];comp=[]
  while stack:
   a=stack.pop();comp.append(a)
   for b in neighbors[a]:
    if b in unseen:unseen.remove(b);stack.append(b)
  components.append(sorted(comp))
 TETGEN=E/'arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen'
 assert TETGEN.exists()
 poly=OUT/'cavities.poly';maxvol=max(volume(X,LT)/max(500,len(LT)*2),1e-12)
 with poly.open('w') as f:
  f.write(f'{len(localnodes)} 3 0 1\n')
  for i,g in enumerate(localnodes,1):f.write(f'{i} '+' '.join(format(float(v),'.17g') for v in X[g])+' 1\n')
  f.write(f'{len(bf)} 1\n')
  for tri in bf:f.write('1 0\n3 '+' '.join(str(lid[int(v)]) for v in tri)+'\n')
  f.write(f'0\n{len(components)}\n')
  for k,c in enumerate(components,1):
   assert len(set(map(int,A[c])))==1,'material interface cannot be erased'
   seed=X[T[c[0]]].astype(float).mean(0)
   f.write(f'{k} '+' '.join(format(float(v),'.17g') for v in seed)+f' {int(A[c[0]])} {maxvol:.17g}\n')
 report={'input_path':str(IN),'input_sha256':sha(IN),'owner014_source_sha256':sha(src),'tetgen_sha256':sha(TETGEN),'bad_rows':bad.tolist(),'selected_rows':rows.tolist(),'selected_tets':len(rows),'input_cavity_nodes':len(localnodes),'components':[len(c) for c in components],'closure':closure,'all_cavity_input_points_retained_in_PLC':True,'native_admission':False}
 (OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
 # Bounded first attempt: no added Steiner coordinates. No original point is omitted.
 cmd=[str(TETGEN),'-pYq1.2/10AS0',str(poly)]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
 (OUT/'tetgen.stdout.txt').write_text(p.stdout);(OUT/'tetgen.stderr.txt').write_text(p.stderr)
 report.update(command=cmd,tetgen_exit=p.returncode)
 if p.returncode:raise RuntimeError('TetGen failed')
 local=read_nodes(OUT/'cavities.1.node');tet,attr=read_ele(OUT/'cavities.1.ele')
 order=sorted(local);idrow={v:i for i,v in enumerate(order)}
 LX=np.asarray([local[i] for i in order],np.float32)
 TT=np.asarray([[idrow[int(v)] for v in f] for f in tet],np.int64)
 neg=np.flatnonzero(dets(LX,TT)<0);TT[neg,0],TT[neg,1]=TT[neg,1].copy(),TT[neg,0].copy()
 lookup={X[int(g)].tobytes():int(g) for g in localnodes};mapping=[]
 for point in LX:
  assert point.tobytes() in lookup,'unexpected Steiner point in S0 trial'
  mapping.append(lookup[point.tobytes()])
 mapping=np.asarray(mapping,np.int64);NTlocal=mapping[TT]
 kept=np.ones(len(T),bool);kept[rows]=False
 NT=np.vstack([T[kept],NTlocal]);NA=np.r_[A[kept],attr]
 outbf,_=face_rows(LX,TT,set(range(len(TT))))
 expected={tuple(sorted(map(int,f))) for f in bf}
 actual={tuple(sorted(map(int,mapping[list(f)]))) for f in outbf}
 support_start=49951
 required=np.arange(support_start,len(X))
 refs=np.unique(NT);missing=required[~np.isin(required,refs)]
 O=np.zeros(len(X),bool);O[z['physical_outer_node_ids']]=True
 pairs=np.asarray([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
 edges=np.unique(np.sort(NT[:,pairs].reshape(-1,2),axis=1),axis=0)
 F=z['physical_shell_faces'];fe=np.asarray([[0,1],[0,2],[1,2]])
 bedges=np.unique(np.sort(F[:,fe].reshape(-1,2),axis=1),axis=0)
 enc=lambda e:e[:,0]*len(X)+e[:,1]
 badedges=edges[np.all(O[edges],axis=1)&~np.isin(enc(edges),enc(bedges))]
 allfaces=np.concatenate([NT[:,[1,2,3]],NT[:,[0,3,2]],NT[:,[0,1,3]],NT[:,[0,2,1]]])
 u,c=np.unique(np.sort(allfaces,axis=1),axis=0,return_counts=True)
 badfaces=u[(c==2)&np.all(O[u],axis=1)]
 dq=quality(X,NT);dd=dets(X,NT)
 report.update(output_tets=len(NT),output_nodes=len(X),nonpositive_tets=int((dd<=0).sum()),minimum_normalized_determinant=float(dq.min()),quality_below_1e5=int((dq<1e-5).sum()),local_boundary_keys_exact=expected==actual,global_volume_relative_change=float((volume(X,NT)-volume(X,T))/volume(X,T)),support_nodes_missing=missing.tolist(),remaining_internal_all_O_edges=len(badedges),remaining_internal_all_O_faces=len(badfaces),remaining_all_O_tets=int(np.all(O[NT],axis=1).sum()),elapsed_s=time.monotonic()-start)
 report['geometry_support_gate_pass']=bool(np.all(dd>0) and dq.min()>=1e-5 and expected==actual and not len(missing) and not len(badedges) and not len(badfaces) and not report['remaining_all_O_tets'] and abs(report['global_volume_relative_change'])<2e-6)
 report['conservative_material_transfer_pending']=True
 np.savez_compressed(OUT/'candidate-geometry-only.npz',points_world_m=X,tetrahedra=NT,region_attributes=NA,physical_outer_node_ids=z['physical_outer_node_ids'],physical_outer_faces=z['physical_outer_faces'],physical_shell_faces=z['physical_shell_faces'],construction_closure_faces=z['construction_closure_faces'],replaced_403_tet_rows=rows,new_local_tets=NTlocal,unchanged_403_tet_rows=np.flatnonzero(kept))
 report['candidate_sha256']=sha(OUT/'candidate-geometry-only.npz')
 (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['selected_rows','bad_rows']}),flush=True)
if __name__=='__main__':main()

