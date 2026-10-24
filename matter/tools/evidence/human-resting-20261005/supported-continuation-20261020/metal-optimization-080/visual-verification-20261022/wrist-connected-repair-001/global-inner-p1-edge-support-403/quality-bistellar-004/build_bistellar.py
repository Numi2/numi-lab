from pathlib import Path
import ast,collections,hashlib,json,time,itertools,numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'global-inner-p1-edge-support-403';IN=D/'quality-bistellar-003/candidate-geometry-only.npz';OUT=D/'quality-bistellar-004'
src=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/build_014-replay.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
wanted={'dets','quality','volume','face_rows'}
ns=dict(np=np,collections=collections); tree=ast.parse(src.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[]),str(src),'exec'),ns)
globals().update({k:ns[k] for k in wanted})
def boundary(t):
 f=np.concatenate([t[:,[1,2,3]],t[:,[0,3,2]],t[:,[0,1,3]],t[:,[0,2,1]]])
 u,ix,c=np.unique(np.sort(f,axis=1),axis=0,return_index=True,return_counts=True)
 if np.any(c>2):return None
 o=f[ix[c==1]];m=np.argmin(o,axis=1);ca=np.stack([o[np.arange(len(o)),(m+i)%3] for i in range(3)],axis=1)
 return sorted(map(tuple,ca))
def orient(x,t):
 t=np.asarray(t,np.int64);d=dets(x,t)
 if np.any(d==0):return None
 n=d<0;t[n,0],t[n,1]=t[n,1].copy(),t[n,0].copy()
 return t
def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False)
 z=np.load(IN,allow_pickle=False);X=z['points_world_m'];T=z['tetrahedra'];A=z['region_attributes']
 active={i:tuple(map(int,t)) for i,t in enumerate(T)}
 attr={i:int(a) for i,a in enumerate(A)}
 lineage={i:{i} for i in range(len(T))}
 fac=collections.defaultdict(set);edg=collections.defaultdict(set)
 def keys(t):return [tuple(sorted(f)) for f in itertools.combinations(t,3)],[tuple(sorted(e)) for e in itertools.combinations(t,2)]
 def add(i,t):
  f,e=keys(t)
  for k in f:fac[k].add(i)
  for k in e:edg[k].add(i)
 def drop(i,t):
  f,e=keys(t)
  for k in f:fac[k].discard(i)
  for k in e:edg[k].discard(i)
 for i,t in active.items():add(i,t)
 O=np.zeros(len(X),bool);O[z['physical_outer_node_ids']]=True
 shell=z['physical_shell_faces']
 boundaryedges={tuple(sorted(e)) for f in shell for e in itertools.combinations(f,2)}
 boundaryfaces={tuple(sorted(f)) for f in shell}
 def support_ok(ts):
  for t in ts:
   if np.all(O[t]):return False
   for e in itertools.combinations(t,2):
    if O[e[0]] and O[e[1]] and tuple(sorted(e)) not in boundaryedges:return False
   for f in itertools.combinations(t,3):
    if np.all(O[list(f)]) and tuple(sorted(f)) not in boundaryfaces:return False
  return True
 log=[];attempted=0
 for iteration in range(24):
  ids=np.asarray(sorted(active),np.int64);ts=np.asarray([active[i] for i in ids],np.int64);qs=quality(X,ts)
  bad=ids[qs<1e-5]
  if not len(bad):break
  best=None;seen=set()
  for badid in bad:
   fkeys,ekeys=keys(active[int(badid)])
   # Include every two-cell face and every three-cell edge adjacent to a bad cell.
   cavities=[]
   for f in fkeys:
    rows=tuple(sorted(fac[f]))
    if len(rows)==2:
     v=set(active[rows[0]])|set(active[rows[1]]);opp=sorted(v-set(f))
     if len(v)==5 and len(opp)==2:
      new=[[opp[0],opp[1],f[j],f[(j+1)%3]] for j in range(3)]
      cavities.append((rows,new,'2-3'))
   for e in ekeys:
    rows=tuple(sorted(edg[e]))
    if len(rows)==3:
     v=set().union(*(set(active[i]) for i in rows));ring=sorted(v-set(e))
     if len(v)==5 and len(ring)==3:
      cavities.append((rows,[[e[0],*ring],[e[1],*ring]],'3-2'))
    elif len(rows)==4:
     v=set().union(*(set(active[i]) for i in rows));ring=sorted(v-set(e))
     if len(v)==6 and len(ring)==4:
      # Ring adjacency identifies two alternative octahedron diagonals.
      rg=set()
      for i in rows:
       uv=tuple(sorted(set(active[i])-set(e)))
       if len(uv)==2:rg.add(uv)
      for u,vv in itertools.combinations(ring,2):
       if (u,vv) in rg:continue
       oth=sorted(set(ring)-{u,vv})
       new=[[u,vv,e[0],oth[0]],[u,vv,oth[0],e[1]],[u,vv,e[1],oth[1]],[u,vv,oth[1],e[0]]]
       cavities.append((rows,new,'4-4'))
    elif 5<=len(rows)<=8:
     verts=set().union(*(set(active[i]) for i in rows));ring=sorted(verts-set(e))
     if len(ring)!=len(rows):continue
     rg=collections.defaultdict(set)
     for i in rows:
      uv=sorted(set(active[i])-set(e))
      if len(uv)==2:rg[uv[0]].add(uv[1]);rg[uv[1]].add(uv[0])
     if any(len(rg[v])!=2 for v in ring):continue
     cycle=[min(ring)];previous=None;current=cycle[0]
     for step in range(len(ring)-1):
      options=sorted(rg[current]-({previous} if previous is not None else set()))
      nxt=options[0]
      if nxt==cycle[0] or nxt in cycle:break
      cycle.append(nxt);previous,current=current,nxt
     if len(cycle)!=len(ring) or cycle[0] not in rg[cycle[-1]]:continue
     def triangulate(poly):
      if len(poly)==3:return [[tuple(poly)]]
      out=[]
      for k in range(1,len(poly)-1):
       left=triangulate(poly[:k+1]) if k>=2 else [[]]
       right=triangulate(poly[k:]) if len(poly)-k>=3 else [[]]
       for ll in left:
        for rr in right:out.append(ll+rr+[(poly[0],poly[k],poly[-1])])
      return out
     for triangles in triangulate(cycle):
      new=[[e[0],*tri] for tri in triangles]+[[e[1],*tri] for tri in triangles]
      cavities.append((rows,new,f'{len(rows)}-{len(new)}'))
   for rows,new,kind in cavities:
    key=(rows,tuple(sorted(tuple(sorted(t)) for t in new)))
    if key in seen:continue
    seen.add(key);attempted+=1
    if len({attr[i] for i in rows})!=1:continue
    old=np.asarray([active[i] for i in rows],np.int64);new=orient(X,new)
    if new is None or boundary(old)!=boundary(new) or not support_ok(new):continue
    if set(old.ravel())!=set(new.ravel()):continue
    oq=quality(X,old);nq=quality(X,new)
    if np.any(nq<=0):continue
    # Strict lexicographic improvement; preserve a passing cavity's threshold.
    oldscore=(int((oq<1e-5).sum()),-float(oq.min()))
    newscore=(int((nq<1e-5).sum()),-float(nq.min()))
    if not newscore<oldscore:continue
    if abs(volume(X,new)-volume(X,old))>max(volume(X,old)*1e-8,1e-18):continue
    score=(newscore[0]-oldscore[0],newscore[1]-oldscore[1])
    if best is None or score<best[0]:best=(score,rows,new,kind,oq,nq)
  if best is None:break
  _,rows,new,kind,oq,nq=best;newid=max(active)+1;ancestry=set().union(*(lineage[i] for i in rows));mat=attr[rows[0]]
  for i in rows:drop(i,active[i]);del active[i];del attr[i];del lineage[i]
  for k,t in enumerate(new):
   i=newid+k;active[i]=tuple(map(int,t));attr[i]=mat;lineage[i]=ancestry.copy();add(i,active[i])
  log.append({'iteration':iteration,'kind':kind,'old_active_rows':list(rows),'new_active_rows':list(range(newid,newid+len(new))),'input_candidate_ancestry':sorted(ancestry),'old_min_quality':float(oq.min()),'new_min_quality':float(nq.min()),'old_below_count':int((oq<1e-5).sum()),'new_below_count':int((nq<1e-5).sum())})
 ids=np.asarray(sorted(active),np.int64);NT=np.asarray([active[i] for i in ids],np.int64);NA=np.asarray([attr[i] for i in ids],np.int64);nq=quality(X,NT)
 ptr=[0];ix=[]
 for i in ids:ix.extend(sorted(lineage[i]));ptr.append(len(ix))
 rep={'input_path':str(IN),'input_sha256':sha(IN),'owner014_sha256':sha(src),'attempted_cavities':attempted,'accepted_flips':log,'output_tets':len(NT),'output_nodes':len(X),'nonpositive_tets':int((dets(X,NT)<=0).sum()),'quality_below_1e5':int((nq<1e-5).sum()),'minimum_normalized_determinant':float(nq.min()),'boundary_oriented_exact':boundary(T)==boundary(NT),'all_support_nodes_still_referenced':bool(np.isin(np.arange(49951,len(X)),np.unique(NT)).all()),'support_gate_pass':support_ok(NT),'global_relative_volume_change':float((volume(X,NT)-volume(X,T))/volume(X,T)),'material_overlap_transfer_pending':True,'native_admission':False,'elapsed_s':time.monotonic()-start}
 np.savez_compressed(OUT/'candidate-geometry-only.npz',points_world_m=X,tetrahedra=NT,region_attributes=NA,physical_outer_node_ids=z['physical_outer_node_ids'],physical_outer_faces=z['physical_outer_faces'],physical_shell_faces=z['physical_shell_faces'],construction_closure_faces=z['construction_closure_faces'],parent_input_candidate_offsets=np.asarray(ptr,np.int64),parent_input_candidate_rows=np.asarray(ix,np.int64))
 rep['candidate_sha256']=sha(OUT/'candidate-geometry-only.npz')
 rep['geometry_support_gate_pass']=bool(rep['nonpositive_tets']==0 and rep['quality_below_1e5']==0 and rep['boundary_oriented_exact'] and rep['all_support_nodes_still_referenced'] and rep['support_gate_pass'] and abs(rep['global_relative_volume_change'])<1e-10)
 (OUT/'result.json').write_text(json.dumps(rep,indent=2)+'\n')
 print(json.dumps(rep),flush=True)
if __name__=='__main__':main()

