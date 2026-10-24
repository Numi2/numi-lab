import hashlib, importlib.util, json, math, os, sys, time, struct, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, vstack

ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=ROOT/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
AUD=E/'joint-functional-owner-audit-20261023'
FIT=AUD/'original128-finite-sat-fit-001l'
PART=FIT/'partial-step-001'
PREV=PART/'candidate.nhskin'
SEED=FIT/'relinearization-002e/iteration-003/candidate.nhskin'
SEED_SHA='99f349c5535ea3fec9720c98cfba44709563e8c66c361f85d1c5e75a8468f1bc'
SEED_REPORT=FIT/'relinearization-002e/iteration-003/iteration-report.json'
SEED_REPORT_SHA='ce0667db83af0509b57b9e05780256ba558b98ff898e99472c0d02ff31f6b153'
RESET=FIT/'reset-source128-refinement.nhskin'
NPZ=PART/'partial-step.npz'
EXACT=PART/'exact-self-004'
OUT=FIT/'relinearization-002g-root-guard-fixed'
RB=ROOT/'tools/rebind_human_skin_reference.py'
LIB=AUD/'original128-local-vertex-fit-003-conservative-area/build_original128_local_vertex_fit.py'
SAT=E/'finite-triangle-contact-planes-268/tools/human_skin_contact_planes.py'
LEX=Path('/tmp/human_skin_repair_lp_276.py')
CI_SRC=E/'incremental-exact-audit-owner-253/src'
IDENT=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
POSES={
 'shoulder_elevation':E/'joint-source-decomposition-053/shoulder-elevation-050/registered-skin-poses.json',
 'elbow_flexion':E/'joint-source-decomposition-053/elbow-flexion-080/registered-skin-poses.json',
 'shoulder_plane':Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001/shoulder_plane_bilateral_q35q73_m0p3_at_elevation0p5.json')}
IDENT_SHA='0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd'
PREV_SHA='152d2811f4f57d85435d97269f0259ee7a895d1dbee5174751b4e0d56cc9cf29'
RESET_SHA='dc5eea4c2bcad4fed5536c8fa64d4e207b2496bd3aa784fd7b8587e14c949fd6'
MAX_STEP=1.0
MAX_CUM=2.0
MAX_POSE_AXIS=1.25
TARGET=0.001
GUARD_MARGIN=0.20
AREA_FLOOR=0.10
AREA_LP_FLOOR=0.10
TV_WEIGHT=0.1
NEAR_MM=0.25
MAX_ITER=4
HISTORY_HELPER='/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-repair-guard-owner-284/tools/human_skin_contact_history.py'
CASE_NAMES=('rest','shoulder_elevation','elbow_flexion','shoulder_plane')

def shab(b): return hashlib.sha256(b).hexdigest()
def sha(p): return shab(Path(p).read_bytes())
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def rest_doc(s):
 rows=s.get('all_bodies',s.get('bodies'))
 return {'source_poses':[{'core_body':int(r['body_index']),'position':r['rest']['position_world_m'],'quaternion_xyzw':r['rest']['orientation_world_xyzw']} for r in rows]}
def pose_doc(s):
 if 'source_poses' in s:return s
 return {'source_poses':[{'core_body':int(r['body_index']),'position':r['current']['position_world_m'],'quaternion_xyzw':r['current']['orientation_world_xyzw']} for r in s['bodies']] }
def pointbits(x):
 return np.asarray(x,dtype=np.float32).tobytes()
def fmtkey(row):
 return tuple(np.asarray(row,dtype=np.float32).view(np.uint32).tolist())
def pose_records(V,F,qmap,members,valid,ci):
 n=len(V);eff=np.arange(n,dtype=np.int64)
 for q,ids in enumerate(members):
  if len(ids)<2:continue
  first=int(ids[0]);same=all(fmtkey(V[first])==fmtkey(V[int(j)]) for j in ids[1:])
  if valid.get(q,False) and same:eff[np.asarray(ids,dtype=np.int64)]=q
  else:eff[np.asarray(ids,dtype=np.int64)]=n+np.asarray(ids,dtype=np.int64)
 used=sorted(set(map(int,eff[np.asarray(F).ravel()])))
 cmap={u:i for i,u in enumerate(used)}
 rep={}
 for i,u in enumerate(eff):
  rep.setdefault(int(u),i)
 reps=np.asarray([rep[u] for u in used],np.int64)
 qface=np.asarray([[cmap[int(eff[int(v)])] for v in tri] for tri in F],np.int64)
 recs=ci._records([ci.float32_point_lattice_key(V[int(v)]) for v in reps],qface.tolist())
 return ci._immutable_self_pair_records(recs),eff

def build_keys(F,N,Xreset,edge,ew,source_nv,fit,ci):
 qmap,members,valid,reasons,collapsed=fit.source_topology_map(F,N,Xreset,ci)
 if collapsed:raise RuntimeError(('source reset topology collapses',collapsed[:8]))
 classid={}
 for q,mem in enumerate(members):
  good=len(mem)>1 and valid.get(q,False)
  for v in mem:classid[int(v)]=('q',int(q)) if good else ('v',int(v))
 keys=[classid.get(i,('v',i)) for i in range(source_nv)]
 for ee,ww in zip(edge,ew):
  a,b=map(int,ee);wa,wb=map(float,ww);ka,kb=keys[a],keys[b]
  if repr(ka)<=repr(kb):keys.append(('e',ka,kb,round(wb,12)))
  else:keys.append(('e',kb,ka,round(wa,12)))
 if len(keys)!=N:raise RuntimeError('key ancestry length')
 return qmap,members,valid,keys

def face_edges(F,face_ids):
 out=set()
 for tri in F[np.asarray(face_ids,np.int64)]:
  for j in range(3):out.add(tuple(sorted((int(tri[j]),int(tri[(j+1)%3])))))
 return sorted(out)

def current_pairs_from_json(case):
 doc=json.loads((EXACT/f'{case}.json').read_text())
 return set(tuple(map(int,p)) for p in doc['candidate_pairs']),set(tuple(map(int,p)) for p in doc['removed_pairs']),doc

def violation_sum(rows,d):
 vals=np.asarray([r['gap0_mm']+sum(v*d[k//3,k%3] for k,v in r['coeff'].items())-r['target_mm'] for r in rows],float)
 return float(np.maximum(0.,-vals).sum()),int(np.sum(vals < -1e-5)),float(vals.min(initial=0.))

def main():
 history=loadmod('contact_history284',HISTORY_HELPER)
 if OUT.exists():raise RuntimeError('output exists')
 if sha(PREV)!=PREV_SHA or sha(SEED)!=SEED_SHA or sha(SEED_REPORT)!=SEED_REPORT_SHA or sha(RESET)!=RESET_SHA or sha(IDENT)!=IDENT_SHA:raise RuntimeError('pinned input SHA mismatch')
 OUT.mkdir(parents=True)
 shutil.copyfile(__file__,OUT/'relinearize_shoulder_iterative.py')
 rb=loadmod('rb_relinearize_iter',RB);fit=loadmod('fit_relinearize_iter',LIB);sat=loadmod('sat_relinearize_iter',SAT);lex=loadmod('lex_relinearize_iter',LEX)
 sys.path.insert(0,str(CI_SRC));from numilab_human import cardiac_cavity_intersections as ci
 cand_raw=SEED.read_bytes();cand=rb.decode(cand_raw);reset=rb.decode(RESET.read_bytes())
 F=np.asarray(cand['inds'],np.int64).reshape(-1,3);N=int(cand['nv'])
 if reset['nv']!=N or reset['inds'].tobytes()!=cand['inds'].tobytes():raise RuntimeError('topology/base mismatch')
 W=np.asarray(cand['weights'],np.float64);Xlocal=np.asarray(cand['verts'][:,:3],np.float64)
 q=np.load(NPZ,allow_pickle=False);free=np.asarray(q['free_vertex_ids'],np.int64);roi=np.asarray(q['roi_face_rows'],np.int64)
 if free.shape!=(274,) or roi.shape!=(658,):raise RuntimeError('pinned local ROI changed')
 edgeanc=np.load(AUD/'original128-local-edge-refinement-005h/refinement-ancestry.npz',allow_pickle=False)
 edge=np.asarray(edgeanc['new_vertex_edge_source_ids'],np.int64);ew=np.asarray(edgeanc['new_vertex_edge_weights'],np.float64)
 iddoc=json.loads(IDENT.read_text());restpose=rest_doc(iddoc)
 A0,b0=rb.reference_transforms(cand,restpose,[0,0,0]);A0s,b0s=rb.reference_transforms(reset,restpose,[0,0,0])
 Xreset=rb.positions(reset,A0s,b0s).astype(np.float32)
 M0=np.einsum('nl,lij->nij',W,A0);cond=np.linalg.cond(M0)
 if not np.isfinite(cond).all() or cond.max()>10:raise RuntimeError(('reference map condition',float(cond.max())))
 inv0=np.linalg.inv(M0)
 all_docs={'rest':restpose}
 for name,p in POSES.items():all_docs[name]=pose_doc(json.loads(p.read_text()))
 cases={};D={}
 for name,doc in all_docs.items():
  A,b=rb.reference_transforms(cand,doc,[0,0,0]);V=rb.positions(cand,A,b).astype(np.float32)
  As,bs=rb.reference_transforms(reset,doc,[0,0,0]);S=rb.positions(reset,As,bs).astype(np.float32)
  Mp=np.einsum('nl,lij->nij',W,A);Dn=np.einsum('nij,njk->nik',Mp,inv0)
  cases[name]={'A':A,'b':b,'V':V,'S':S};D[name]=Dn
 source_nv=N-len(edge)
 qmap,members,valid,keys=build_keys(F,N,Xreset,edge,ew,source_nv,fit,ci)
 # Construct exact full-baseline caches in memory from pinned complete exact pair reports; no new full-surface audit.
 caches={};pairsets={};removedsets={};initial_cache_meta={}
 seed_doc=json.loads(SEED_REPORT.read_text())
 if seed_doc.get('candidate_sha256')!=SEED_SHA:raise RuntimeError('seed exact receipt candidate hash mismatch')
 for name in CASE_NAMES:
  curr_recs,eff=pose_records(cases[name]['V'],F,qmap,members,valid,ci)
  original_now,original_removed,doc=current_pairs_from_json(name if name!='rest' else 'rest')
  original_pairs=original_now|original_removed
  seed_pairs=set(tuple(map(int,p)) for p in seed_doc['pairsets'][name])
  expected=int(seed_doc['exact_poses'][name]['candidate_count'])
  if len(seed_pairs)!=expected:raise RuntimeError(('seed exact receipt count mismatch',name,len(seed_pairs),expected))
  cache=ci._CompleteSelfPairBaseline(curr_recs,tuple(sorted(seed_pairs)))
  caches[name]=cache;pairsets[name]=seed_pairs;removedsets[name]=original_pairs-seed_pairs
  # Recover the one historical receipt omitted by the old runner, using
  # exact incremental comparison from the retained complete partial-step cache.
  firstpath=FIT/'relinearization-002c/iteration-001/candidate.nhskin'
  firstsha=sha(firstpath)
  if firstsha!='309ec0143bc24d708d070340ad2be2aa4f676625583d87a3c80b60ecd3ed5e97':raise RuntimeError('historical candidate mismatch')
  prev_payload=rb.decode(PREV.read_bytes());first_payload=rb.decode(firstpath.read_bytes())
  pv=rb.positions(prev_payload,cases[name]['A'],cases[name]['b']).astype(np.float32)
  fv=rb.positions(first_payload,cases[name]['A'],cases[name]['b']).astype(np.float32)
  pr,_=pose_records(pv,F,qmap,members,valid,ci)
  fr,_=pose_records(fv,F,qmap,members,valid,ci)
  historical=ci._CompleteSelfPairBaseline(pr,tuple(sorted(original_now)))
  _,hrep=ci._update_complete_self_pair_baseline(historical,fr)
  fpairs=set(tuple(map(int,p)) for p in hrep['triangle_pairs'])
  removedsets[name] |= original_now-fpairs
  (OUT/('recovered-historical-'+name+'.json')).write_text(json.dumps({'candidate_sha256':firstsha,'previous_candidate_sha256':PREV_SHA,'complete_pair_count':len(fpairs),'triangle_pairs':[list(p) for p in sorted(fpairs)],'exact_incremental_report':hrep},indent=2)+'\n')
  history_paths=[FIT/'relinearization-002d/iteration-002/iteration-report.json', SEED_REPORT]
  prior=fpairs
  prior_sha=firstsha
  for hp in history_paths:
   hd=json.loads(hp.read_text())
   if hd['previous_candidate_sha256']!=prior_sha: raise RuntimeError(('broken guard history',str(hp)))
   following=set(tuple(map(int,p)) for p in hd['pairsets'][name])
   removedsets[name] |= prior-following
   prior=following;prior_sha=hd['candidate_sha256']
  if prior!=seed_pairs or prior_sha!=SEED_SHA:raise RuntimeError('history seed mismatch')
  initial_cache_meta[name]={'records':len(curr_recs),'pairs':len(seed_pairs),'removed_guard_pairs':len(removedsets[name]),'source128_receipt_pairs':len(original_pairs),'seed_exact_elapsed_s':float(seed_doc['exact_poses'][name].get('elapsed_s',0.))}
  print(json.dumps({'event':'seed_complete_cache','case':name,'seed_pairs':len(seed_pairs),'resolved_pair_guards':len(removedsets[name]),'source128_pairs':len(original_pairs)},sort_keys=True),flush=True)
 # Verify exact report candidate hash and exact helper pin before any solve.
 sr=json.loads((EXACT/'report.json').read_text())
 if sr['inputs']['candidate_payload_sha256']!=PREV_SHA:raise RuntimeError('exact report candidate mismatch')
 if sr['inputs']['exact_helper_sha256']!=sha(Path(ci.__file__)):raise RuntimeError('exact helper hash mismatch')
 # Build source seam class alias check.
 if (len(valid),sum(bool(v) for v in valid.values()))!=(79,59):raise RuntimeError(('seam source signature',len(valid),sum(bool(v) for v in valid.values())))
 prev_normal_cache={name:{} for name in CASE_NAMES}
 previous_delta=np.zeros((N,3),np.float64)
 previous_delta[free]=(rb.positions(cand,A0,b0).astype(np.float64)[free]-Xreset.astype(np.float64)[free])*1000.
 # Recreate the one-face-ring expansion used by iteration 001.
 edge_to_faces=defaultdict(list)
 for fi,t in enumerate(F):
  for j in range(3):edge_to_faces[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))].append(fi)
 adj=defaultdict(set)
 for fs in edge_to_faces.values():
  if len(fs)==2:adj[fs[0]].add(fs[1]);adj[fs[1]].add(fs[0])
 expanded=set(map(int,roi))
 for fi in list(expanded):expanded.update(adj[fi])
 ec=defaultdict(int)
 for fi in expanded:
  for j in range(3):ec[tuple(sorted((int(F[fi,j]),int(F[fi,(j+1)%3]))))]+=1
 fixed2=np.asarray(sorted({v for e,c in ec.items() if c==1 for v in e}),np.int64)
 verts2=np.unique(F[np.asarray(sorted(expanded),np.int64)].ravel())
 free2=np.setdiff1d(verts2,fixed2)
 if not set(map(int,free)).issubset(set(map(int,free2))):raise RuntimeError('one-ring expansion lost old free vertices')
 roi=np.asarray(sorted(expanded),np.int64);free=free2
 print(json.dumps({'event':'seed_patch_expansion','faces':len(roi),'free_vertices':len(free),'fixed_vertices':len(fixed2)},sort_keys=True),flush=True)
 steps=[]
 last_candidate_path=SEED
 expanded_once=True
 it=4
 while it<=MAX_ITER:
  iter_dir=OUT/f'iteration-{it:03d}';iter_dir.mkdir()
  print(json.dumps({'event':'iteration_start','iteration':it,'current_pairs':{k:len(v) for k,v in pairsets.items()},'removed_guards':{k:len(v) for k,v in removedsets.items()},'free_vertices':len(free)},sort_keys=True),flush=True)
  if len(free)==0:break
  fl={int(v):i for i,v in enumerate(free)}
  contacts=[];contact_stats={};pair_normals={name:{} for name in CASE_NAMES}
  previous_pairsets={name:set(pairsets[name]) for name in CASE_NAMES}
  previous_removedsets={name:set(removedsets[name]) for name in CASE_NAMES}
  previous_pairs_for_iteration={k:set(v) for k,v in pairsets.items()}
  # Swept-AABB pair enumeration: all movable ROI faces against full source face bank.
  near_pairs_by_case={}
  for name in CASE_NAMES:
   V=cases[name]['V'].astype(np.float64);Dn=D[name]
   tri=V[F].astype(np.float64);lo=tri.min(axis=1);hi=tri.max(axis=1);cent=tri.mean(axis=1)
   rad=np.linalg.norm(tri-cent[:,None,:],axis=2).max(axis=1)
   move=np.zeros(len(F),np.float64)
   for fi in np.flatnonzero(np.any(np.isin(F,free),axis=1)):
    ids=F[int(fi)];act=[int(v) for v in ids if int(v) in fl]
    if act:
     # The LP below directly bounds each posed vertex displacement component.
     # Sweep that enforced space bound, rather than inflating it through D twice.
     move[int(fi)]=MAX_POSE_AXIS*math.sqrt(3.0)
   tree=cKDTree(cent);maxrad=float(rad.max());maxmove=float(move.max(initial=0.))
   swept=set()
   moving=np.flatnonzero(move>0)
   for fa0 in moving:
    fa=int(fa0);radius=float(rad[fa]+maxrad+(move[fa]+maxmove+GUARD_MARGIN)*.001)
    js=tree.query_ball_point(cent[fa],radius)
    for jb0 in js:
     fb=int(jb0)
     if fb==fa:continue
     a,b=(fa,fb) if fa<fb else (fb,fa)
     axisgap=np.maximum(np.maximum(lo[a]-hi[b],lo[b]-hi[a]),0.)
     if float(np.linalg.norm(axisgap))*1000. <= move[a]+move[b]+GUARD_MARGIN:
      swept.add((a,b))
   union=set(pairsets[name])|set(removedsets[name])|swept
   # Restrict solving to local contacts with at least one face in current ROI/free-star, but exact records remain complete globally.
   roi_set=set(map(int,roi)); union={p for p in union if p[0] in roi_set or p[1] in roi_set or (np.any(np.isin(F[p[0]],free)) or np.any(np.isin(F[p[1]],free)))}
   current=pairsets[name];prior=removedsets[name];near_added=union-current-prior
   near_pairs_by_case[name]=swept
   nrows=0;exactpairs=0;guardpairs=0;fixedcontr=0;sharedpairs=0;maxgap=-1e99
   for fa,fb in sorted(union):
    ta=F[fa];tb=F[fb];A=V[ta];B=V[tb]
    common=[]
    for ia in range(3):
     for ib in range(3):
      va,vb=int(ta[ia]),int(tb[ib])
      if keys[va]==keys[vb] and np.array_equal(cases[name]['V'][va],cases[name]['V'][vb]) and np.allclose(Dn[va],Dn[vb],rtol=0,atol=1e-10):
       common.append((ia,ib))
    is_exact=(fa,fb) in current
    is_guard=(fa,fb) in prior
    # Keep finite-SAT guards only within the proposed trust sweep. Exact pairs always enter.
    rstA=cases['rest']['V'][ta].astype(np.float64);rstB=cases['rest']['V'][tb].astype(np.float64)
    raxes=sat.finite_triangle_support_planes(rstA,rstB)
    nrest=np.asarray(raxes[0]['normal'],np.float64)
    local=np.mean(Dn[np.r_[ta,tb]],axis=0)
    try: preferred=np.linalg.solve(local.T,nrest)
    except np.linalg.LinAlgError: preferred=nrest.copy()
    norm=float(np.linalg.norm(preferred))
    if not np.isfinite(norm) or norm<1e-12:preferred=nrest.copy()
    else:preferred/=norm
    axes=sat.finite_triangle_support_planes(A,B,preferred_normal=preferred)
    if not axes:axes=sat.finite_triangle_support_planes(A,B)
    if not axes:continue
    if common:
     ranked=[]
     for ax in axes:
      n=np.asarray(ax['normal'],np.float64);G=((A[:,None,:]-B[None,:,:])@n)*1000.
      vals=[float(G[i,j]) for i in range(3) for j in range(3) if (i,j) not in common]
      ranked.append((min(vals,default=0.),ax,G))
     score,chosen,G=max(ranked,key=lambda x:x[0])
    else:
     chosen=axes[0];G=((A[:,None,:]-B[None,:,:])@chosen['normal'])*1000.;score=float(chosen['gap_mm'])
    # Nearby guard only if actual SAT interval is within a certified trust bound.
    trust=move[fa]+move[fb]+GUARD_MARGIN
    if not is_exact and not is_guard and float(chosen['gap_mm'])>trust:continue
    if not is_exact and not is_guard and common and float(chosen['gap_mm'])>=-1e-8:continue
    n=np.asarray(chosen['normal'],np.float64);pair_normals[name][(fa,fb)]=n.tolist()
    maxgap=max(maxgap,float(chosen['gap_mm']))
    exactpairs+=int(is_exact);guardpairs+=int(is_guard or (fa,fb) in near_added);sharedpairs+=int(bool(common))
    for ia in range(3):
     for ib in range(3):
      same=(ia,ib) in common
      gap=float(G[ia,ib])
      target=0.0 if same else TARGET
      coeff={}
      for v,sign in ((int(ta[ia]),1.0),(int(tb[ib]),-1.0)):
       qix=fl.get(v)
       if qix is None:continue
       g=sign*(Dn[v].T@n)
       for cc in range(3):
        key=3*qix+cc;coeff[key]=coeff.get(key,0.)+float(g[cc])
      rhs=target-gap
      row={'case':name,'faces':[fa,fb],'corners':[ia,ib],'shared_identity':same,'normal':n.tolist(),'gap0_mm':gap,'target_mm':target,'rhs':rhs,'coeff':coeff,'guard':bool(not is_exact)}
      if not coeff and rhs>1e-7:fixedcontr+=1
      contacts.append(row);nrows+=1
   contact_stats[name]={'exact_pairs':exactpairs,'resolved_or_swept_guard_pairs':guardpairs,'swept_aabb_candidates':len(swept),'constraint_rows':nrows,'shared_pairs':sharedpairs,'fixed_contradiction_rows':fixedcontr,'largest_kept_gap_mm':maxgap}
  # If current fixed ring blocks the separating direction, expand the local movable set one face ring, once.
  fixed_total=sum(v['fixed_contradiction_rows'] for v in contact_stats.values())
  if fixed_total and not expanded_once:
   # Expand only through the current local ROI boundary: one edge-adjacent face ring, then anchor its new outer boundary.
   edge_to_faces=defaultdict(list)
   for fi,t in enumerate(F):
    for j in range(3):edge_to_faces[tuple(sorted((int(t[j]),int(t[(j+1)%3]))))].append(fi)
   adj=defaultdict(set)
   for fs in edge_to_faces.values():
    if len(fs)==2:adj[fs[0]].add(fs[1]);adj[fs[1]].add(fs[0])
   add=set(int(f) for f in roi)
   for f in list(add):add.update(adj[f])
   counts=defaultdict(int)
   for f in add:
    for j in range(3):counts[tuple(sorted((int(F[f,j]),int(F[f,(j+1)%3]))))]+=1
   boundary=[e for e,c in counts.items() if c==1]
   fixed2=np.asarray(sorted({v for e in boundary for v in e}),np.int64)
   verts=np.unique(F[np.asarray(sorted(add),np.int64)].ravel())
   free2=np.setdiff1d(verts,fixed2)
   if len(free2)>len(free) and set(map(int,free)).issubset(set(map(int,free2))):
    roi=np.asarray(sorted(add),np.int64);free=free2;expanded_once=True
    print(json.dumps({'event':'expanded_local_patch','reason':'fixed_boundary_contradictions','fixed_conflicts':fixed_total,'faces':len(roi),'free_vertices':len(free),'fixed_vertices':len(fixed2)},sort_keys=True),flush=True)
    shutil.rmtree(iter_dir)
    continue
  # Source-relative nonlinear oriented-area inequalities for every local face in all four poses.
  roi_tris=F[roi]; area_rows=[]; area0={}
  for name in CASE_NAMES:
   V=cases[name]['V'].astype(np.float64);S=cases[name]['S'].astype(np.float64);Dn=D[name]
   P=V[roi_tris];T=S[roi_tris]
   cs=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);den=np.linalg.norm(cs,axis=1)
   if np.any(den<1e-20):raise RuntimeError(('source-degenerate area face',name,int(np.argmin(den))))
   cc=np.cross(P[:,1]-P[:,0],P[:,2]-P[:,0]);ratio=np.einsum('ij,ij->i',cc,cs)/np.maximum(den*den,1e-300)
   ix=int(np.argmin(ratio));area0[name]={'min_ratio':float(ratio[ix]),'face':int(roi[ix]),'face_count':len(roi)}
   for qix,fi in enumerate(roi):
    tri=F[int(fi)];p=P[qix];n=cs[qix]/den[qix];cur=float(ratio[qix]);co={}
    grads=(np.cross(p[1]-p[2],n),np.cross(p[2]-p[0],n),np.cross(p[0]-p[1],n))
    for k,g0 in enumerate(grads):
     v=int(tri[k]);j=fl.get(v)
     if j is None:continue
     g=Dn[v].T@g0*(.001/den[qix])
     for cc0 in range(3):
      col=3*j+cc0;co[col]=co.get(col,0.)+float(g[cc0])
    area_rows.append({'coeff':co,'rhs':AREA_LP_FLOOR-cur})
  edges=face_edges(F,roi)
  ties=[]
  verts_set=set(map(int,np.unique(F[roi])));fixedset=verts_set-set(map(int,free))
  fl={int(v):i for i,v in enumerate(free)}
  for qid,mem in enumerate(members):
   if len(mem)<2 or not valid.get(qid,False):continue
   active=[fl[int(v)] for v in mem if int(v) in fl]
   anchored=any(int(v) not in verts_set or int(v) in fixedset for v in mem)
   if anchored:
    for j in active:
     for ax in range(3):ties.append((j,None,ax))
   elif len(active)>1:
    for j in active[1:]:
     for ax in range(3):ties.append((active[0],j,ax))
  fn=fit.make_lp
  fn.__globals__['MAX_DISP_MM']=MAX_STEP
  fn.__globals__['TV_WEIGHT']=TV_WEIGHT
  fn.__globals__['SLACK_WEIGHT']=1e6
  solve_info={}
  def logged_lp(*args,**kwargs):
   c=np.asarray(args[0] if args else kwargs['c'],np.float64);A=kwargs['A_ub'].tocsr();b=np.asarray(kwargs['b_ub'],np.float64);bounds=list(kwargs['bounds'])
   n=3*len(free);m=len(contacts);slack_start=len(c)-m
   cum=(cases['rest']['V'][free].astype(np.float64)-Xreset[free].astype(np.float64))*1000.
   for j in range(n):
    lo=max(-MAX_STEP,-MAX_CUM-cum[j//3,j%3]);hi=min(MAX_STEP,MAX_CUM-cum[j//3,j%3])
    bounds[j]=(lo,hi)
   # Bound each incremental posed vertex displacement on all states to prevent excessive sweep/stretch.
   rr=[];cc=[];vv=[];rhs=[];row=0
   for name in CASE_NAMES:
    Dn=D[name]
    for j,v in enumerate(free):
     for ax in range(3):
      co=Dn[int(v)][ax]
      for k,val in enumerate(co):
       if abs(float(val))>1e-14:
        rr.append(row);cc.append(3*j+k);vv.append(float(val))
      rhs.append(MAX_POSE_AXIS);row+=1
      for k,val in enumerate(co):
       if abs(float(val))>1e-14:
        rr.append(row);cc.append(3*j+k);vv.append(-float(val))
      rhs.append(MAX_POSE_AXIS);row+=1
   P=coo_matrix((vv,(rr,cc)),shape=(row,len(c))).tocsr()
   A=vstack([A,P],format='csr');b=np.r_[b,np.asarray(rhs)]
   savepath=iter_dir/'lp-A-ub.npz'
   from scipy.sparse import save_npz
   save_npz(savepath,A);np.savez_compressed(iter_dir/'lp-problem.npz',c=c,b_ub=b,bounds_low=np.asarray([(-np.inf if z[0] is None else z[0]) for z in bounds]),bounds_high=np.asarray([(np.inf if z[1] is None else z[1]) for z in bounds]))
   result,diag=lex.solve_contact_lp_lexicographic(c,A,b,bounds,slack_start,m,time_limit=90.,slack_sum_tolerance=1e-7)
   solve_info.update(diag);solve_info.update(status=int(result.status),success=bool(result.success),message=str(result.message),nit=int(getattr(result,'nit',-1)),fun=None if result.fun is None else float(result.fun))
   return result
  fn.__globals__['linprog']=logged_lp
  try:
   delta,solver=fn(contacts,area_rows,free,edges,previous_delta[free],ties);solver['lexicographic']=solve_info
  except Exception as ex:
   rep={'status':'lp_failed','error':str(ex),'input_hashes':{'candidate':sha(PREV),'exact_report':sha(EXACT/'report.json')},'contact_stats':contact_stats,'area0':area0,'solver':solve_info}
   (iter_dir/'iteration-report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
   print(json.dumps({'event':'lp_failed','iteration':it,'error':str(ex),'stats':contact_stats},sort_keys=True),flush=True);break
  d0=np.zeros((N,3),np.float64);d0[free]=delta
  l0=violation_sum(contacts,delta)
  # Exact nonlinear pose area check and choose largest positive area step that reduces contact violation.
  trials=[];chosen=None;alpha_used=None;current_violation=violation_sum(contacts,np.zeros((len(free),3)))[0]
  def eval_area(alpha):
   z=delta*alpha;ans={}
   for name in CASE_NAMES:
    V=cases[name]['V'].astype(np.float64).copy()
    V[free]+=np.einsum('nij,nj->ni',D[name][free],z*.001)
    P=V[roi_tris];T=cases[name]['S'][roi_tris].astype(np.float64)
    cs=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);cc=np.cross(P[:,1]-P[:,0],P[:,2]-P[:,0]);den=np.einsum('ij,ij->i',cs,cs)
    rr=np.einsum('ij,ij->i',cc,cs)/np.maximum(den,1e-300);ii=int(np.argmin(rr))
    ans[name]={'min_ratio':float(rr[ii]),'face':int(roi[ii]),'below_floor':int(np.sum(rr<AREA_FLOOR))}
   return ans
  for alpha in (1.,.75,.5,.25,.125):
   lm=violation_sum(contacts,delta*alpha);area=eval_area(alpha)
   # Merit must decrease from the current exact local state; geometry is separately exact-audited after F32.
   ok=min(x['min_ratio'] for x in area.values())>=AREA_FLOOR and lm[0]<current_violation-1e-5
   trials.append({'alpha':alpha,'linearized_slack_sum_mm':lm[0],'violating_rows':lm[1],'minimum_margin_mm':lm[2],'area':area,'accepted':bool(ok)})
   if ok:chosen=delta*alpha;alpha_used=alpha;break
  if chosen is None:
   rep={'status':'no_area_safe_merit_step','solver':solver,'contact_stats':contact_stats,'linearized_before':current_violation,'linearized_fullstep':l0[0],'area_before':area0,'trials':trials,'free_vertices':len(free),'contact_rows':len(contacts),'area_rows':len(area_rows),'swept_pair_counts':{k:len(v) for k,v in near_pairs_by_case.items()}}
   (iter_dir/'iteration-report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
   print(json.dumps({'event':'no_step','iteration':it,'current_slack':current_violation,'full_step_slack':l0[0],'contact_rows':len(contacts),'stats':contact_stats},sort_keys=True),flush=True);break
  # Emit a geometry-only candidate by exact registered rest inverse, preserving all original bind/weight bytes.
  target_current=cases['rest']['V'].astype(np.float64).copy()
  target=target_current.astype(np.float64).copy()
  target_current[free]+=chosen*.001
  target[free]=target_current[free]
  local=Xlocal.copy();raw=bytearray(cand_raw)
  for v in free:
   if np.linalg.norm(chosen[fl[int(v)]])<=1e-12:continue
   displacement=chosen[fl[int(v)]]*.001
   local[int(v)]+=np.linalg.solve(M0[int(v)],displacement)
   target[int(v)]=target_current[int(v)]
   struct.pack_into('<3f',raw,cand['vertoff']+int(v)*56,*map(float,local[int(v)]))
  # ABI-5 normals: area-weighted incident-triangle normals from registered rest coordinates.
  affected=np.flatnonzero(np.any(np.isin(F,free),axis=1));normalrows=np.unique(F[affected].ravel())
  tri=target[F].astype(np.float32).astype(np.float64);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
  sums=np.zeros((N,3),np.float64);np.add.at(sums,F.ravel(),np.repeat(cross,3,axis=0))
  lens=np.linalg.norm(sums,axis=1)
  if np.any(lens[normalrows]<=1e-20):raise RuntimeError(('normal cancellation',int(normalrows[np.flatnonzero(lens[normalrows]<=1e-20)[0]])))
  for v in normalrows:struct.pack_into('<3f',raw,cand['vertoff']+int(v)*56+12,*map(float,sums[int(v)]/lens[int(v)]))
  out_raw=bytes(raw);out_payload=rb.decode(out_raw);candidate_path=iter_dir/'candidate.nhskin';candidate_path.write_bytes(out_raw)
  if out_payload['inds'].tobytes()!=cand['inds'].tobytes() or out_payload['weights'].tobytes()!=cand['weights'].tobytes() or out_payload['binds'].tobytes()!=cand['binds'].tobytes():raise RuntimeError('geometry-only output mutated topology/weights/binds')
  emitted_rest=rb.positions(out_payload,A0,b0).astype(np.float32);resterr=np.linalg.norm(emitted_rest-target.astype(np.float32),axis=1)
  if float(resterr.max(initial=0.))>2e-6:raise RuntimeError(('rest rebind error',float(resterr.max())))
  # Exact incremental full-surface self audit, updating resident caches for all 4 poses.
  newcases={};newD={};exact_reports={}
  for name in CASE_NAMES:
   A=cases[name]['A'];b=cases[name]['b'];Vnew=rb.positions(out_payload,A,b).astype(np.float32)
   newcases[name]={'A':A,'b':b,'V':Vnew,'S':cases[name]['S']}
   recs,eff=pose_records(Vnew,F,qmap,members,valid,ci)
   newcache,report=ci._update_complete_self_pair_baseline(caches[name],recs)
   newpairs=set(tuple(map(int,p)) for p in report['triangle_pairs'])
   caches[name]=newcache;pairsets[name]=newpairs
   exact_reports[name]=report
  removedsets=history.accumulate_resolved_contacts(previous_removedsets,previous_pairsets,pairsets)
  if any(not previous_removedsets[k].issubset(removedsets[k]) for k in CASE_NAMES): raise RuntimeError('resolved history shrank')
  current_delta=(emitted_rest.astype(np.float64)-Xreset.astype(np.float64))*1000.
  step_report={'schema':'numi.human.shoulder-elbow-sat-relinearization-step.v1','iteration':it,'status':'exact_incremental_gate_complete','candidate_sha256':shab(out_raw),'candidate_path':str(candidate_path),'previous_candidate_sha256':shab(cand_raw),'exact_helper_sha256':sha(Path(ci.__file__)),'finite_sat_helper_sha256':sha(SAT),'lp_helper_sha256':sha(LEX),'max_rest_rebind_error_m':float(resterr.max(initial=0.)),'geometry_only':True,'resolved_guard_pairs':{k:[list(p) for p in sorted(v)] for k,v in removedsets.items()},'guard_helper_sha256':sha(HISTORY_HELPER),'unchanged_weights_binds_faces':True,'free_vertices':len(free),'moved_vertex_count':int(np.sum(np.linalg.norm(chosen,axis=1)>1e-12)),'max_step_component_mm':float(np.abs(chosen).max(initial=0.)),'max_step_vector_mm':float(np.linalg.norm(chosen,axis=1).max(initial=0.)),'cumulative_max_component_mm':float(np.abs(current_delta[free]).max(initial=0.)),'alpha':alpha_used,'contact_rows':len(contacts),'linearized_violation_before_mm':current_violation,'linearized_violation_after_mm':violation_sum(contacts,chosen)[0],'solver':solver,'contact_stats':contact_stats,'area_before':area0,'area_trials':trials,'swept_aabb_pair_counts':{k:len(v) for k,v in near_pairs_by_case.items()},'exact_poses':{}}
  for name,rep in exact_reports.items():
   oldn=len(previous_pairsets[name])
   currn=int(rep['count'])
   step_report['exact_poses'][name]={'previous_count':int(oldn),'candidate_count':currn,'removed_count':int(len(set(json.loads((EXACT/f'{name}.json').read_text())['candidate_pairs'])-set(tuple(map(int,p)) for p in rep['triangle_pairs'])) if it==1 else len([])),'new_count':None,'changed_faces':int(rep['selected_face_count']),'reused_pairs':int(rep['reused_unchanged_crossing_pairs']),'elapsed_s':float(rep.get('elapsed_s',0.))}
   # robust set differences against current parent cache pairset before update are captured below in next receipt
  for name,rep in exact_reports.items():
   # Reconstruct prior/current using the cache update report's repeated set plus previous data retained.
   prior_pairs=set(pairsets[name]) # already new; use the former current pairset from exact json for first step below
   prior_json=previous_pairsets[name]
   new_pairs=pairsets[name]
   step_report['exact_poses'][name]['removed_count']=len(prior_json-new_pairs)
   step_report['exact_poses'][name]['new_count']=len(new_pairs-prior_json)
  step_report['pairsets']={k:[list(p) for p in sorted(v)] for k,v in pairsets.items()}
  (iter_dir/'exact-pose-results.json').write_text(json.dumps({k:{'count':v['count'],'triangle_pairs':v['triangle_pairs'],'complete_surface_face_count':v['complete_surface_face_count'],'selected_face_count':v['selected_face_count'],'exact_incremental_reuse':v['exact_incremental_reuse'],'reused_unchanged_crossing_pairs':v['reused_unchanged_crossing_pairs'],'aabb_candidate_pairs':v.get('aabb_candidate_pairs'),'elapsed_s':v.get('elapsed_s')} for k,v in exact_reports.items()},indent=2,sort_keys=True)+'\n')
  (iter_dir/'iteration-report.json').write_text(json.dumps(step_report,indent=2,sort_keys=True)+'\n')
  steps.append(step_report)
  # Update all in-memory state for the next relinearization, no baseline recomputation.
  cand_raw=out_raw;cand=out_payload;Xlocal=np.asarray(cand['verts'][:,:3],np.float64);cases=newcases;previous_delta=current_delta
  print(json.dumps({'event':'iteration_complete','iteration':it,'candidate_sha256':step_report['candidate_sha256'],'exact_counts':{k:v['candidate_count'] for k,v in step_report['exact_poses'].items()},'new_pairs':{k:v['new_count'] for k,v in step_report['exact_poses'].items()},'removed_pairs':{k:v['removed_count'] for k,v in step_report['exact_poses'].items()},'linear_violation_before':current_violation,'linear_violation_after':step_report['linearized_violation_after_mm'],'exact_elapsed_s':{k:v['elapsed_s'] for k,v in step_report['exact_poses'].items()}},sort_keys=True),flush=True)
  # Require meaningful exact improvement or stop; do not repeat a stalled linear family.
  prev_counts={k:len(previous_pairsets[k]) for k in CASE_NAMES}
  after_counts={k:step_report['exact_poses'][k]['candidate_count'] for k in CASE_NAMES}
  if sum(after_counts.values())>=sum(prev_counts.values()) and step_report['linearized_violation_after_mm']>=current_violation*.995:
   step_report['stop_reason']='no_measured_exact_or_linearized_progress'
   (iter_dir/'iteration-report.json').write_text(json.dumps(step_report,indent=2,sort_keys=True)+'\n')
   break
  last_candidate_path=candidate_path
  if sum(after_counts.values())==0:break
  it+=1
 final={'schema':'numi.human.shoulder-elbow-sat-relinearization-batch.v1','input_candidate_sha256':SEED_SHA,'seed_previous_sha256':PREV_SHA,'seed_iteration_report_sha256':SEED_REPORT_SHA,'reset_sha256':RESET_SHA,'exact004_sha256':sha(EXACT/'report.json'),'exact_helper_sha256':sha(Path(ci.__file__)),'pose_cache_initialized_from_complete_exact_receipt':True,'iterations':steps,'final_candidate_sha256':shab(cand_raw),'final_pair_counts':{k:len(pairsets[k]) for k in CASE_NAMES},'final_candidate_path':str(last_candidate_path),'status':'intermediate_not_admitted'}
 (OUT/'batch-report.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'event':'batch_complete','status':final['status'],'iterations':len(steps),'final_pair_counts':final['final_pair_counts'],'candidate_sha256':final['final_candidate_sha256']},sort_keys=True),flush=True)

if __name__=='__main__':main()
