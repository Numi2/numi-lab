from __future__ import annotations
import hashlib, importlib.util, json, sys, time, traceback
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from types import SimpleNamespace

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=C.parent/'audit-159/report.json'
TOPO=S/'per-bank-workers-20261027/b65-99/attempt-002/fan-hole-ancestry-check.json'
ENDPOINTS=S/'per-bank-workers-20261027/b65-99/attempt-003-exact-endpoint-identity/receipt.json'
PREFLIGHT=S/'per-bank-workers-20261027/b65-99/attempt-004-bone-aware-connected-patch-prep-001/witness-preflight.json'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=ROOT/'audit_skin_bone_surfaces_159.py'
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
Q_PIN='5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
OUT=Path(__file__).resolve().parent
C_PIN='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
A_PIN='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
PREFLIGHT_PIN='579ab33b2a1922af8075d7825b93a2a132e0f84aaf072b85c56a2196f9083855'
CLEARANCE_MM=0.05
SCALE_M=0.001
TRUST_STEP_MM=0.5
MAX_TOTAL_MM=3.0
MAX_OUTER_ITERS=1
AREA_FLOOR=0.10
SMOOTH_WEIGHT=0.5
started=time.monotonic()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def frac(x):return Fraction.from_float(float(x))
def exact_tri(t):return tuple(tuple(frac(x) for x in p) for p in np.asarray(t).reshape(3,3))
def exact_point(row):return tuple(Fraction(x) for x in row)
def barycentric(p,tri,exact):
 n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]));axes=exact._axes(n)
 a,b,c,q=[tuple(v[k] for k in axes) for v in (tri[0],tri[1],tri[2],p)]
 den=exact._orient2(a,b,c)
 if den==0:raise RuntimeError('degenerate exact triangle during witness mapping')
 return (exact._orient2(b,c,q)/den,exact._orient2(c,a,q)/den,exact._orient2(a,b,q)/den)
def patch_topology(F,rows):
 use=Counter();adj=defaultdict(set)
 for r in rows:
  f=F[int(r)]
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
   a=int(a);b=int(b);e=tuple(sorted((a,b)));use[e]+=1;adj[a].add(b);adj[b].add(a)
 if any(v not in (1,2) for v in use.values()):raise RuntimeError('patch has nonmanifold incidence')
 boundary=[e for e,n in use.items() if n==1];ba=defaultdict(set);deg=Counter()
 for a,b in boundary:ba[a].add(b);ba[b].add(a);deg[a]+=1;deg[b]+=1
 loops=0;seen=set()
 for v in ba:
  if v in seen:continue
  loops+=1;stack=[v];seen.add(v)
  while stack:
   x=stack.pop()
   for y in ba[x]:
    if y not in seen:seen.add(y);stack.append(y)
 comps=0;sv=set()
 for v in adj:
  if v in sv:continue
  comps+=1;stack=[v];sv.add(v)
  while stack:
   x=stack.pop()
   for y in adj[x]:
    if y not in sv:sv.add(y);stack.append(y)
 euler=len(adj)-len(use)+len(rows)
 meta={'vertex_count':len(adj),'edge_count':len(use),'face_count':len(rows),'boundary_edge_count':len(boundary),
  'boundary_vertex_count':len(ba),'boundary_loop_count':loops,'boundary_degree_histogram':{str(k):int(v) for k,v in Counter(deg.values()).items()},
  'connected_components':comps,'euler_characteristic':int(euler),'is_single_disk':bool(comps==1 and loops==1 and euler==1 and all(v==2 for v in deg.values()))}
 return boundary,set(ba),meta

def mesh_records(points,faces,face_ids,vertex_ids,exact):
 pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)]
 rec=[]
 for fid,face in zip(face_ids,faces):
  inds=tuple(map(int,face));tri=tuple(pts[i] for i in inds)
  n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
  if not any(n):raise RuntimeError(f'emitted degenerate skin triangle {fid}')
  lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
  rec.append((tri,lo,hi,int(fid),tuple(int(vertex_ids[i]) for i in inds)))
 return rec

def quality(points_local,faces,base_local):
 tri=np.asarray(points_local,np.float64)[faces];ref=np.asarray(base_local,np.float64)[faces]
 pn=np.cross(ref[:,1]-ref[:,0],ref[:,2]-ref[:,0]);cn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 p2=np.einsum('ij,ij->i',pn,pn);pl=np.sqrt(np.maximum(p2,1e-300))
 ur=np.linalg.norm(cn,axis=1)/pl;sr=np.einsum('ij,ij->i',pn,cn)/np.maximum(p2,1e-300)
 ang=[];asp=[]
 for t in tri:
  e=[float(np.linalg.norm(t[(i+1)%3]-t[i])) for i in range(3)];area=float(np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))*0.5)
  asp.append(max(e)**2/(4*np.sqrt(3)*max(area,1e-30)))
  aa=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
   aa.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-30),-1,1)))))
  ang.append(min(aa))
 return {'face_count':len(faces),'minimum_unsigned_parent_area_ratio':float(ur.min()),'minimum_signed_orientation_ratio':float(sr.min()),
  'minimum_angle_degrees':float(min(ang)),'p05_angle_degrees':float(np.percentile(ang,5)),
  'maximum_aspect_measure':float(max(asp)),'orientation_positive':bool(np.all(sr>0))}

def main():
 report={'schema':'numi.human.bank65_99.witnessed-trust-region-highs-lp.v1','status':'STARTED',
  'candidate_path':str(C),'geometry_or_native_admission':False,'candidate_modified':False,
  'scope':'Private 248-face local patch; one sparse HiGHS LP with hard linearized area, edge-gradient, boundary and trust rows plus explicit witness slacks; exact local target-bank recheck after emitted Float32; local self only after strict mapped parent-identity reduction.',
  'limits':{'trust_step_mm':TRUST_STEP_MM,'max_outer_iterations':MAX_OUTER_ITERS,'max_cumulative_node_motion_mm':MAX_TOTAL_MM,
    'clearance_halfspace_mm':CLEARANCE_MM,'shape_area_ratio_floor':AREA_FLOOR}}
 try:
  if sha(C)!=C_PIN or sha(A)!=A_PIN or sha(PREFLIGHT)!=PREFLIGHT_PIN:raise RuntimeError('candidate/audit/preflight pin mismatch')
  Pdoc=json.load(open(PREFLIGHT));top=json.load(open(TOPO));edoc=json.load(open(ENDPOINTS));auditdoc=json.load(open(A))
  Z={}
  with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
  P32=np.asarray(Z['points_world_m'],np.float32);P=P32.astype(np.float64);F=np.asarray(Z['source_O_faces'],np.int64)
  base_rows=sorted(set(map(int,top['two_ring_patch_summary'][0]['face_rows']))|set(map(int,top['fan_expansion'][0]['fan_face_rows'])))
  rows=sorted(set(base_rows)|set(map(int,Pdoc['measured_expansion']['added_face_rows'])))
  if len(rows)!=248:raise RuntimeError(f'measured patch face count mismatch: {len(rows)}')
  boundary,boundary_nodes,topometa=patch_topology(F,rows)
  if not topometa['is_single_disk'] or len(boundary)!=62:raise RuntimeError(f'measured expanded patch no longer a 62-edge disk: {topometa}')
  ind=np.asarray(Z['face_source_ancestry_indptr'],np.int64)
  if any(ind[r+1]<=ind[r] for r in rows):raise RuntimeError('expanded patch has a missing source-face ancestry row')
  patchfaces=F[np.asarray(rows,np.int64)];patchnodes=np.unique(patchfaces.reshape(-1));fixed=np.asarray(sorted(boundary_nodes),np.int64)
  node_set=set(map(int,patchnodes));fixed_set=set(map(int,fixed));movable=np.asarray(sorted(node_set-fixed_set),np.int64)
  helper=loadmod('b65_99_fit_bank_helper',HELPER);audit_tool=loadmod('b65_99_fit_audit_owner',AUDIT_TOOL);exact=audit_tool.exact_module()
  if sha(QPATH)!=Q_PIN:raise RuntimeError('pinned sparse LP helper changed')
  q=loadmod('b65_99_fit_witness_lp_helper',QPATH)
  T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
  bmask=(ids[:,0]==65)&(ids[:,1]==99);bone=np.asarray(T[bmask],np.float64);bone_ids=np.asarray(ids[bmask,3],np.int64)
  bmap={int(v):i for i,v in enumerate(bone_ids)}
  bn=np.cross(bone[:,1]-bone[:,0],bone[:,2]-bone[:,0]);bl=np.linalg.norm(bn,axis=1)
  if len(bone)!=342 or np.any(bl==0):raise RuntimeError('invalid local bank triangle inventory')
  bn/=bl[:,None];volume6=float(sum(float(a@np.cross(b,c)) for a,b,c in bone));bn*=1 if volume6>0 else -1
  bkeys=[tuple(exact.float32_point_lattice_key(x) for x in tri) for tri in bone]
  edge_faces=defaultdict(list);vertex_faces=defaultdict(list)
  for bi,ks in enumerate(bkeys):
   for i,j in ((0,1),(1,2),(2,0)):edge_faces[tuple(sorted((ks[i],ks[j])))].append(bi)
   for k in ks:vertex_faces[k].append(bi)
  def normal_for(bary,bi):
   zero=[i for i,v in enumerate(bary) if v==0];inc={bi}
   if len(zero)>=2:
    i=next(i for i,v in enumerate(bary) if v==1);inc.update(vertex_faces[bkeys[bi][i]])
   elif len(zero)==1:
    ij=[i for i in range(3) if i!=zero[0]];inc.update(edge_faces[tuple(sorted((bkeys[bi][ij[0]],bkeys[bi][ij[1]])))])
   n=bn[sorted(inc)].sum(axis=0);ln=float(np.linalg.norm(n))
   if ln<1e-10:raise RuntimeError('cancelled adjacent bone normals at witness')
   return n/ln
  bone_records=audit_tool.records(bone,bone_ids,exact);prepared_bone=exact.prepare_point_location(bone_records)
  original_local=(P-pos[65])@M[65].T-off[65]
  orig_bank=next(b for b in auditdoc['banks'] if (int(b['body']),int(b['stable_bone_id']))==(65,99))
  original_pairs={(int(f),int(b)) for f,b in orig_bank['result']['triangle_pairs']}
  if len(original_pairs)!=148:raise RuntimeError('original comparable 65/99 pair set changed')
  # Exact rational cached witness anchors; the sub-ULP coordinates are never emitted.
  anchors=[];anchor_keys=set();endpoint_receipts=edoc['exact_intersections']['exact_pair_endpoint_feature_witnesses']
  for rec in endpoint_receipts:
   sf=int(rec['skin_face']);bid=int(rec['bone_triangle']);bi=bmap[bid];s_tri=exact_tri(original_local[F[sf]]);b_tri=exact_tri(bone[bi])
   for ord,raw in enumerate(rec['exact_local_endpoints_m']):
    q=exact_point(raw);sw=barycentric(q,s_tri,exact);bw=barycentric(q,b_tri,exact)
    if sum(sw)!=1 or min(sw)<0 or max(sw)>1:raise RuntimeError('cached exact skin barycentric is outside source face')
    if tuple(sum(sw[i]*s_tri[i][k] for i in range(3)) for k in range(3))!=q:raise RuntimeError('cached exact witness fails rational reconstruction')
    key=(sf,bid,tuple(map(str,q)))
    if key in anchor_keys:continue
    anchor_keys.add(key);anchors.append((np.asarray(F[sf],np.int64),np.asarray([float(x) for x in sw]),
      np.asarray([float(x) for x in q]),normal_for(bw,bi),sf,bid,tuple(map(str,q))))
  if len(anchors)<296:raise RuntimeError(f'cached exact witness inventory short: {len(anchors)}')
  # Baseline shape and exact inside classifications for the patch nodes.
  base_quality=quality(original_local,patchfaces,original_local)
  min_angle_floor=max(0.5,0.9*base_quality['minimum_angle_degrees'])
  aspect_cap=max(1.25*base_quality['maximum_aspect_measure'],base_quality['maximum_aspect_measure']+1.0)
  baseline_locations=Counter(exact.point_location_prepared(tuple(frac(x) for x in original_local[n]),prepared_bone)['location'] for n in patchnodes)
  baseline_inside_nodes=sorted(int(n) for n in patchnodes if exact.point_location_prepared(tuple(frac(x) for x in original_local[n]),prepared_bone)['location'] in ('inside','boundary'))
  OUT.mkdir(parents=True,exist_ok=True)
  report.update({'input_pins':{'candidate_sha256':sha(C),'audit159_sha256':sha(A),'witness_preflight_sha256':sha(PREFLIGHT),
    'cached_endpoint_receipt_sha256':sha(ENDPOINTS),'sparse_lp_helper_sha256':sha(QPATH),'bone_inventory_count':len(bone)},
   'topology':{'base_parent_face_count':len(base_rows),'measured_added_face_count':len(rows)-len(base_rows),
    'patch_face_rows':rows,'patch_face_rows_sha256':hashlib.sha256(np.asarray(rows,np.int64).tobytes()).hexdigest(),**topometa,
    'fixed_boundary_node_count':len(fixed),'movable_patch_node_count':len(movable),'all_faces_have_source_face_csr':True,
    'face_ids_and_csr_ancestry_preserved':True},
   'cached_witnesses':{'exact_pair_count':len(original_pairs),'deduplicated_rational_anchor_count':len(anchors),
    'distinct_exact_points':len(set(a[6] for a in anchors)),'endpoint_coordinates_emitted':False},
   'baseline':{'shape':base_quality,'minimum_angle_floor':min_angle_floor,'maximum_aspect_cap':aspect_cap,
    'patch_node_location_counts':dict(baseline_locations),'inside_or_boundary_node_ids':baseline_inside_nodes},
   'optimizer_parameters':{'trust_step_mm':TRUST_STEP_MM,'outer_iterations':MAX_OUTER_ITERS,
    'total_movement_cap_mm':MAX_TOTAL_MM,'area_floor':AREA_FLOOR,'shape_gate_min_angle_deg':min_angle_floor,
    'shape_gate_max_aspect':aspect_cap,'objective':'minimum incremental displacement with graph Laplacian smoothing'}})
  # Surface graph regularizer over movable nodes.
  vmap={int(n):i for i,n in enumerate(movable)};nv=len(movable);edges=set()
  for f in patchfaces:
   for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):edges.add(tuple(sorted((int(a),int(b)))))
  L=np.zeros((nv,nv),np.float64)
  for a,b in edges:
   ia=vmap.get(a);ib=vmap.get(b)
   if ia is not None:L[ia,ia]+=1
   if ib is not None:L[ib,ib]+=1
   if ia is not None and ib is not None:L[ia,ib]-=1;L[ib,ia]-=1
  H3=np.kron(np.eye(nv)+SMOOTH_WEIGHT*L,np.eye(3))
  pnorm=np.cross(original_local[patchfaces[:,1]]-original_local[patchfaces[:,0]],original_local[patchfaces[:,2]]-original_local[patchfaces[:,0]])
  plen=np.linalg.norm(pnorm,axis=1)
  if np.any(plen==0):raise RuntimeError('degenerate parent face in patch')
  punit=pnorm/plen[:,None]
  current_world=P32.copy();current_local=original_local.copy();all_constraints=list(anchors)
  rounds=[];selected=None
  for it in range(1,MAX_OUTER_ITERS+1):
   vlookup={int(n):i for i,n in enumerate(movable)}
   def unpack(y):
    x=current_local.copy();x[movable]+=np.asarray(y).reshape(nv,3)*SCALE_M;return x
   def half_fun(y):
    x=unpack(y);out=np.empty(len(all_constraints),np.float64)
    for k,(nodes,w,qpoint,n,*_) in enumerate(all_constraints):out[k]=(float(n@(np.sum(x[nodes]*w[:,None],axis=0)-qpoint))-CLEARANCE_MM/1000)/SCALE_M
    return out
   def shape_fun(y):
    x=unpack(y);out=np.empty(len(patchfaces),np.float64)
    for k,f in enumerate(patchfaces):
     a,b,c=x[f];cn=np.cross(b-a,c-a);out[k]=float(punit[k]@cn)/plen[k]-AREA_FLOOR
    return out
   # Sparse HiGHS LP: hard first-order parent-area and edge-gradient rows; each
   # exact witness gets an explicit bounded nonnegative clearance slack.
   nvar=3*nv;nwit=len(all_constraints);nabs=nvar;nfull=2*nvar+nwit
   ar=[];ac=[];av=[];area_rhs=[]
   for k,f in enumerate(patchfaces):
    ia,ib,ic=map(int,f);a,b,c=current_local[ia],current_local[ib],current_local[ic];pn=punit[k];den=plen[k]
    ratio=float(pn@np.cross(b-a,c-a))/den;area_rhs.append(ratio-AREA_FLOOR)
    grads=(np.cross(pn,c-b),np.cross(c-a,pn),np.cross(pn,b-a))
    for node,g in zip((ia,ib,ic),grads):
     j=vlookup.get(node)
     if j is not None:
      for axis in range(3):
       val=-SCALE_M*float(g[axis])/den
       if val:ar.append(k);ac.append(3*j+axis);av.append(val)
   Aarea=sparse.coo_matrix((av,(ar,ac)),shape=(len(patchfaces),nvar)).tocsr()
   wr=[];wc=[];wv=[];wreq=[]
   for k,(nodes,w,qpoint,n,*_) in enumerate(all_constraints):
    current_mm=1000.0*float(n@(np.sum(current_local[nodes]*w[:,None],axis=0)-qpoint))
    wreq.append(CLEARANCE_MM-current_mm)
    for node,weight in zip(nodes,w):
     j=vlookup.get(int(node))
     if j is not None:
      for axis in range(3):
       val=float(weight)*float(n[axis])
       if val:wr.append(k);wc.append(3*j+axis);wv.append(val)
   Gw=sparse.coo_matrix((wv,(wr,wc)),shape=(nwit,nvar)).tocsr()
   points_mm=current_local*1000.0
   EG,eb,edge_meta=q.build_edge_gradient_constraints(points_mm,patchfaces,movable,0.5)
   Zaa=sparse.csr_matrix((len(patchfaces),nvar));Zam=sparse.csr_matrix((len(patchfaces),nwit))
   Zen=sparse.csr_matrix((EG.shape[0],nvar));Zem=sparse.csr_matrix((EG.shape[0],nwit))
   Zw=sparse.csr_matrix((nwit,nvar))
   I=sparse.eye(nvar,format='csr');Is=sparse.eye(nwit,format='csr')
   Aarea_lp=sparse.hstack((Aarea,Zaa,Zam),format='csr')
   Awitness_lp=sparse.hstack((-Gw,Zw,-Is),format='csr')
   Aedge_lp=sparse.hstack((-EG,Zen,Zem),format='csr')
   Aabs_pos=sparse.hstack((I,-I,sparse.csr_matrix((nvar,nwit))),format='csr')
   Aabs_neg=sparse.hstack((-I,-I,sparse.csr_matrix((nvar,nwit))),format='csr')
   A_ub=sparse.vstack((Aarea_lp,Awitness_lp,Aedge_lp,Aabs_pos,Aabs_neg),format='csr')
   b_ub=np.r_[np.asarray(area_rhs),-np.asarray(wreq),-np.asarray(eb),np.zeros(2*nvar)]
   objective=np.r_[np.zeros(nvar),np.ones(nabs),np.full(nwit,1000.0)]
   bounds=[(-TRUST_STEP_MM,TRUST_STEP_MM)]*nvar+[(0.0,TRUST_STEP_MM)]*nabs+[(0.0,CLEARANCE_MM)]*nwit
   lp=linprog(objective,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs',options={'time_limit':30.0,'presolve':True})
   if lp.success:
    delta=np.asarray(lp.x[:nvar],np.float64);slacks=np.asarray(lp.x[2*nvar:],np.float64)
   else:
    delta=np.zeros(nvar,np.float64);slacks=np.zeros(nwit,np.float64)
   res=SimpleNamespace(success=bool(lp.success),message=str(lp.message),nit=int(lp.nit or 0),x=delta,fun=float(lp.fun) if lp.fun is not None else None)
   slack_positive=slacks>1e-7
   rec={'iteration':it,'solver':'HiGHS sparse LP','solver_success':bool(lp.success),'solver_status':int(lp.status),
    'message':str(lp.message),'iterations':int(lp.nit or 0),'LP_variables':int(nfull),'LP_rows':int(A_ub.shape[0]),
    'witness_constraint_count':nwit,'shape_constraint_count':len(patchfaces),'linearized_area_rows_hard':int(len(patchfaces)),
    'hard_edge_gradient_rows':int(EG.shape[0]),'edge_gradient':edge_meta,
    'initial_min_witness_scaled':float(half_fun(np.zeros(nvar)).min()),
    'final_min_witness_scaled':float(half_fun(delta).min()),
    'final_min_double_shape_constraint':float(shape_fun(delta).min()),'objective':res.fun,
    'witness_slack_positive_count':int(slack_positive.sum()),'witness_slack_sum_mm':float(slacks.sum()),
    'witness_slack_max_mm':float(slacks.max(initial=0.0)),
    'witnesses_predicted_full_clearance':int((~slack_positive).sum())}
   if not lp.success:
    rec['status']='HIGHS_FAILED';rounds.append(rec);report['status']='PRIVATE_REJECTED_HIGHS';report['rounds']=rounds;break
   if not res.success:
    rec['status']='HIGHS_FAILED';rounds.append(rec);report['status']='PRIVATE_REJECTED_HIGHS';report['rounds']=rounds;break
   proposed=unpack(res.x)
   total_disp=np.linalg.norm(proposed[patchnodes]-original_local[patchnodes],axis=1)
   rec['max_cumulative_displacement_mm']=float(total_disp.max()*1000)
   rec['rms_cumulative_displacement_mm']=float(np.sqrt(np.mean(total_disp**2))*1000)
   if rec['max_cumulative_displacement_mm']>MAX_TOTAL_MM+1e-6:
    rec['status']='CUMULATIVE_MOVEMENT_CAP_FAILED';rounds.append(rec);report['status']='PRIVATE_REJECTED_MOVEMENT_CAP';report['rounds']=rounds;break
   if rec['final_min_double_shape_constraint'] < -1e-8:
    rec['status']='DOUBLE_SHAPE_FAILED';rounds.append(rec);report['status']='PRIVATE_REJECTED_DOUBLE_SHAPE';report['rounds']=rounds;break
   # Float32 emit only at actual existing skin nodes. Fixed outer boundary is bitwise retained.
   world=current_world.copy()
   world[movable]=((proposed[movable]+off[65])@M[65]+pos[65]).astype(np.float32)
   world[fixed]=P32[fixed]
   emitted_local=(world.astype(np.float64)-pos[65])@M[65].T-off[65]
   qshape=quality(emitted_local,patchfaces,original_local)
   shape_pass=bool(qshape['orientation_positive'] and qshape['minimum_unsigned_parent_area_ratio']>=AREA_FLOOR and
    qshape['minimum_angle_degrees']>=min_angle_floor and qshape['maximum_aspect_measure']<=aspect_cap)
   boundary_same=bool(np.array_equal(world[fixed].view(np.uint32),P32[fixed].view(np.uint32)))
   rec.update({'emitted_f32_shape':qshape,'emitted_shape_pass':shape_pass,'fixed_boundary_bitwise_preserved':boundary_same})
   round_path=OUT/f'candidate-private-iteration-{it:02d}.npz'
   if shape_pass and boundary_same:
    emitted_z={k:v for k,v in Z.items()};emitted_z['points_world_m']=world
    np.savez_compressed(round_path,**emitted_z);rec.update({'candidate_path':str(round_path),'candidate_sha256':sha(round_path)})
   if not shape_pass or not boundary_same:
    rec['status']='EMITTED_SHAPE_OR_BOUNDARY_FAILED';rounds.append(rec);report['status']='PRIVATE_REJECTED_EMITTED_SHAPE_OR_BOUNDARY';report['rounds']=rounds;break
   # Exact local patch-versus-target-bone screen on the emitted Float32 nodes.
   patch_records=mesh_records(emitted_local,patchfaces,rows,np.arange(len(P),dtype=np.int64),exact)
   bgate=exact._audit_pair(patch_records,bone_records,same_surface=False)
   cur_pairs={(int(f),int(b)) for f,b in bgate['triangle_pairs']}
   added=sorted(cur_pairs-original_pairs);removed=sorted(original_pairs-cur_pairs)
   rec.update({'exact_local_bone_child_pair_count':int(bgate['count']),'parent_identity_pair_count':len(cur_pairs),
    'parent_identity_added_vs_original':len(added),'parent_identity_removed_vs_original':len(removed),
    'net_parent_identity_delta_vs_original':len(cur_pairs)-len(original_pairs),
    'parent_identity_improved':len(cur_pairs)<len(original_pairs),'exact_local_bone_pairs':bgate['triangle_pairs']})
   loc=Counter(exact.point_location_prepared(tuple(frac(x) for x in emitted_local[n]),prepared_bone)['location'] for n in patchnodes)
   loc_inside=sorted(int(n) for n in patchnodes if exact.point_location_prepared(tuple(frac(x) for x in emitted_local[n]),prepared_bone)['location'] in ('inside','boundary'))
   new_inside=sorted(set(loc_inside)-set(baseline_inside_nodes))
   rec.update({'patch_node_location_counts':dict(loc),'inside_or_boundary_node_ids':loc_inside,'new_inside_or_boundary_node_ids':new_inside})
   rounds.append(rec);current_world=world;current_local=emitted_local
   if len(cur_pairs)<len(original_pairs):
    # Local self check only follows strict identity improvement.
    comp=np.asarray([r for r in range(len(F)) if r not in set(rows)],np.int64)
    patch_tri=emitted_local[patchfaces];lo=patch_tri.min(axis=(0,1));hi=patch_tri.max(axis=(0,1))
    ct=emitted_local[F[comp]];cl=ct.min(axis=1);ch=ct.max(axis=1)
    active=np.flatnonzero(np.all(ch>=lo,axis=1)&np.all(cl<=hi,axis=1));comp_rows=comp[active]
    comp_records=mesh_records(emitted_local,F[comp_rows],comp_rows,np.arange(len(P),dtype=np.int64),exact)
    self_internal=exact._audit_pair(patch_records,patch_records,same_surface=True)
    self_comp=exact._audit_pair(patch_records,comp_records,same_surface=True)
    self_gate={'patch_internal_pair_count':int(self_internal['count']),'patch_internal_pairs':self_internal['triangle_pairs'],
      'complement_aabb_active_face_count':int(len(comp_rows)),'patch_vs_complement_pair_count':int(self_comp['count']),
      'patch_vs_complement_pairs':self_comp['triangle_pairs'],'pass':int(self_internal['count'])==0 and int(self_comp['count'])==0}
    selected={'iteration':it,'world':world,'local':emitted_local,'path':str(round_path),'sha256':sha(round_path)}
    report.update({'status':'PRIVATE_PARENT_IDENTITY_IMPROVEMENT_SCREENED','exact_local_bone_gate':{'pair_count':int(bgate['count']),
      'pairs':bgate['triangle_pairs'],'parent_identity_pair_count':len(cur_pairs),'added_vs_original':[list(x) for x in added],
      'removed_vs_original':[list(x) for x in removed],'net_identity_delta':len(cur_pairs)-len(original_pairs),
      'strict_improvement':True},'sampled_inside_comparison':{'scope':'all unique nodes in the 248-face patch',
      'baseline_inside_or_boundary_node_ids':baseline_inside_nodes,'final_inside_or_boundary_node_ids':loc_inside,
      'new_inside_or_boundary_node_ids':new_inside,'no_new_inside_or_boundary_witnesses':not new_inside},
      'exact_local_self_gate':self_gate,'selected_private_candidate_path':str(round_path),'selected_private_candidate_sha256':sha(round_path)})
    if not self_gate['pass'] or new_inside:
     report['status']='PRIVATE_REJECTED_NEW_LOCAL_SELF_OR_INSIDE_WITNESS'
    break
   if it==MAX_OUTER_ITERS:
    report['status']='PRIVATE_REJECTED_NO_PARENT_IDENTITY_IMPROVEMENT';break
   # Iterative correction: new exact intersections on current Float32 geometry become new barycentric constraints.
   new_n=0
   for sf,bid in sorted(cur_pairs):
    bi=bmap.get(bid)
    if bi is None:raise RuntimeError(f'unknown stable bone id after exact screen: {bid}')
    ts=exact_tri(emitted_local[F[sf]]);tb=exact_tri(bone[bi])
    hits=sorted(set(exact.triangle_intersection_points(ts,tb)))
    if not hits:raise RuntimeError(f'exact pair {(sf,bid)} had no reconstructed current intersection points')
    for hit in hits:
     sw=barycentric(hit,ts,exact);bw=barycentric(hit,tb,exact)
     if sum(sw)!=1 or min(sw)<0:raise RuntimeError('current exact witness barycentric outside skin face')
     key=(int(sf),int(bid),tuple(map(str,hit)))
     if key in anchor_keys:continue
     anchor_keys.add(key);all_constraints.append((np.asarray(F[sf],np.int64),np.asarray([float(x) for x in sw]),
       np.asarray([float(x) for x in hit]),normal_for(bw,bi),int(sf),int(bid),tuple(map(str,hit))))
     new_n+=1
   rec['new_exact_witness_constraints_added_for_next_step']=new_n
   if new_n==0:
    report['status']='PRIVATE_REJECTED_NO_NEW_WITNESSES_FOR_ITERATION';break
  report['rounds']=rounds
  if selected:report['candidate_modified']=False
 except Exception as e:
  report.update({'status':report.get('status') if report.get('status')!='STARTED' else 'PRIVATE_WITNESS_FIT_ABORTED',
    'failure':repr(e),'traceback':traceback.format_exc()})
 report['elapsed_seconds']=time.monotonic()-started;report['script_path']=str(Path(__file__));report['script_sha256']=sha(Path(__file__))
 rp=OUT/'report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('traceback','rounds','exact_local_bone_gate','sampled_inside_comparison','exact_local_self_gate')},indent=2,allow_nan=False))
 print('report_path',rp,'report_sha256',sha(rp))
 if report.get('rounds'):
  for r in report['rounds']:
   print('ROUND',json.dumps({k:v for k,v in r.items() if k not in ('exact_local_bone_pairs','inside_or_boundary_node_ids','new_inside_or_boundary_node_ids')},sort_keys=True))

main()
