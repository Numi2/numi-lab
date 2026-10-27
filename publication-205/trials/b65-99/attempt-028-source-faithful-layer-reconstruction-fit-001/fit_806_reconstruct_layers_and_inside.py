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
C=S/'per-bank-workers-20261027/combined-d071-b109-b61-001/candidate.npz'
A=C.parent/'audit-159/report.json'
TOPO=S/'per-bank-workers-20261027/b65-99/attempt-002/fan-hole-ancestry-check.json'
ENDPOINTS=S/'per-bank-workers-20261027/b65-99/attempt-026-806-connected-witness-fit-001/current-exact-endpoints.json'
PREFLIGHT=S/'per-bank-workers-20261027/b65-99/attempt-004-bone-aware-connected-patch-prep-001/witness-preflight.json'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=ROOT/'audit_skin_bone_surfaces_159.py'
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
Q_PIN='5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
OUT=Path(__file__).resolve().parent/'fit-run-001'
LAYER_FILE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/combined-d071-b109-b61-001/skin-fat-layers-541.npz')
LAYER_PIN='eb41bf0d1c11f1ee0a94672058063c51350e8e0dca1483a6ae7568cb03a78b57'
LAYER_REPORT=LAYER_FILE.with_name('shape-layer-report.json')
LAYER_REPORT_PIN='dbd46128636b61740ca45ed4e6061acef6cbb94b00e9564676c7a5a26e318306'
RECON_OWNER=ROOT/'connected-layer-reconstruction-127/matter/tools/human_skin_fat_layer_reconstruction.py'
RECON_PIN='2a7f7ed02a5c90af11e09ffd8b8d2a34b2f3ff9c3b08ecb08514cccb29dc565e'
TRANSFER_OWNER=S/'per-bank-workers-20261027/b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py'
TRANSFER_PIN='fb53b372fd534a1f81e29944a1455ef249b674d4c8c043940fca442c3a7aad51'
INSIDE130=C.parent/'audit-130/summary.json'
SELF_PATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/combined-d071-b109-b61-001/audit-130/surface-self-pairs.npy')
SELF_PIN='55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de'
C_PIN='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
A_PIN='4cc8fd5d5e8fb651733fd380001e2b9847a0040960d7c8865c37f25fe5b8a09d'
PREFLIGHT_PIN='579ab33b2a1922af8075d7825b93a2a132e0f84aaf072b85c56a2196f9083855'
CLEARANCE_MM=0.05
SCALE_M=0.001
TRUST_STEP_MM=2.5
MAX_TOTAL_MM=3.0
MAX_OUTER_ITERS=1
BACKTRACK_FACTORS=tuple(1.0/(2.0**k) for k in range(17))
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


def make_layer_guard(layer_file, faces, affected, support, world_gradient_to_local):
 with np.load(layer_file,allow_pickle=False) as z:
  if not np.array_equal(z['outer_faces'],faces[:,[0,2,1]]):raise RuntimeError('transferred layer orientation/topology does not match candidate')
  planes=[z['outer_points_f64']*1000.,z['inner_points_f64']*1000.,z['deep_points_f64']*1000.]
  oriented=z['outer_faces']
 local={int(node):i for i,node in enumerate(support)}
 R=[];C=[];V=[];bounds=[];skipped=0;nonpositive=0
 for fi in affected:
  original=oriented[int(fi)];a,b,c=sorted(map(int,original))
  parity=(int(original[0]>original[1])+int(original[0]>original[2])+int(original[1]>original[2]))&1
  expected=1. if parity else -1.
  for li in (0,1):
   out,inn=planes[li],planes[li+1]
   tetra=[([(a,out[a]),(b,out[b]),(c,out[c]),(c,inn[c])],expected),
          ([(a,out[a]),(b,out[b]),(b,inn[b]),(c,inn[c])],-expected),
          ([(a,out[a]),(a,inn[a]),(b,inn[b]),(c,inn[c])],expected)]
   for vertices,sign in tetra:
    p0,p1,p2,p3=[v[1] for v in vertices]
    u=p1-p0;v=p2-p0;w=p3-p0
    signed=float(sign*np.dot(u,np.cross(v,w))/6.0)
    grads=[None,sign*np.cross(v,w)/6.0,sign*np.cross(w,u)/6.0,sign*np.cross(u,v)/6.0]
    grads[0]=-grads[1]-grads[2]-grads[3]
    combined={}
    for (node,_),grad in zip(vertices,grads):
     if node in local:combined[node]=combined.get(node,np.zeros(3))+grad
    combined={node:grad@world_gradient_to_local for node,grad in combined.items()}
    nrm=float(np.sqrt(sum(np.dot(g,g) for g in combined.values())))
    if nrm<=1e-13:skipped+=1;continue
    if signed<=0:nonpositive+=1
    lower=(-.8*signed if signed>0 else 0.0)/nrm
    row=len(bounds);bounds.append(lower)
    for node,grad in combined.items():
     for axis in range(3):
      if grad[axis]:R.append(row);C.append(3*local[node]+axis);V.append(grad[axis]/nrm)
 mat=sparse.coo_matrix((V,(R,C)),shape=(len(bounds),3*len(support))).tocsr()
 return mat,np.asarray(bounds),{'rows':len(bounds),'affected_face_count':len(affected),'baseline_nonpositive_rows':nonpositive,'skipped_fixed_or_zero_rows':skipped}

def layer_emission_nonregression(layer_file, baseline_world_f32, emitted_world_f32, faces, affected):
 with np.load(layer_file,allow_pickle=False) as z:
  outer0=np.asarray(z['outer_points_f32'],np.float32).astype(np.float64)*1000.0
  inner0=np.asarray(z['inner_points_f32'],np.float32).astype(np.float64)*1000.0
  deep0=np.asarray(z['deep_points_f32'],np.float32).astype(np.float64)*1000.0
  inner0_f64=np.asarray(z['inner_points_f64'],np.float64)
  deep0_f64=np.asarray(z['deep_points_f64'],np.float64)
  oriented=np.asarray(z['outer_faces'],np.int64)
 delta_m=np.asarray(emitted_world_f32,np.float32).astype(np.float64)-np.asarray(baseline_world_f32,np.float32).astype(np.float64)
 outer1=np.asarray(emitted_world_f32,np.float32).astype(np.float64)*1000.0
 inner1=((inner0_f64+delta_m).astype(np.float32).astype(np.float64))*1000.0
 deep1=((deep0_f64+delta_m).astype(np.float32).astype(np.float64))*1000.0
 checks=[]
 for li,(o0,i0,o1,i1) in enumerate(((outer0,inner0,outer1,inner1),(inner0,deep0,inner1,deep1))):
  for fi in affected:
   original=oriented[int(fi)];a,b,c=sorted(map(int,original))
   parity=(int(original[0]>original[1])+int(original[0]>original[2])+int(original[1]>original[2]))&1
   expected=1.0 if parity else -1.0
   tets=[([(a,o0[a]),(b,o0[b]),(c,o0[c]),(c,i0[c])],[(a,o1[a]),(b,o1[b]),(c,o1[c]),(c,i1[c])],expected),
         ([(a,o0[a]),(b,o0[b]),(b,i0[b]),(c,i0[c])],[(a,o1[a]),(b,o1[b]),(b,i1[b]),(c,i1[c])],-expected),
         ([(a,o0[a]),(a,i0[a]),(b,i0[b]),(c,i0[c])],[(a,o1[a]),(a,i1[a]),(b,i1[b]),(c,i1[c])],expected)]
   for oldpts,newpts,sign in tets:
    def vol(ps):
     p0,p1,p2,p3=[p for _,p in ps]
     return float(sign*np.dot(p1-p0,np.cross(p2-p0,p3-p0))/6.0)
    old,new=vol(oldpts),vol(newpts)
    floor=.2*old if old>0 else old
    checks.append((li,int(fi),old,new,new-floor,old>0))
 margins=np.asarray([v[4] for v in checks],np.float64)
 tol=1e-9
 violations=[{'layer':'skin' if li==0 else 'fat','face_row':fi,'old_oriented_volume_mm3':old,'new_oriented_volume_mm3':new,'required_min_mm3':new-margin,'margin_mm3':margin,'baseline_positive':positive}
             for li,fi,old,new,margin,positive in checks if margin < -tol]
 positive=[q for q in checks if q[5]];bad=[q for q in checks if not q[5]]
 return {'pass':not violations,'checked_tetrahedra':len(checks),'tolerance_mm3':tol,'violation_count':len(violations),
  'first_violations':violations[:20],'minimum_margin_mm3':float(margins.min(initial=np.inf)),
  'baseline_positive_count':len(positive),'baseline_nonpositive_count':len(bad),
  'minimum_baseline_positive_new_to_old_ratio':float(min((q[3]/q[2] for q in positive),default=1.0)),
  'worst_baseline_nonpositive_change_mm3':float(min((q[3]-q[2] for q in bad),default=0.0))}


def point_triangle_distances(points,triangles):
    P=np.asarray(points,dtype=np.float64).reshape(-1,3)
    T=np.asarray(triangles,dtype=np.float64).reshape(-1,3,3)
    out=np.empty(len(P),dtype=np.float64);nearest=np.empty(len(P),dtype=np.int64)
    A=T[:,0];B=T[:,1];C=T[:,2];AB=B-A;AC=C-A
    d00=np.einsum('ij,ij->i',AB,AB);d01=np.einsum('ij,ij->i',AB,AC);d11=np.einsum('ij,ij->i',AC,AC)
    den=d00*d11-d01*d01
    if np.any(den<=0): raise RuntimeError('degenerate target bone triangle in distance query')
    for k,p in enumerate(P):
        AP=p-A;d20=np.einsum('ij,ij->i',AP,AB);d21=np.einsum('ij,ij->i',AP,AC)
        u=(d11*d20-d01*d21)/den;v=(d00*d21-d01*d20)/den
        inside=(u>=0)&(v>=0)&(u+v<=1)
        plane=A+u[:,None]*AB+v[:,None]*AC
        d2=np.einsum('ij,ij->i',plane-p,plane-p)
        d2[~inside]=np.inf
        for X,Y in ((A,B),(B,C),(C,A)):
            E=Y-X;ee=np.einsum('ij,ij->i',E,E)
            t=np.clip(np.einsum('ij,ij->i',p-X,E)/ee,0.0,1.0)
            Q=X+t[:,None]*E;ed2=np.einsum('ij,ij->i',Q-p,Q-p)
            d2=np.minimum(d2,ed2)
        nearest[k]=int(np.argmin(d2));out[k]=float(np.sqrt(d2[nearest[k]]))
    return out,nearest

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 report={'schema':'numi.human.bank65_99.layer-guarded-exact-witness-fit.v1','status':'STARTED','input_note':'Current 806 exact pair endpoints were rebuilt from audit-159 on the pinned Float32 baseline; target constraints use all 148 current child pairs and 84 immediate-parent identities. Existing layer guard covers every affected transferred skin/fat prism, including baseline invalid identities; exact local self gate is run if and only if strict mapped-pair reduction with no new mapped identities occurs.',
  'candidate_path':str(C),'geometry_or_native_admission':False,'candidate_modified':False,
  'scope':'Private one-step 248-face disk fit on current 806; 148 exact crossing pairs plus 29 audited inside-node closest-plane constraints; area/edge/trust and fixed-boundary rows are hard; after each emitted Float32 outer trial, reconstruct local O/I/D with the pinned owner, exact fixed boundary layers and transferred scalar thickness fields; require emitted shape and changed-cell nonregression before exact target screen. Candidate remains private diagnostic only.',
  'limits':{'trust_step_mm':TRUST_STEP_MM,'max_outer_iterations':MAX_OUTER_ITERS,'max_cumulative_node_motion_mm':MAX_TOTAL_MM,
    'clearance_halfspace_mm':CLEARANCE_MM,'shape_area_ratio_floor':AREA_FLOOR,
    'backtrack_trial_count':len(BACKTRACK_FACTORS),'minimum_backtrack_factor':min(BACKTRACK_FACTORS)}}
 try:
  if sha(C)!=C_PIN or sha(A)!=A_PIN or sha(PREFLIGHT)!=PREFLIGHT_PIN:raise RuntimeError('candidate/audit/preflight pin mismatch')
  if sha(LAYER_FILE)!=LAYER_PIN or sha(LAYER_REPORT)!=LAYER_REPORT_PIN or sha(SELF_PATH)!=SELF_PIN:raise RuntimeError('candidate-aligned layer/self evidence pin mismatch')
  lreport=json.loads(LAYER_REPORT.read_text())
  baseline_self_pairs=np.load(SELF_PATH,allow_pickle=False)
  if baseline_self_pairs.ndim!=2 or baseline_self_pairs.shape[1]!=2:raise RuntimeError('baseline self-pair witness artifact malformed')
  if len(baseline_self_pairs)!=0:raise RuntimeError('current 806 baseline is not self-clear; require explicit self witness constraints before this fit')
  if lreport.get('shape',{}).get('candidate_sha256')!=C_PIN or lreport.get('layer',{}).get('candidate_layer_sha256')!=LAYER_PIN:raise RuntimeError('shape/layer report is not anchored to the pinned 806 candidate/layer')
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
  patchfaces=F[np.asarray(rows,np.int64)];patchnodes=np.unique(patchfaces.reshape(-1))
  anchor_rows=np.asarray(Z['typed_coarse_anchor_output_rows_018'],np.int64)
  anchor_overlap=np.intersect1d(patchnodes,anchor_rows).astype(np.int64)
  fixed=np.asarray(sorted(set(map(int,boundary_nodes))|set(map(int,anchor_overlap))),np.int64)
  node_set=set(map(int,patchnodes));fixed_set=set(map(int,fixed));movable=np.asarray(sorted(node_set-fixed_set),np.int64)
  affected_rows=np.flatnonzero(np.any(np.isin(F,movable),axis=1)).astype(np.int64);affected_set=set(map(int,affected_rows));shape_faces=F[affected_rows]
  helper=loadmod('b65_99_fit_bank_helper',HELPER);audit_tool=loadmod('b65_99_fit_audit_owner',AUDIT_TOOL);exact=audit_tool.exact_module()
  if sha(QPATH)!=Q_PIN:raise RuntimeError('pinned sparse LP helper changed')
  lp_helper=loadmod('b65_99_fit_witness_lp_helper',QPATH)
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
  anchors=[];anchor_keys=set();endpoint_receipts=edoc['exact_pair_endpoint_feature_witnesses']
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
  if len(anchors)<len(original_pairs):raise RuntimeError(f'current exact witness inventory short: {len(anchors)}')
  # Pin and load the existing local O/I/D reconstruction and cell-flag owners.
  if sha(RECON_OWNER)!=RECON_PIN or sha(TRANSFER_OWNER)!=TRANSFER_PIN:raise RuntimeError('layer owner pin mismatch')
  recon_owner=loadmod('b65_99_reconstruction_owner',RECON_OWNER)
  transfer_owner=loadmod('b65_99_layer_flag_owner',TRANSFER_OWNER)
  with np.load(LAYER_FILE,allow_pickle=False) as lz:
   layer_base_O=np.asarray(lz['outer_points_f64'],np.float64).copy()
   layer_base_I=np.asarray(lz['inner_points_f64'],np.float64).copy()
   layer_base_D=np.asarray(lz['deep_points_f64'],np.float64).copy()
   layer_skin_field=np.asarray(lz['skin_thickness_transferred_m'],np.float64).copy()
   layer_fat_field=np.asarray(lz['fat_thickness_transferred_m'],np.float64).copy()
   layer_oriented=np.asarray(lz['outer_faces'],np.int64).copy()
  if not np.array_equal(layer_base_O.astype(np.float32),P32):raise RuntimeError('layer outer Float32 is not pinned to candidate 806')
  if not np.array_equal(layer_oriented,F[:,[0,2,1]]):raise RuntimeError('layer topology/winding differs from pinned candidate')
  if layer_base_I.shape!=P.shape or layer_base_D.shape!=P.shape:raise RuntimeError('baseline layer point dimensions mismatch')
  # Add outward-plane constraints from every audited 65/99 inside-node closest witness.
  inside_doc=json.load(open(INSIDE130))
  if inside_doc.get('candidate_sha256')!=C_PIN:raise RuntimeError('inside audit is not pinned to current 806')
  inside_witnesses=[w for w in inside_doc['inside_witnesses'] if int(w['bank_body'])==65 and int(w['bank_stable_id'])==99]
  inside_constraints=[];inside_targets=[]
  patchnode_set=set(map(int,patchnodes))
  for w in inside_witnesses:
   node=int(w['output_node']);bid=int(w['nearest_triangle'])
   if node not in patchnode_set:raise RuntimeError(f'inside node {node} is outside the connected patch')
   bi=bmap.get(bid)
   if bi is None:raise RuntimeError(f'nearest bone triangle {bid} has no stable-ID mapping')
   qpoint=np.asarray([float(w['closest_'+c]) for c in 'xyz'],np.float64)
   qquery=np.asarray([float(w['query_local_'+c]) for c in 'xyz'],np.float64)
   if np.linalg.norm(qquery-original_local[node])>1e-9:raise RuntimeError(f'body-local inside query mismatch for node {node}')
   n=bn[bi].copy()
   projected=1000.0*float(n@(original_local[node]-qpoint))
   reported=1000.0*float(w['signed_face_distance_m'])
   if projected>=0 or abs(projected-reported)>0.02:raise RuntimeError(f'inside signed-depth/face-normal parity failed for node {node}: projected={projected} reported={reported}')
   inside_constraints.append((np.asarray([node],np.int64),np.asarray([1.0]),qpoint,n,-1,bid,('inside-node',node,bid)))
   inside_targets.append({'node':node,'nearest_bone_triangle':bid,'baseline_signed_plane_depth_mm':projected,
      'baseline_audit_nearest_distance_mm':1000.0*float(w['nearest_distance_m']),'closest_local_m':qpoint.tolist()})
  if len(inside_constraints)!=29:raise RuntimeError(f'expected 29 target-bank inside rows, found {len(inside_constraints)}')
  # Baseline shape and exact inside classifications for the patch nodes.
  base_quality=quality(original_local,shape_faces,original_local)
  min_angle_floor=max(0.5,0.9*base_quality['minimum_angle_degrees'])
  aspect_cap=max(1.25*base_quality['maximum_aspect_measure'],base_quality['maximum_aspect_measure']+1.0)
  baseline_locations=Counter(exact.point_location_prepared(tuple(frac(x) for x in original_local[n]),prepared_bone)['location'] for n in patchnodes)
  baseline_inside_nodes=sorted(int(n) for n in patchnodes if exact.point_location_prepared(tuple(frac(x) for x in original_local[n]),prepared_bone)['location'] in ('inside','boundary'))
  OUT.mkdir(parents=True,exist_ok=True)
  report.update({'input_pins':{'candidate_sha256':sha(C),'audit159_sha256':sha(A),'witness_preflight_sha256':sha(PREFLIGHT),'shape_layer_report_sha256':sha(LAYER_REPORT),
    'current_exact_endpoint_receipt_sha256':sha(ENDPOINTS),'inside130_sha256':sha(INSIDE130),'reconstruction_owner_sha256':sha(RECON_OWNER),'transfer_flag_owner_sha256':sha(TRANSFER_OWNER),'sparse_lp_helper_sha256':sha(QPATH),'bone_inventory_count':len(bone),'current_endpoint_receipt_sha256':sha(ENDPOINTS),'current_pair_manifest_sha256':sha(ENDPOINTS.with_name('current-pair-manifest.json')),'layer_file_sha256':sha(LAYER_FILE),'layer_report_sha256':sha(LAYER_REPORT),'baseline_self_pair_artifact_sha256':sha(SELF_PATH)},
   'baseline_guard_witnesses':{'surface_self_pair_count':int(len(baseline_self_pairs)),'surface_self_pair_path':str(SELF_PATH),'layer_baseline_failure_counts':{layer: {kind: int(counts.get('before_count',0)) for kind,counts in kinds.items()} for layer,kinds in lreport.get('inherited_d071_12_face_layer_comparison',{}).get('per_layer',{}).items()},'layer_guard_scope':'local O/I/D reconstruction after emitted-Float32 outer fit; compare authored, nonpositive and conditioned prism identities on every face incident to the active patch; no cell-mass mapping or physical admission'},
   'topology':{'base_parent_face_count':len(base_rows),'measured_added_face_count':len(rows)-len(base_rows),
    'patch_face_rows':rows,'patch_face_rows_sha256':hashlib.sha256(np.asarray(rows,np.int64).tobytes()).hexdigest(),**topometa,
    'fixed_boundary_node_count':len(boundary_nodes),'fixed_patch_node_count':len(fixed),'typed_anchor_overlap_count':len(anchor_overlap),'movable_patch_node_count':len(movable),'changed_incident_face_row_count':len(affected_rows),'all_faces_have_source_face_csr':True,
    'face_ids_and_csr_ancestry_preserved':True},
   'cached_witnesses':{'exact_pair_count':len(original_pairs),'deduplicated_rational_anchor_count':len(anchors),
    'distinct_exact_points':len(set(a[6] for a in anchors)),'endpoint_coordinates_emitted':False,'inside_node_constraint_count':len(inside_constraints),'inside_node_constraints':inside_targets},
   'baseline':{'shape':base_quality,'minimum_angle_floor':min_angle_floor,'maximum_aspect_cap':aspect_cap,
    'patch_node_location_counts':dict(baseline_locations),'inside_or_boundary_node_ids':baseline_inside_nodes},
   'optimizer_parameters':{'trust_step_mm':TRUST_STEP_MM,'outer_iterations':MAX_OUTER_ITERS,'inside_target_node_count':len(inside_constraints),'layer_method':'connected normal-depth O/I/D reconstruction anchored at the fixed 62-node patch boundary',
    'total_movement_cap_mm':MAX_TOTAL_MM,'area_floor':AREA_FLOOR,'shape_gate_min_angle_deg':min_angle_floor,
    'shape_gate_max_aspect':aspect_cap,'objective':'minimum incremental displacement with graph Laplacian smoothing'}})
  layer_changed_rows=np.flatnonzero(np.any(np.isin(F,patchnodes),axis=1)).astype(np.int64)
  base_skin_flags=transfer_owner.face_flags(layer_base_O,layer_base_I,layer_oriented,layer_skin_field)
  base_fat_flags=transfer_owner.face_flags(layer_base_I,layer_base_D,layer_oriented,layer_fat_field)
  layer_boundary=np.asarray(sorted(map(int,boundary_nodes)),np.int64)
  outside_patch=np.setdiff1d(np.arange(len(P),dtype=np.int64),patchnodes)
  def reconstruct_trial_layers(world_trial):
   O1=np.asarray(world_trial,np.float32).astype(np.float64)
   result=recon_owner.reconstruct_connected_layers(
    O1,layer_oriented,layer_base_I-layer_base_O,layer_base_D-layer_base_I,
    layer_boundary,O1[layer_boundary],layer_base_I[layer_boundary],layer_base_D[layer_boundary],
    skin_depth_m=layer_skin_field,fat_depth_m=layer_fat_field,orientation_sign=+1,
    active_vertex_indices=patchnodes)
   I64=np.asarray(result.inner_points,np.float64);D64=np.asarray(result.deep_points,np.float64)
   I32=I64.astype(np.float32).astype(np.float64);D32=D64.astype(np.float32).astype(np.float64)
   sf=transfer_owner.face_flags(O1,I32,layer_oriented[layer_changed_rows],layer_skin_field)
   ff=transfer_owner.face_flags(I32,D32,layer_oriented[layer_changed_rows],layer_fat_field)
   bs={k:np.asarray(v)[layer_changed_rows] for k,v in base_skin_flags.items()}
   bf={k:np.asarray(v)[layer_changed_rows] for k,v in base_fat_flags.items()}
   def category(new,old,rowids):
    add_auth=np.flatnonzero(np.asarray(new['authored_bad'])&~np.asarray(old['authored_bad']))
    rem_auth=np.flatnonzero(~np.asarray(new['authored_bad'])&np.asarray(old['authored_bad']))
    add_np=np.argwhere(np.asarray(new['nonpositive'])&~np.asarray(old['nonpositive']))
    rem_np=np.argwhere(~np.asarray(new['nonpositive'])&np.asarray(old['nonpositive']))
    add_cd=np.argwhere(np.asarray(new['conditioned'])&~np.asarray(old['conditioned']))
    rem_cd=np.argwhere(~np.asarray(new['conditioned'])&np.asarray(old['conditioned']))
    return {'authored_bad_added_face_rows':[int(rowids[i]) for i in add_auth],
      'authored_bad_removed_face_rows':[int(rowids[i]) for i in rem_auth],
      'nonpositive_added_face_tets':[[int(rowids[i]),int(j)] for i,j in add_np],
      'nonpositive_removed_face_tets':[[int(rowids[i]),int(j)] for i,j in rem_np],
      'conditioned_added_face_tets':[[int(rowids[i]),int(j)] for i,j in add_cd],
      'conditioned_removed_face_tets':[[int(rowids[i]),int(j)] for i,j in rem_cd],
      'pass':len(add_auth)==0 and len(add_np)==0 and len(add_cd)==0}
   sc=category(sf,bs,layer_changed_rows);fc=category(ff,bf,layer_changed_rows)
   active_skin=np.linalg.norm(I32[patchnodes]-O1[patchnodes],axis=1)
   active_fat=np.linalg.norm(D32[patchnodes]-I32[patchnodes],axis=1)
   metrics={'skin':sc,'fat':fc,'changed_layer_face_count':int(len(layer_changed_rows)),
    'active_node_count':int(len(patchnodes)),'fixed_boundary_count':int(len(layer_boundary)),
    'fixed_boundary_inner_bitwise':bool(np.array_equal(I32[layer_boundary],layer_base_I.astype(np.float32).astype(np.float64)[layer_boundary])),
    'fixed_boundary_deep_bitwise':bool(np.array_equal(D32[layer_boundary],layer_base_D.astype(np.float32).astype(np.float64)[layer_boundary])),
    'outside_patch_inner_unchanged':bool(np.array_equal(I32[outside_patch],layer_base_I.astype(np.float32).astype(np.float64)[outside_patch])),
    'outside_patch_deep_unchanged':bool(np.array_equal(D32[outside_patch],layer_base_D.astype(np.float32).astype(np.float64)[outside_patch])),
    'skin_normalized_det_min_changed_faces':float(np.min(sf['normalized_det'],initial=np.inf)),
    'fat_normalized_det_min_changed_faces':float(np.min(ff['normalized_det'],initial=np.inf)),
    'skin_actual_norm_depth_min_active_m':float(active_skin.min()),
    'skin_actual_norm_depth_max_active_m':float(active_skin.max()),
    'fat_actual_norm_depth_min_active_m':float(active_fat.min()),
    'fat_actual_norm_depth_max_active_m':float(active_fat.max()),
    'skin_scalar_field_max_abs_error_active_m':float(np.max(np.abs(active_skin-layer_skin_field[patchnodes]))),
    'fat_scalar_field_max_abs_error_active_m':float(np.max(np.abs(active_fat-layer_fat_field[patchnodes]))),
    'owner_diagnostics':result.diagnostics}
   metrics['pass']=bool(sc['pass'] and fc['pass'] and metrics['fixed_boundary_inner_bitwise'] and metrics['fixed_boundary_deep_bitwise'] and metrics['outside_patch_inner_unchanged'] and metrics['outside_patch_deep_unchanged'])
   return metrics,I32,D32
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
  pnorm=np.cross(original_local[shape_faces[:,1]]-original_local[shape_faces[:,0]],original_local[shape_faces[:,2]]-original_local[shape_faces[:,0]])
  plen=np.linalg.norm(pnorm,axis=1)
  if np.any(plen==0):raise RuntimeError('degenerate parent face in patch')
  punit=pnorm/plen[:,None]
  current_world=P32.copy();current_local=original_local.copy();all_constraints=list(anchors)+list(inside_constraints)
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
    x=unpack(y);out=np.empty(len(shape_faces),np.float64)
    for k,f in enumerate(shape_faces):
     a,b,c=x[f];cn=np.cross(b-a,c-a);out[k]=float(punit[k]@cn)/plen[k]-AREA_FLOOR
    return out
   # Sparse HiGHS LP: hard first-order parent-area and edge-gradient rows; each
   # exact witness gets an explicit bounded nonnegative clearance slack.
   nvar=3*nv;nwit=len(all_constraints);nabs=nvar;nfull=2*nvar+nwit
   ar=[];ac=[];av=[];area_rhs=[]
   for k,f in enumerate(shape_faces):
    ia,ib,ic=map(int,f);a,b,c=current_local[ia],current_local[ib],current_local[ic];pn=punit[k];den=plen[k]
    ratio=float(pn@np.cross(b-a,c-a))/den;area_rhs.append(ratio-AREA_FLOOR)
    grads=(np.cross(pn,c-b),np.cross(c-a,pn),np.cross(pn,b-a))
    for node,g in zip((ia,ib,ic),grads):
     j=vlookup.get(node)
     if j is not None:
      for axis in range(3):
       val=-SCALE_M*float(g[axis])/den
       if val:ar.append(k);ac.append(3*j+axis);av.append(val)
   Aarea=sparse.coo_matrix((av,(ar,ac)),shape=(len(shape_faces),nvar)).tocsr()
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
   EG,eb,edge_meta=lp_helper.build_edge_gradient_constraints(points_mm,shape_faces,movable,0.5)
   LG=sparse.csr_matrix((0,nvar));lb=np.zeros(0);layer_meta={'mode':'post-emission connected O/I/D reconstruction','linearized_inherited_offset_rows':0}
   Zaa=sparse.csr_matrix((len(shape_faces),nvar));Zam=sparse.csr_matrix((len(shape_faces),nwit))
   Zen=sparse.csr_matrix((EG.shape[0],nvar));Zem=sparse.csr_matrix((EG.shape[0],nwit))
   Zw=sparse.csr_matrix((nwit,nvar))
   I=sparse.eye(nvar,format='csr');Is=sparse.eye(nwit,format='csr')
   Aarea_lp=sparse.hstack((Aarea,Zaa,Zam),format='csr')
   Awitness_lp=sparse.hstack((-Gw,Zw,-Is),format='csr')
   Aedge_lp=sparse.hstack((-EG,Zen,Zem),format='csr')
   Zln=sparse.csr_matrix((LG.shape[0],nabs));Zlw=sparse.csr_matrix((LG.shape[0],nwit))
   Alayer_lp=sparse.hstack((-LG,Zln,Zlw),format='csr')
   Aabs_pos=sparse.hstack((I,-I,sparse.csr_matrix((nvar,nwit))),format='csr')
   Aabs_neg=sparse.hstack((-I,-I,sparse.csr_matrix((nvar,nwit))),format='csr')
   A_ub=sparse.vstack((Aarea_lp,Awitness_lp,Aedge_lp,Alayer_lp,Aabs_pos,Aabs_neg),format='csr')
   b_ub=np.r_[np.asarray(area_rhs),-np.asarray(wreq),-np.asarray(eb),-np.asarray(lb),np.zeros(2*nvar)]
   objective=np.r_[np.zeros(nvar),np.ones(nabs),np.full(nwit,1000.0)]
   bounds=[(-TRUST_STEP_MM,TRUST_STEP_MM)]*nvar+[(0.0,TRUST_STEP_MM)]*nabs+[(0.0,CLEARANCE_MM)]*nwit
   lp=linprog(objective,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs',options={'time_limit':30.0,'presolve':True})
   if lp.success:
    delta=np.asarray(lp.x[:nvar],np.float64);slacks=np.asarray(lp.x[2*nvar:],np.float64)
   else:
    delta=np.zeros(nvar,np.float64);slacks=np.zeros(nwit,np.float64)
   res=SimpleNamespace(success=bool(lp.success),message=str(lp.message),nit=int(lp.nit or 0),x=delta,fun=float(lp.fun) if lp.fun is not None else None)
   direction_path=OUT/'lp-direction.npz'
   direction_sha=None
   if lp.success:
    # Preserve both the full optimizer vector (displacement, L1 auxiliaries, witness slacks)
    # and the displacement direction used by the bounded line search.
    np.savez_compressed(direction_path,optimizer_vector=np.asarray(lp.x,np.float64),
      displacement_mm=delta,witness_slacks_mm=slacks)
    direction_sha=sha(direction_path)
    report['lp_direction_artifact']={'path':str(direction_path),'sha256':direction_sha,
      'optimizer_vector_length':int(len(lp.x)),'displacement_length':int(len(delta)),
      'witness_slack_length':int(len(slacks)),'persisted_before_shape_check':True}
   slack_positive=slacks>1e-7
   rec={'iteration':it,'solver':'HiGHS sparse LP','solver_success':bool(lp.success),'solver_status':int(lp.status),
    'message':str(lp.message),'iterations':int(lp.nit or 0),'LP_variables':int(nfull),'LP_rows':int(A_ub.shape[0]),
    'witness_constraint_count':nwit,'shape_constraint_count':len(shape_faces),'linearized_area_rows_hard':int(len(shape_faces)),
    'hard_edge_gradient_rows':int(EG.shape[0]),'edge_gradient':edge_meta,'hard_layer_guard_rows':int(LG.shape[0]),'layer_guard':layer_meta,
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
   backtracking=[];chosen=None
   for alpha in BACKTRACK_FACTORS:
    scaled_delta=delta*alpha
    double_min=float(shape_fun(scaled_delta).min())
    trial={'alpha':float(alpha),'double_shape_min_constraint':double_min,
      'double_shape_pass':bool(double_min>=-1e-8)}
    if double_min < -1e-8:
     backtracking.append(trial);continue
    proposed_trial=unpack(scaled_delta)
    total_disp_trial=np.linalg.norm(proposed_trial[patchnodes]-original_local[patchnodes],axis=1)
    max_disp_trial=float(total_disp_trial.max()*1000)
    trial['max_cumulative_displacement_mm']=max_disp_trial
    if max_disp_trial>MAX_TOTAL_MM+1e-6:
     trial['movement_cap_pass']=False;backtracking.append(trial);continue
    world_trial=current_world.copy()
    local_step_mm=scaled_delta.reshape(nv,3)
    world_trial[movable]=(P32[movable].astype(np.float64)+(local_step_mm@M[65])/1000.0).astype(np.float32)
    world_trial[fixed]=P32[fixed]
    emitted_local_trial=(world_trial.astype(np.float64)-pos[65])@M[65].T-off[65]
    qshape_trial=quality(emitted_local_trial,shape_faces,original_local)
    layer_trial,Itrial,Dtrial=reconstruct_trial_layers(world_trial)
    shape_pass_trial=bool(qshape_trial['orientation_positive'] and
      qshape_trial['minimum_unsigned_parent_area_ratio']>=AREA_FLOOR and
      qshape_trial['minimum_angle_degrees']>=min_angle_floor and
      qshape_trial['maximum_aspect_measure']<=aspect_cap)
    boundary_same_trial=bool(np.array_equal(world_trial[fixed].view(np.uint32),P32[fixed].view(np.uint32)))
    anchors_same_trial=bool(np.array_equal(world_trial[anchor_rows].view(np.uint32),P32[anchor_rows].view(np.uint32)))
    trial.update({'movement_cap_pass':True,'emitted_f32_shape':qshape_trial,'emitted_layer_nonregression':layer_trial,
      'emitted_shape_pass':shape_pass_trial,'emitted_layer_pass':bool(layer_trial['pass']),'fixed_boundary_bitwise_preserved':boundary_same_trial,'typed_anchors_bitwise_preserved':anchors_same_trial})
    backtracking.append(trial)
    if shape_pass_trial and layer_trial['pass'] and boundary_same_trial and anchors_same_trial:
     chosen=(float(alpha),proposed_trial,world_trial,emitted_local_trial,qshape_trial,layer_trial,Itrial,Dtrial,max_disp_trial,
       float(np.sqrt(np.mean(total_disp_trial**2))*1000))
     break
   rec['backtracking_trials']=backtracking
   rec['backtracking_direction_artifact']=report.get('lp_direction_artifact')
   if chosen is None:
    rec['status']='NO_BACKTRACKED_HARD_SHAPE_FEASIBLE'
    rounds.append(rec);report['status']='PRIVATE_REJECTED_NO_BACKTRACKED_HARD_SHAPE_FEASIBLE';report['rounds']=rounds;break
   alpha,proposed,world,emitted_local,qshape,layer_result,Itrial,Dtrial,max_disp,rms_disp=chosen
   rec.update({'selected_backtrack_factor':alpha,'selected_double_shape_min_constraint':float(shape_fun(delta*alpha).min()),
    'max_cumulative_displacement_mm':max_disp,'rms_cumulative_displacement_mm':rms_disp,
    'emitted_f32_shape':qshape,'emitted_shape_pass':True,'emitted_layer_nonregression':layer_result,'emitted_layer_pass':True,'fixed_boundary_bitwise_preserved':True,'typed_anchors_bitwise_preserved':True,
    'backtrack_direction_artifact':report.get('lp_direction_artifact')})
   round_path=OUT/f'candidate-private-iteration-{it:02d}.npz'
   emitted_z={k:v for k,v in Z.items()};emitted_z['points_world_m']=world
   np.savez_compressed(round_path,**emitted_z)
   rec.update({'candidate_path':str(round_path),'candidate_sha256':sha(round_path)})
   layer_path=OUT/'skin-fat-layers-541-private.npz'
   lzout={k:np.asarray(v).copy() for k,v in np.load(LAYER_FILE,allow_pickle=False).items()}
   lzout.update({'outer_points_f64':world.astype(np.float64),'outer_points_f32':world.astype(np.float32),
      'inner_points_f64':Itrial.astype(np.float64),'inner_points_f32':Itrial.astype(np.float32),
      'deep_points_f64':Dtrial.astype(np.float64),'deep_points_f32':Dtrial.astype(np.float32),
      'outer_faces':layer_oriented,'inner_faces':layer_oriented,'deep_faces':layer_oriented,
      'source_outer_faces_raw':F})
   np.savez_compressed(layer_path,**lzout)
   rec['private_reconstructed_layer_artifact']={'path':str(layer_path),'sha256':sha(layer_path),'physical_cell_mass_or_native_admission':False}
   # Exact local patch-versus-target-bone screen on the emitted Float32 nodes.
   patch_records=mesh_records(emitted_local,shape_faces,affected_rows,np.arange(len(P),dtype=np.int64),exact)
   bgate=exact._audit_pair(patch_records,bone_records,same_surface=False)
   local_pairs={(int(f),int(b)) for f,b in bgate['triangle_pairs']}
   unchanged_pairs={pair for pair in original_pairs if pair[0] not in affected_set}
   cur_pairs=unchanged_pairs|local_pairs
   added=sorted(cur_pairs-original_pairs);removed=sorted(original_pairs-cur_pairs)
   parent_map=np.asarray(Z['refinement_face_parent_face_row'],np.int64)
   baseline_mapped={(int(parent_map[sf]),bid) for sf,bid in original_pairs}
   current_mapped={(int(parent_map[sf]),bid) for sf,bid in cur_pairs}
   mapped_added=sorted(current_mapped-baseline_mapped);mapped_removed=sorted(baseline_mapped-current_mapped)
   mapped_improved=(len(current_mapped)<len(baseline_mapped) and len(mapped_added)==0 and len(mapped_removed)>0)
   rec.update({'exact_local_bone_affected_child_pair_count':int(bgate['count']),'updated_full_child_pair_count':len(cur_pairs),
    'baseline_full_child_pair_count':len(original_pairs),'child_pair_added_count':len(added),'child_pair_removed_count':len(removed),
    'child_pair_added': [list(x) for x in added],'child_pair_removed':[list(x) for x in removed],
    'immediate_parent_mapped_identity_pair_count':len(current_mapped),'baseline_immediate_parent_mapped_identity_pair_count':len(baseline_mapped),
    'mapped_identity_added_vs_current':[list(x) for x in mapped_added],'mapped_identity_removed_vs_current':[list(x) for x in mapped_removed],
    'mapped_identity_net_delta':len(current_mapped)-len(baseline_mapped),'mapped_identity_improved':mapped_improved,'strict_acceptance_requires_no_added_mapped_identities':True,
    'exact_local_bone_affected_pairs':bgate['triangle_pairs']})
   loc=Counter(exact.point_location_prepared(tuple(frac(x) for x in emitted_local[n]),prepared_bone)['location'] for n in patchnodes)
   loc_inside=sorted(int(n) for n in patchnodes if exact.point_location_prepared(tuple(frac(x) for x in emitted_local[n]),prepared_bone)['location'] in ('inside','boundary'))
   new_inside=sorted(set(loc_inside)-set(baseline_inside_nodes))
   rec.update({'patch_node_location_counts':dict(loc),'inside_or_boundary_node_ids':loc_inside,'new_inside_or_boundary_node_ids':new_inside,'resolved_inside_or_boundary_node_ids':sorted(set(baseline_inside_nodes)-set(loc_inside))})
   inside_rows=[];remaining_before=[];remaining_after=[]
   selected_nodes=np.asarray([int(x['node']) for x in inside_targets],np.int64)
   base_dist,_=point_triangle_distances(original_local[selected_nodes],bone)
   after_dist,_=point_triangle_distances(emitted_local[selected_nodes],bone)
   for k,target in enumerate(inside_targets):
    node=int(target['node']);location=exact.point_location_prepared(tuple(frac(x) for x in emitted_local[node]),prepared_bone)['location']
    inside_rows.append({'node':node,'baseline_location':'inside','after_location':location,
      'baseline_audit_nearest_distance_mm':float(target['baseline_audit_nearest_distance_mm']),
      'baseline_recomputed_nearest_distance_mm':float(base_dist[k]*1000.0),
      'after_nearest_distance_mm':float(after_dist[k]*1000.0),
      'baseline_signed_plane_depth_mm':float(target['baseline_signed_plane_depth_mm'])})
    if location in ('inside','boundary'):remaining_before.append(float(base_dist[k]*1000.0));remaining_after.append(float(after_dist[k]*1000.0))
   rec['inside_depth_comparison']={'baseline_inside_count':len(inside_targets),
      'after_inside_or_boundary_count':len(remaining_after),
      'resolved_inside_node_rows':[q['node'] for q in inside_rows if q['after_location']=='outside'],
      'remaining_inside_node_rows':[q['node'] for q in inside_rows if q['after_location'] in ('inside','boundary')],
      'baseline_median_depth_for_remaining_nodes_mm':float(np.median(remaining_before)) if remaining_before else None,
      'after_median_depth_for_remaining_nodes_mm':float(np.median(remaining_after)) if remaining_after else None,
      'baseline_max_depth_for_remaining_nodes_mm':float(np.max(remaining_before)) if remaining_before else None,
      'after_max_depth_for_remaining_nodes_mm':float(np.max(remaining_after)) if remaining_after else None,
      'nodes':inside_rows,'distance_metric':'Euclidean point-to-triangle distance; exact inside/boundary/outside classification is reported separately'}
   rounds.append(rec);current_world=world;current_local=emitted_local
   if mapped_improved:
    # Local self check only follows strict identity improvement.
    comp=np.asarray([r for r in range(len(F)) if r not in affected_set],np.int64)
    patch_tri=emitted_local[shape_faces];lo=patch_tri.min(axis=(0,1));hi=patch_tri.max(axis=(0,1))
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
      'pairs':bgate['triangle_pairs'],'immediate_parent_mapped_identity_pair_count':len(current_mapped),'baseline_immediate_parent_mapped_identity_pair_count':len(baseline_mapped),
      'mapped_identity_added':[list(x) for x in mapped_added],'mapped_identity_removed':[list(x) for x in mapped_removed],
      'mapped_identity_net_delta':len(current_mapped)-len(baseline_mapped),'strict_mapped_improvement':True},'sampled_inside_comparison':{'scope':'all unique nodes in the 248-face patch that are potentially affected by movable nodes',
      'baseline_inside_or_boundary_node_ids':baseline_inside_nodes,'final_inside_or_boundary_node_ids':loc_inside,
      'new_inside_or_boundary_node_ids':new_inside,'resolved_inside_or_boundary_node_ids':sorted(set(baseline_inside_nodes)-set(loc_inside)),'no_new_inside_or_boundary_witnesses':not new_inside},
      'exact_local_self_gate':self_gate,'selected_private_candidate_path':str(round_path),'selected_private_candidate_sha256':sha(round_path)})
    if not self_gate['pass'] or new_inside:
     report['status']='PRIVATE_REJECTED_NEW_LOCAL_SELF_OR_INSIDE_WITNESS'
    break
   if it==MAX_OUTER_ITERS:
    report['status']='PRIVATE_REJECTED_NO_MAPPED_PARENT_IDENTITY_IMPROVEMENT';break
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
