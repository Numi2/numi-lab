import pathlib,csv,json,hashlib,runpy,numpy as np,time
e=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");p=e/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight";q=e/"bone-clearance-fit-038-candidate-query/run-002/source-preflight"
t0=time.monotonic();tr=p/"exact-emitted.triangles.csv";nr=p/"exact-emitted.nodes.csv";br=p/"exact-emitted.bodies.csv"
a=np.loadtxt(tr,delimiter=",",skiprows=1);assert a.shape==(469136,23)
V=a[:,5:14].astype(np.float32);assert np.array_equal(V.view(np.uint32),a[:,14:23].astype(np.uint32))
T=V.astype(float).reshape(-1,3,3);ids=a[:,1:5].astype(np.int64);assert np.array_equal(ids.astype(float),a[:,1:5])
n=np.loadtxt(nr,delimiter=",",skiprows=1);assert n.shape==(7233,13);assert np.array_equal(n[:,5:9].astype(np.float32).view(np.uint32),n[:,9:13].astype(np.uint32));assert np.array_equal(n[:,0],np.arange(7233));gids=n[:,1].astype(np.int64);X=n[:,5:8].astype(np.float32).astype(float)
candidate=np.load(e/"bone-clearance-fit-037-componentwise-035affine-002/candidate.npz");assert np.array_equal(X.astype(np.float32),candidate["points_world_f32"].astype(np.float32));assert np.array_equal(gids,candidate["global_node_ids"])
gmap={int(g):i for i,g in enumerate(gids)}
bodies={int(r["body_index"]):r for r in csv.DictReader(br.open())};loc={};matrix={};maxM=0
for body in np.unique(ids[:,0]):
 b=bodies[int(body)];pos=np.array([b["position_"+c+"_m"] for c in "xyz"],np.float32).astype(float);quat=np.array([b["orientation_q"+c] for c in "xyzw"],np.float32).astype(float);quat[:3]*=-1
 M=np.array([float(b[f"world_to_body_m{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3,3)
 def rotate(v):return v+2*np.cross(quat[:3],np.cross(quat[:3],v)+quat[3]*v)
 Mcheck=np.column_stack([rotate(v) for v in np.eye(3)]);maxM=max(maxM,float(np.max(np.abs(M-Mcheck))));assert np.max(np.abs(M-Mcheck))<1e-12
 off=np.array([b["com_offset_"+c+"_m"] for c in "xyz"],float);loc[int(body)]=(X-pos)@M.T-off;matrix[int(body)]=M
query=list(csv.DictReader((q/"all-bank-containment.csv").open()));qkeys=set();maxlocal=0
for r in query:
 body=int(r["bank_body"]);g=int(r["global_node"]);bone=int(r["bank_stable_id"]);key=(g,body,bone);assert key not in qkeys;qkeys.add(key)
 y=np.array([r["query_local_"+c] for c in "xyz"],float);err=float(np.linalg.norm(y-loc[body][gmap[g]]));maxlocal=max(maxlocal,err);assert err<1e-12,(key,err)
bankkeys=np.unique(ids[:,:2],axis=0);assert len(bankkeys)==185;actual=set();bankcounts={}
inventory={(int(r["body"]),int(r["stable_bone_id"])):int(r["triangle_count"]) for r in csv.DictReader((q/"all-bank-containment.csv.banks.csv").open())}
for body,bone in bankkeys:
 ix=(ids[:,0]==body)&(ids[:,1]==bone);tri=T[ix];lo=tri.min(axis=(0,1));hi=tri.max(axis=(0,1));bankcounts[(int(body),int(bone))]=int(ix.sum())
 inside_aabb=np.all(loc[int(body)]>=lo,axis=1)&np.all(loc[int(body)]<=hi,axis=1)
 actual.update((int(g),int(body),int(bone)) for g in gids[inside_aabb])
assert bankcounts==inventory
assert actual==qkeys,dict(extra=list(actual-qkeys)[:4],missing=list(qkeys-actual)[:4])
ix=np.flatnonzero((ids[:,0]==23)&(ids[:,1]==32)&(ids[:,3]==1381));assert len(ix)==1
exact=runpy.run_path(str(e/"root_exact_segment_witness.py"));hit=exact["classify_segment_triangle"](loc[23][gmap[5023]],loc[23][gmap[7434]],T[ix[0]])
assert hit["classification"]=="proper_interior_hit"


# New baseline audit: original 041 was explicitly the 037002 candidate, not baseline.
# Reuse exact compiled banks only after frozen 037 frame/query calibration above.
AP=e/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
ap=np.load(AP,allow_pickle=False)
X=ap['points'].astype(np.float32).astype(float)
assert np.array_equal(gids,ap['global_nodes'])
baseline_csv=e/'native-candidate020-all-bank-containment-035/source-preflight-003/initial-skin-fem-nodes.csv'
baseline_rows=list(csv.DictReader(baseline_csv.open()))
BX=np.array([[float(r[c+'_m']) for c in 'xyz'] for r in baseline_rows],np.float32).astype(float)
assert np.array_equal(BX,X)
for body in loc:
 b=bodies[body];pos=np.array([b['position_'+c+'_m'] for c in 'xyz'],np.float32).astype(float)
 off=np.array([b['com_offset_'+c+'_m'] for c in 'xyz'],float)
 loc[body]=(X-pos)@matrix[body].T-off
q=e/'native-candidate020-all-bank-containment-035/source-preflight-003'
query=list(csv.DictReader((q/'all-bank-containment.csv').open()))
qkeys=set();maxlocal=0.
for r in query:
 body=int(r['bank_body']);g=int(r['global_node']);bone=int(r['bank_stable_id']);key=(g,body,bone)
 assert key not in qkeys;qkeys.add(key)
 y=np.array([r['query_local_'+c] for c in 'xyz'],float)
 err=float(np.linalg.norm(y-loc[body][gmap[g]]));maxlocal=max(maxlocal,err)
 assert err<1e-12,(key,err)
actual=set()
for body,bone in bankkeys:
 ix=(ids[:,0]==body)&(ids[:,1]==bone);tri=T[ix];lo=tri.min(axis=(0,1));hi=tri.max(axis=(0,1))
 inside_aabb=np.all(loc[int(body)]>=lo,axis=1)&np.all(loc[int(body)]<=hi,axis=1)
 actual.update((int(g),int(body),int(bone)) for g in gids[inside_aabb])
assert actual==qkeys,dict(extra=list(actual-qkeys)[:4],missing=list(qkeys-actual)[:4])
OUT=e/"baseline-exact-material-routes-010";OUT.mkdir(exist_ok=False)
qmap={(int(z["global_node"]),int(z["bank_body"]),int(z["bank_stable_id"])):z["classification"] for z in query}
for li in range(3):
 assert np.array_equal(n[li*2411:(li+1)*2411,2],np.full(2411,li))
 assert np.array_equal(n[li*2411:(li+1)*2411,3],np.arange(2411))
counts=dict(total_column_bank_pairs=2411*185,total_segment_bank_pairs=4822*185,AABB_candidate_segment_bank_pairs=0,triangle_AABB_candidates=0,exact_candidates=0,proper_hits=0,boundary_hits=0,coplanar_candidates=0,hit_segment_bank_pairs=0)
records=[];uncertain=[]
for body,bone in bankkeys:
 body,bone=int(body),int(bone)
 ix=np.flatnonzero((ids[:,0]==body)&(ids[:,1]==bone));tris=T[ix];stable=ids[ix,3];source=int(ids[ix[0],2])
 tl=tris.min(axis=1);th=tris.max(axis=1);bl=tl.min(axis=0);bh=th.max(axis=0)
 L=loc[body].reshape(3,2411,3)
 for seg in range(2):
  AA=L[seg];BB=L[seg+1];rl=np.minimum(AA,BB);rh=np.maximum(AA,BB)
  active=np.flatnonzero(np.all(rh>=bl,axis=1)&np.all(rl<=bh,axis=1))
  counts["AABB_candidate_segment_bank_pairs"]+=len(active)
  for row in active:
   cand=np.flatnonzero(np.all(th>=rl[row],axis=1)&np.all(tl<=rh[row],axis=1))
   counts["triangle_AABB_candidates"]+=len(cand)
   if not len(cand):continue
   tri=tris[cand];d=BB[row]-AA[row];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
   pp=np.cross(d,e2);dd=np.einsum("ij,ij->i",e1,pp)
   scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(d)
   near=np.abs(dd)<=64*np.finfo(float).eps*scale
   inv=np.divide(1.,dd,out=np.zeros_like(dd),where=dd!=0)
   vv=AA[row]-tri[:,0];u=inv*np.einsum("ij,ij->i",vv,pp);qq=np.cross(vv,e1);v=inv*(qq@d);tt=inv*np.einsum("ij,ij->i",e2,qq)
   possible=near|((u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(tt>=-1e-10)&(tt<=1+1e-10))
   hits=[]
   for ci in np.flatnonzero(possible):
    counts["exact_candidates"]+=1
    er=exact["classify_segment_triangle"](AA[row],BB[row],tri[ci]);cl=er["classification"];tid=int(stable[cand[ci]])
    if cl in ("coplanar_requires_2d_test","invalid_degenerate_triangle"):
     counts["coplanar_candidates"]+=cl=="coplanar_requires_2d_test"
     uncertain.append(dict(row=int(row),segment=seg,body=body,bone_stable_id=bone,triangle_stable_id=tid,classification=cl));continue
    if cl not in ("proper_interior_hit","boundary_hit"):continue
    counts["proper_hits" if cl=="proper_interior_hit" else "boundary_hits"]+=1
    hits.append((er["t"],tid,int(cand[ci]),er))
   if not hits:continue
   counts["hit_segment_bank_pairs"]+=1;hits.sort(key=lambda z:(z[0],z[1]))
   t_hit,tid,trirow,er=hits[0];worldstart=X[seg*2411+row];worldend=X[(seg+1)*2411+row]
   point=worldstart+t_hit*(worldend-worldstart);normal=np.cross(tris[trirow,1]-tris[trirow,0],tris[trirow,2]-tris[trirow,0]);normal=matrix[body].T@normal;normal/=np.linalg.norm(normal)
   if np.dot(normal,worldstart-point)<0:normal=-normal
   globalstart=int(gids[seg*2411+row]);globalend=int(gids[(seg+1)*2411+row])
   startclass=qmap.get((globalstart,body,bone),"outside_AABB_proven");endclass=qmap.get((globalend,body,bone),"outside_AABB_proven")
   firstboundary=er["classification"]!="proper_interior_hit" or (len(hits)>1 and abs(hits[1][0]-t_hit)<1e-10)
   rec=dict(row=int(row),segment=seg,segment_name="O-I" if seg==0 else "I-D",body=body,bone_stable_id=bone,source_record_index=source,triangle_stable_id=tid,t=t_hit,start_classification=startclass,end_classification=endclass,hit_triangle_count=len(hits),first_hit_ambiguous=firstboundary,exact_first_hit=er,plane_point_world_m=point.tolist(),unit_exterior_normal_world=normal.tolist(),segment_start_world_f32_m=worldstart.tolist(),segment_end_world_f32_m=worldend.tolist(),segment_start_query_local_m=AA[row].tolist(),segment_end_query_local_m=BB[row].tolist(),triangle_body_local_f32_m=tris[trirow].tolist(),all_hits=[dict(t=h[0],triangle_stable_id=h[1],classification=h[3]["classification"]) for h in hits])
   records.append(rec)
groups={}
for rec in records:groups.setdefault((rec["row"],rec["body"],rec["bone_stable_id"]),[]).append(rec)
planes=[]
for key,rr in sorted(groups.items()):
 rr.sort(key=lambda z:(z["segment"],z["t"],z["triangle_stable_id"]));r=rr[0]
 r["is_first_route_hit"]=True
 if r["start_classification"] in ("outside","outside_AABB_proven") and not r["first_hit_ambiguous"]:
  mask=6 if r["segment"]==0 else 4
  plane={k:r[k] for k in ["row","segment","body","bone_stable_id","source_record_index","triangle_stable_id","plane_point_world_m","unit_exterior_normal_world"]}
  plane["target_layer_mask"]=mask;plane["clearance_m"]=.0001;planes.append(plane)
summary=dict(status="EXACT_NATIVE_GEOMETRY_ROUTE_WITNESSES_NOT_FULL_TISSUE_CLEARANCE",counts=counts,columns_with_hits=len({r["row"] for r in records}),column_bank_groups_with_hits=len(groups),construction_planes=len(planes),uncertainties=len(uncertain),native_query_frame_max_error_m=maxlocal,inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [AP,baseline_csv,tr,nr,br,q/"all-bank-containment.csv",e/"root_exact_segment_witness.py"]},elapsed_seconds=time.monotonic()-t0,limitations=["Exact rational signs certify narrowphase candidates; this is not a layer-triangle/tet clearance test.","Coplanar and first-boundary feature hits remain explicit unresolved witnesses.","Construction planes are local exterior-side choices, not a physical force/state owner.","No native physical step or anatomy acceptance."],route_witnesses=records,construction_planes_world=planes,uncertain_candidates=uncertain)
(OUT/"route-report.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k not in ["route_witnesses","construction_planes_world","uncertain_candidates"]},indent=2))

