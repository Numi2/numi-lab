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


OUT=e/'outward-skin-bone-exits-017';OUT.mkdir(exist_ok=False)
AP=e/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
X=np.load(AP)['points'].astype(np.float32).astype(float)
oz=np.load(e/'bone-clearance-fit-041-exact-material-routes/baseline-outer-inside-local-planes.npz')
rows=[];hp=[];nn=[];cc=[];bb=[];bk=[]
for i,c in enumerate(oz['constraint_local_node'].astype(int)):
 body=int(oz['constraint_body_index'][i]);bone=int(oz['constraint_bone_stable_id'][i]);b=bodies[body]
 pos=np.array([b['position_'+k+'_m'] for k in 'xyz'],np.float32).astype(float)
 off=np.array([b['com_offset_'+k+'_m'] for k in 'xyz'],float)
 direction=X[c]-X[2411+c];direction/=np.linalg.norm(direction)
 AA=(X[c]-pos)@matrix[body].T-off
 end=X[c]+.1*direction;BB=(end-pos)@matrix[body].T-off
 ix=np.flatnonzero((ids[:,0]==body)&(ids[:,1]==bone));tris=T[ix]
 lo=np.minimum(AA,BB);hi=np.maximum(AA,BB)
 cand=np.flatnonzero(np.all(tris.max(axis=1)>=lo,axis=1)&np.all(tris.min(axis=1)<=hi,axis=1))
 d=BB-AA;e1=tris[cand,1]-tris[cand,0];e2=tris[cand,2]-tris[cand,0]
 pp=np.cross(d,e2);dd=np.einsum('ij,ij->i',e1,pp);inv=np.divide(1.,dd,out=np.zeros_like(dd),where=dd!=0)
 vv=AA-tris[cand,0];u=inv*np.einsum('ij,ij->i',vv,pp);qq=np.cross(vv,e1);v=inv*(qq@d);tt=inv*np.einsum('ij,ij->i',e2,qq)
 possible=(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(tt>=-1e-10)&(tt<=1+1e-10)
 hits=[]
 for ci in np.flatnonzero(possible):
  er=exact['classify_segment_triangle'](AA,BB,tris[cand[ci]])
  if er['classification'] in ('proper_interior_hit','boundary_hit'):hits.append((er['t'],int(cand[ci]),er))
 hits.sort(key=lambda v:v[0])
 if not hits:
  rows.append(dict(column=int(c),body=body,bone=bone,status='NO_EXACT_EXIT_FOUND'));continue
 t,j,er=hits[0];point=X[c]+.1*t*direction
 normal=matrix[body].T@np.cross(tris[j,1]-tris[j,0],tris[j,2]-tris[j,0]);normal/=np.linalg.norm(normal)
 if normal@direction<0:normal=-normal
 ambiguous=er['classification']!='proper_interior_hit' or (len(hits)>1 and abs(hits[1][0]-t)<1e-10)
 rows.append(dict(column=int(c),body=body,bone=bone,triangle=int(ids[ix[j],3]),status='AMBIGUOUS_EXIT' if ambiguous else 'EXACT_OUTWARD_DIRECTION_EXIT',exit_distance_m=float(.1*t),hit_world_m=point.tolist(),outward_normal_world=normal.tolist(),ray_world=direction.tolist(),normal_dot_ray=float(normal@direction),nearest_exit_normal_dot_ray=float(oz['constraint_outward_normal_world'][i]@direction),exact_witness=er))
 if not ambiguous:cc.append(c);bb.append(body);bk.append(bone);hp.append(point);nn.append(normal)
np.savez_compressed(OUT/'outward-exit-planes.npz',constraint_local_node=np.array(cc),constraint_body_index=np.array(bb),constraint_bone_stable_id=np.array(bk),constraint_closest_world_m=np.array(hp),constraint_outward_normal_world=np.array(nn),constraint_margin_m=np.full(len(cc),.0001))
report=dict(status='DIRECTIONAL_EXIT_CONSTRUCTION_PLANES_NOT_GEOMETRY_ADMISSION',rows=rows,certified_planes=len(cc),inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [AP,tr,br,e/'bone-clearance-fit-041-exact-material-routes/baseline-outer-inside-local-planes.npz']},limitations=['Outward defined by existing O-I layer direction, not newly measured anatomy.','Source O12 inside classification from retained baseline native query; nearest proper positive exit selected with exact rational narrowphase.','Triangle plane is a local construction constraint, not complete bone or tissue clearance.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'planes':len(cc),'rows':rows},indent=2))
