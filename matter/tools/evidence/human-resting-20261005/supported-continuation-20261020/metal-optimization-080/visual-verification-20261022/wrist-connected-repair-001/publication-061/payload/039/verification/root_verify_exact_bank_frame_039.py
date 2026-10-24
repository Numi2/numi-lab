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
out=dict(status="PASS_EXACT_NATIVE_BANK_FRAME_AND_ROW946_WITNESS",triangles=len(T),banks=len(bankkeys),candidate_nodes=len(X),total_node_bank_pairs=len(X)*len(bankkeys),exact_F32_triangle_components_verified=int(V.size),native_query_keys=len(qkeys),all_missing_keys_AABB_proven_outside=len(X)*len(bankkeys)-len(qkeys),max_raw_matrix_difference=maxM,max_native_query_local_residual_m=maxlocal,row946_triangle1381_exact_sign_result=hit,input_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [tr,nr,br]},elapsed_seconds=time.monotonic()-t0,limitation="Certifies emitted frames, sparse-query coverage and one exact crossing witness; no all-column, triangle/tet-clearance, or native physical admission.")
(e/"root-independent-exact-bank-frame-039.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
