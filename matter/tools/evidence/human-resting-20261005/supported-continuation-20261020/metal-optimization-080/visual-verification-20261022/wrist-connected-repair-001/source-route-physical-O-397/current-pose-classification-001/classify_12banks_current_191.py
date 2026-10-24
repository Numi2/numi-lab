#!/usr/bin/env python3
from __future__ import annotations
import ast, hashlib, importlib.util, json, math, os, sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np

E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=E/'source-route-physical-O-397/current-pose-classification-001/results'
BASE=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
BASE_MANIFEST=BASE.with_name('bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json')
REG=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/brachio109-local810-existing-full-parent-source-853b/assets/candidate-current-source.json')
FRESH=E/'source-route-physical-O-397/compiler-12bank-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
FRESH_MANIFEST=FRESH.with_name('bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json')
MODEL=E/'source-route-weighting-owner-385/src/numilab_human/model.py'
BUILDER=E/'source-route-physical-O-397/corrected-12-bank-route-candidate-001/build_weight_candidate_12.py'
CAND=E/'source-route-physical-O-397/corrected-12-bank-route-candidate-001/output/corrected-route-weights-candidate-12.nhtissue'
CAND_RECEIPT=CAND.with_name('candidate-receipt-12.json')
C386=E/'source-route-frame-owner-386/corrected-route-weights-candidate.nhtissue'
C386_RECEIPT=E/'source-route-frame-owner-386/candidate-receipt.json'
INCIDENT=E/'source-route-physical-O-397/wrong-output-incident-001/incident-receipt.json'
POSE=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
CACHE_DIR=E/'registered-surrounding-tissue-native013-191'
INVENTORY=CACHE_DIR/'inventory.json'
CACHE=CACHE_DIR/'registered-tissue-banks.npz'
EMITTER=CACHE_DIR/'prepare_registered_surrounding.py'
REFERENCE=E/'source-route-frame-owner-396/tools/rebind_human_tissue_reference.py'
WITNESS=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/p1-distance-initializer-393/physical-outer-inside-witnesses.json'
OUTER=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/candidate.npz'
WINDING=E/'ankle-parent-tet-refinement-026/generated-224-commonfield-tetgen-volume-actual007/build_commonfield_224.py'
SELECTED={75,79,81,83,85,89,93,97,103,120,122,132}
EXPECTED={
 str(BASE):'8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042',
 str(BASE_MANIFEST):'ffe60f996e92325ce21123da86e259a02a9aaed1a874373a0e71721a2ceba07b',
 str(REG):'31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c',
 str(MODEL):'5bd314fa7ceaebc6c0e72a0115befdda7a19e11b081bc2009cd2a1e94ce01483',
 str(BUILDER):'cbfebb1bdf3e379c8b627a2b824fc4c34f3e85f7fba164877f5c33ca5b11826c',
 str(CAND):'b2d36c825528dedae0394dfa846bae64e9d5793bdac88f2a28133fc9c44c2c67',
 str(CAND_RECEIPT):'6c2ab863ac2314342149b1f489557ead14a4735cadc5599d621c43dbdd08c7db',
 str(C386):'7a40994f497555111652ac9c621b7f8d90b498d6e7be8bfda9193910060a65b2',
 str(C386_RECEIPT):'fa3e8becae2427daf6358945b5b8d6c96e56ab8e5a9e871057477070742a07a4',
 str(INCIDENT):'338406a21c3df87cbe8073a883bf0898072ef0f8f82ac65da06cb78cac0bfdea',
 str(POSE):'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd',
 str(INVENTORY):'b0823cf8f33df2586ba1cd45e5b260cd75d1e35b8d2682b67c15f64ea22bbcc7',
 str(CACHE):'39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792',
 str(EMITTER):'3606321c1265fd1350afb2c1ce765ce7f7afe99d65f0cebfd77a4504c58d235b',
 str(REFERENCE):'7adf9cb321168e859f21ae62cf79ee59d94ce50fffbc0aeac2971feb18defa83',
 str(WITNESS):'e554b1466b58020c7a27d0ce591e3e2f35b4d597525e0e8b71bec7a155c48c82',
 str(OUTER):'e70b3e08aa385c8a321ebb0849257ec4afa04d13050e26fc28647d36146d26ba',
 str(WINDING):'f872076a777d4ff4147f22cb8f8c275fe96ee1912bdf8e129c7dcf65d4730f8c',
 str(FRESH):'3124bb7b711b7e64bc769834946485bb6c0e27a94ab57f4227fc06e6c7a7d414',
 str(FRESH_MANIFEST):'fb7e71285b61d488441a8516ead3fb480f41de4b8e5a8c2f148b5d2ba608f33a',
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
for path,expected in EXPECTED.items():
 actual=sha(path)
 if actual!=expected: raise SystemExit(f'pin mismatch {path}: expected {expected}, got {actual}')
if OUT.exists(): raise SystemExit(f'refusing existing result folder {OUT}')

# Load the existing owner and the exact 191 emitter. This is source replay only.
spec=importlib.util.spec_from_file_location('route_reference_396_classifier',REFERENCE)
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
base=ref.decode_nhtiss(BASE.read_bytes())
candidate=ref.decode_nhtiss(CAND.read_bytes())
c386=ref.decode_nhtiss(C386.read_bytes())
pose_data=json.loads(POSE.read_text())
pose_map=ref.load_pose_map(pose_data,base['registration'])
inventory=json.loads(INVENTORY.read_text())
cache=np.load(CACHE,allow_pickle=False)
bank_map=ref.validate_inventory(inventory,sha(BASE),sha(POSE),sha(CACHE),base['nr'])
rotate,bodyworld=ref.load_exact_191_helpers(EMITTER,EXPECTED[str(EMITTER)],pose_map)
for payload,label in ((candidate,'12-bank'),(c386,'386')):
 if (payload['magic'],payload['abi'],payload['nr'],payload['nb'],payload['nv'],payload['ni'],payload['registration'],payload['source_id']) != (base['magic'],base['abi'],base['nr'],base['nb'],base['nv'],base['ni'],base['registration'],base['source_id']): raise ValueError(label+' header mismatch')
 if not np.array_equal(payload['records'],base['records']):raise ValueError(label+' record table changed')
 if not np.array_equal(payload['indices'],base['indices']):raise ValueError(label+' topology changed')
 if not np.array_equal(payload['bindings']['body'],base['bindings']['body']) or not np.array_equal(payload['bindings']['transform'],base['bindings']['transform']):raise ValueError(label+' binding transforms changed')
 if not np.array_equal(payload['vertices']['p'],base['vertices']['p']) or not np.array_equal(payload['vertices']['n'],base['vertices']['n']):raise ValueError(label+' source positions/normals changed')
record_by_id={int(row[6]):row for row in base['records']}
def faces_for(payload,record):
 fb,bc,fv,vc,fi,ic,sid,layer=map(int,record)
 ix=payload['indices'][fi:fi+ic].astype(np.int64)
 if len(ix) and ix.max()<vc:return ix.reshape(-1,3)
 if len(ix) and ix.min()>=fv and ix.max()<fv+vc:return (ix-fv).reshape(-1,3)
 raise ValueError(f'{sid}: invalid face index span')
# Verify route re-emission changed only selected per-vertex binding-index/value lanes.
allowed=bytearray(len(base['raw']))
for sid in SELECTED:
 rec=record_by_id[sid];fv,vc=int(rec[2]),int(rec[3])
 for j in range(fv,fv+vc):
  lo=base['vertex_offset']+56*j+24;allowed[lo:lo+32]=b''*32
for i,(x,y) in enumerate(zip(base['raw'],candidate['raw'])):
 if x!=y and not allowed[i]:raise ValueError(f'unexpected candidate byte changed outside target lanes at {i}')
# Verify all12 rows changed, 132 exactly matches the already frozen 386 route lane block.
rec386={int(x[6]):x for x in c386['records']}
changed_counts={};stable132_exact=False
for sid in sorted(SELECTED):
 r=record_by_id[sid];fb,bc,fv,vc=map(int,r[:4])
 diffs=0
 for j in range(fv,fv+vc):
  lo=base['vertex_offset']+56*j
  diffs += int(base['raw'][lo+24:lo+56]!=candidate['raw'][lo+24:lo+56])
 changed_counts[str(sid)]=diffs
 if diffs==0:raise ValueError(f'385 selected bank {sid} did not change any route records')
 if sid==132:
  stable132_exact=(candidate['raw'][candidate['vertex_offset']+56*fv:candidate['vertex_offset']+56*(fv+vc)]==c386['raw'][c386['vertex_offset']+56*fv:c386['vertex_offset']+56*(fv+vc)])
  if not stable132_exact:raise ValueError('stable132 385 lanes do not match frozen386')

# Exact 191 current Float32 output and matching baseline bank cache.
base_surfaces={};candidate_surfaces={}
baseline_matches=[]
for sid in sorted(SELECTED):
 rec=record_by_id[sid];cachemeta=bank_map[sid];key=cachemeta['key']
 v0,n0=ref.emit_current(base,rec,pose_map,rotate,bodyworld)
 vc,nc=ref.emit_current(candidate,rec,pose_map,rotate,bodyworld)
 f0=faces_for(base,rec);fc=faces_for(candidate,rec)
 fcache=np.asarray(cache[key+'_faces'],np.int64)
 vcache=np.asarray(cache[key+'_vertices'],np.float32)
 if not np.array_equal(f0,fc):raise ValueError(f'{sid}: face indices differ')
 if not np.array_equal(f0,fcache):raise ValueError(f'{sid}: baseline source/cache face order differs')
 if v0.shape!=vcache.shape:raise ValueError(f'{sid}: baseline cache vertex count differs')
 delta=np.abs(v0.astype(np.float64)-vcache.astype(np.float64))
 maxd=float(np.linalg.norm((v0-vcache).astype(np.float64),axis=1).max())
 exact=bool(np.array_equal(v0,vcache))
 if not exact:raise ValueError(f'{sid}: original payload is not bitwise 191 cache replay (max={maxd})')
 base_surfaces[sid]=(v0,f0);candidate_surfaces[sid]=(vc,fc)
 displacement=np.linalg.norm((vc-v0).astype(np.float64),axis=1)*1000.0
 tri0=v0[f0].astype(np.float64);tri1=vc[fc].astype(np.float64)
 n0f=np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]);n1f=np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0])
 a0=np.linalg.norm(n0f,axis=1);a1=np.linalg.norm(n1f,axis=1)
 cosine=np.einsum('ij,ij->i',n0f,n1f)/np.maximum(a0*a1,1e-300)
 baseline_matches.append({'stable_id':sid,'inventory_key':key,'baseline_191_cache_max_vertex_delta_m':maxd,'baseline_191_cache_exact':exact,'face_order_exact':True,
  'candidate_current_displacement_mm':{'p50':float(np.quantile(displacement,.5)),'p95':float(np.quantile(displacement,.95)),'max':float(displacement.max())},
  'candidate_current_zero_area_faces':int(np.count_nonzero(a1==0)),'candidate_current_reversed_vs_baseline_faces':int(np.count_nonzero(cosine<=0)),
  'candidate_vs_base_current_min_area_ratio':float(np.min(a1/np.maximum(a0,1e-300)))})

# Load precise source lineage. The raw recompile is ancestry evidence only; it is not
# substituted for the historical reconstructed surfaces in the current NHTISS.
old_manifest=json.loads(BASE_MANIFEST.read_text());fresh_manifest=json.loads(FRESH_MANIFEST.read_text())
old_rows={int(x['stable_id']):x for x in old_manifest['source']['surfaces']}
fresh_rows={int(x['stable_id']):x for x in fresh_manifest['source']['surfaces']}
payload_lineage=[]
for sid in sorted(SELECTED):
 old=old_rows[sid];fresh=fresh_rows[sid];rec=record_by_id[sid]
 if old.get('member_sha256')!=fresh.get('member_sha256'):raise ValueError(f'{sid}: raw member hash differs across manifests')
 fb,bc,fv,vc,fi,ic,*_=map(int,rec)
 pbytes=base['vertices'][fv:fv+vc]['p'].tobytes()
 nbytes=base['vertices'][fv:fv+vc]['n'].tobytes()
 fbytes=base['indices'][fi:fi+ic].tobytes()
 fr=fresh_manifest['payload']
 fbank_rec={int(r[6]):r for r in fresh['records']} if isinstance(fresh.get('records'),list) else {}
 payload_lineage.append({
  'stable_id':sid,'member_id':old.get('member_id'),'label':old.get('label'),
  'raw_source_member_sha256':old.get('member_sha256'),
  'historical_reference_reconstruction':old.get('reference_surface_reconstruction'),
  'source_topology_cancellation':old.get('source_topology_cancellation'),
  'current_reconstructed_payload_counts':{'vertices':vc,'faces':ic//3},
  'current_payload_position_sha256':hashlib.sha256(pbytes).hexdigest(),
  'current_payload_normal_sha256':hashlib.sha256(nbytes).hexdigest(),
  'current_payload_face_index_sha256':hashlib.sha256(fbytes).hexdigest(),
  'raw385_recompile_counts':{'vertices':int(fresh.get('vertex_count',0)),'faces':int(fresh.get('triangle_count',0))},
  'raw385_recompile_geometry_used':False,
  'raw385_recompile_manifest_surface':{k:fresh.get(k) for k in ('stable_id','member_id','member_sha256','label','matched_muscles','body_bindings','reference_surface_reconstruction','source_topology_cancellation')},
 })

# Exact body-group windings and closest point-to-triangle distances for all26
# actual physical-O witness nodes, before vs after route-weight correction.
witness_data=json.loads(WITNESS.read_text())
witnesses=witness_data['witnesses']
outer=np.load(OUTER,allow_pickle=False)
points=np.asarray(outer['points_world_m'],dtype=np.float32)
for row in witnesses:
 node=int(row['node_id']);xyz=np.asarray(row['xyz_m'],dtype=np.float32)
 if node<0 or node>=len(points) or not np.array_equal(points[node],xyz):raise ValueError(f'physical-O witness node {node} does not match candidate point')

# Extract only two already-tested geometric classifier functions from frozen224.
tree=ast.parse(WINDING.read_text(),filename=str(WINDING))
want={'consistent_closed','solid_angle_winding'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in want]
if {n.name for n in nodes}!=want:raise ValueError('frozen224 classifier API mismatch')
ns={'Counter':Counter,'math':math,'np':np}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(WINDING),'exec'),ns)
consistent_closed=ns['consistent_closed'];solid_angle_winding=ns['solid_angle_winding']
groups=witness_data['group_contributors']
def welded_group(surface_map,stable_ids):
 Vparts=[];Fparts=[];offset=0
 for sid in stable_ids:
  V,F=surface_map[int(sid)]
  Vparts.append(np.asarray(V,np.float32));Fparts.append(np.asarray(F,np.int64)+offset);offset+=len(V)
 Vcat=np.vstack(Vparts);Fcat=np.vstack(Fparts)
 Vw,inverse=np.unique(Vcat,axis=0,return_inverse=True)
 Fw=inverse[Fcat]
 closed,top=consistent_closed(Fw)
 return Vw,Fw,bool(closed),top
def exact_point_triangle_distance(point,triangles,chunk=8192):
 p=np.asarray(point,np.float64);T=np.asarray(triangles,np.float64)
 best=float('inf');bestrow=-1
 for lo in range(0,len(T),chunk):
  tri=T[lo:lo+chunk];a=tri[:,0];b=tri[:,1];c=tri[:,2]
  ab=b-a;ac=c-a;ap=p[None,:]-a
  d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
  d20=np.einsum('ij,ij->i',ap,ab);d21=np.einsum('ij,ij->i',ap,ac)
  den=d00*d11-d01*d01
  ok=den>np.finfo(np.float64).tiny
  v=np.zeros(len(tri));w=np.zeros(len(tri))
  v[ok]=(d11[ok]*d20[ok]-d01[ok]*d21[ok])/den[ok]
  w[ok]=(d00[ok]*d21[ok]-d01[ok]*d20[ok])/den[ok]
  u=1-v-w
  inside=ok&(u>=0)&(v>=0)&(w>=0)
  proj=a+v[:,None]*ab+w[:,None]*ac
  plane=np.einsum('ij,ij->i',p[None,:]-proj,p[None,:]-proj)
  plane[~inside]=np.inf
  segmins=np.full(len(tri),np.inf)
  for x,y in ((a,b),(b,c),(c,a)):
   edge=y-x;den2=np.einsum('ij,ij->i',edge,edge)
   t=np.zeros(len(tri));good=den2>np.finfo(np.float64).tiny
   t[good]=np.einsum('ij,ij->i',p[None,:]-x[good],edge[good])/den2[good]
   t=np.clip(t,0,1)
   q=x+t[:,None]*edge
   d=np.einsum('ij,ij->i',p[None,:]-q,p[None,:]-q)
   segmins=np.minimum(segmins,d)
  d2=np.minimum(plane,segmins)
  j=int(np.argmin(d2))
  if float(d2[j])<best:
   best=float(d2[j]);bestrow=lo+j
 return math.sqrt(max(best,0.0)),bestrow
def member_distances(point,surface_map,stable_ids):
 best=(float('inf'),None,None)
 for sid in stable_ids:
  V,F=surface_map[int(sid)]
  d,row=exact_point_triangle_distance(point,np.asarray(V,np.float32)[np.asarray(F,np.int64)])
  if d<best[0]:best=(d,int(sid),int(row))
 return best
baseline_group_mesh={};candidate_group_mesh={}
for gi,g in enumerate(groups):
 sids=list(map(int,g['stable_ids']))
 if not set(sids).issubset(SELECTED):raise ValueError(f'group {gi} uses unselected bank IDs')
 baseline_group_mesh[gi]=welded_group(base_surfaces,sids)
 candidate_group_mesh[gi]=welded_group(candidate_surfaces,sids)
# All groups must truly be closed in the unchanged baseline; otherwise winding
# would be an invalid signed-distance classifier and is explicitly refused.
for gi,(V,F,closed,top) in baseline_group_mesh.items():
 if not closed:raise ValueError(f'baseline group {gi} not closed: {top}')
node_results=[]
for row in witnesses:
 node=int(row['node_id']);p=np.asarray(row['xyz_m'],np.float64)
 expected=[list(x) for x in row['inside_groups']]
 group_results=[]
 for gi,g in enumerate(groups):
  sids=list(map(int,g['stable_ids']))
  Vb,Fb,cb,tb=baseline_group_mesh[gi];Vc,Fc,cc,tc=candidate_group_mesh[gi]
  wb=float(solid_angle_winding(np.asarray([p]),Vb,Fb)[0])
  db,sidb,fb=member_distances(p,base_surfaces,sids)
  dc,sidc,fc=member_distances(p,candidate_surfaces,sids)
  if cc:
   wc=float(solid_angle_winding(np.asarray([p]),Vc,Fc)[0])
   insidec=abs(wc)>0.5
   signedc=(-dc if insidec else dc)
  else:
   wc=None;insidec=None;signedc=None
  insideb=abs(wb)>0.5
  group_results.append({
   'stable_ids':sids,'baseline_group_closed':cb,'candidate_group_closed':cc,
   'baseline_winding':wb,'baseline_inside':insideb,
   'candidate_winding':wc,'candidate_inside':insidec,
   'baseline_nearest_surface':{'distance_m':db,'stable_id':sidb,'face_row':fb,'signed_distance_m':-db if insideb else db},
   'candidate_nearest_surface':{'distance_m':dc,'stable_id':sidc,'face_row':fc,'signed_distance_m':signedc},
   'candidate_topology':tc if not cc else None,
  })
 node_results.append({
  'node_id':node,'xyz_m':list(map(float,np.asarray(row['xyz_m'],np.float32))),
  'physical_o_reported_inside_groups':expected,
  'per_group':group_results,
  'baseline_inside_groups':[groups[i]['stable_ids'] for i,x in enumerate(group_results) if x['baseline_inside']],
  'candidate_inside_groups':[groups[i]['stable_ids'] for i,x in enumerate(group_results) if x['candidate_inside'] is True],
  'candidate_unclassifiable_groups':[groups[i]['stable_ids'] for i,x in enumerate(group_results) if x['candidate_inside'] is None],
 })
summary=[]
for gi,g in enumerate(groups):
 br=[x['per_group'][gi] for x in node_results]
 summary.append({'stable_ids':list(map(int,g['stable_ids'])),'body_indices':list(map(int,g['body_indices'])),
  'faces':int(g['faces']),'baseline_closed':baseline_group_mesh[gi][2],'candidate_closed':candidate_group_mesh[gi][2],
  'baseline_inside_witness_count':sum(x['baseline_inside'] for x in br),
  'candidate_inside_witness_count':sum(x['candidate_inside'] is True for x in br),
  'candidate_unclassifiable_witness_count':sum(x['candidate_inside'] is None for x in br),
  'reported_witness_count':len(g.get('inside_node_ids',[])),
 })
# Restore/provenance correction and 386 receipt lineage are explicit; output is
# diagnostic only and not physical/native admission.
incident=json.loads(INCIDENT.read_text())
report={
 'schema':'numi.human.physical-o-26-witness-source-route-reclassification.v1',
 'status':'diagnostic-only-no-native-or-physical-admission',
 'pins':{path:{'sha256':sha(path),'expected_sha256':expected} for path,expected in EXPECTED.items()},
 'method':{
  'source_route_model':'published 385; route points queried in the pinned source registration against the existing reconstructed NHTISS vertex p coordinates. Only stable-record binding-index/weight lanes are rewritten.',
  'current_emission':'exact 191 rotate/bodyworld expressions and 396 NHTISS decode/validation owner using identity-003 current Float32 body states.',
  'inside_classifier':'frozen224 consistent closed body-group surface topology after exact Float32 coordinate welding; solid-angle winding, inside iff abs(winding)>0.5.',
  'distance':'exact point-to-triangle closest distance over every face in each selected closed group; signed only when that group is closed and winding classifies point.',
  'mesh_policy':'current reconstructed original 8874 geometry/topology is retained. The raw 385 subset recompile is used only to verify stable source-member identity and expose historical topology/count differences; it is not substituted as geometry.',
 },
 'source_geometry_lineage':payload_lineage,
 'route_candidate':{
  'stable_ids':sorted(SELECTED),'changed_binding_weight_records_by_stable_id':changed_counts,
  'stable132_lanes_match_386_candidate_byte_exact':stable132_exact,
  'all12_candidate_191_vs_baseline_metrics':baseline_matches,
  'all_target_positions_normals_faces_and_binding_transforms_unchanged':True,
  'all_nonselected_bytes_unchanged':True,
  'original386_restore':{
    'candidate_sha256':sha(C386),
    'receipt_sha256':sha(C386_RECEIPT),
    'receipt_original_hash_not_recovered':'44f2ae1d2fd216d6864e1810c4c99ed11b190981dc900da4c337606d747da022',
    'receipt_restored_from_rebuild_not_byte_identical':incident['386_restoration']['receipt']['status'],
    'incident_receipt_sha256':sha(INCIDENT),
  },
 },
 'physical_o_witness_source':{'candidate_npz_sha256':sha(OUTER),'witness_json_sha256':sha(WITNESS),'witness_count':len(witnesses),'all_node_ids_and_F32_coordinates_match_candidate_npz':True},
 'group_topology':summary,
 'per_node_winding_and_distance':node_results,
 'limitations':['The 26 points are physical outer-surface witnesses from a diagnostic unsolved candidate, not an admitted tissue state.','The route reweight candidate is a static source-route counterfactual, not a contact or mechanical repair.','Only the 12 witness-associated banks/groups were classified; this is not the full 689-bank audit.','Raw source recompile counts can differ from the historical reconstructed payload; raw recompile geometry was not used.'],
}
OUT.mkdir(parents=True,exist_ok=False)
p=OUT/'physical-o-26-route-reclassification.json'
tmp=OUT/'physical-o-26-route-reclassification.json.tmp'
tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,p)
print(json.dumps({'report':str(p),'sha256':sha(p),'summary':summary,
 'baseline_inside':sum(len(x['baseline_inside_groups']) for x in node_results),
 'candidate_inside':sum(len(x['candidate_inside_groups']) for x in node_results),
 'candidate_unclassifiable':sum(len(x['candidate_unclassifiable_groups']) for x in node_results),
 'stable132_386_byte_exact':stable132_exact},indent=2),flush=True)
