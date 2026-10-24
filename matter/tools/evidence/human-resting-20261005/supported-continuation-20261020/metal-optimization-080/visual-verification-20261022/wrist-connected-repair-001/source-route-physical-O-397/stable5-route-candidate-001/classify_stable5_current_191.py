#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, math, runpy, sys
from pathlib import Path
from collections import Counter
import numpy as np

E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BASE=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
MAN=BASE.with_name('bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json')
REG=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/brachio109-local810-existing-full-parent-source-853b/assets/candidate-current-source.json')
MODEL=E/'source-route-weighting-owner-385/src/numilab_human/model.py'
CAND=E/'source-route-physical-O-397/stable5-route-candidate-001/output/corrected-route-weights-candidate-5.nhtissue'
CRECEIPT=CAND.with_name('candidate-receipt-5.json')
POSE=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
CACHE_DIR=E/'registered-surrounding-tissue-native013-191'
INV=CACHE_DIR/'inventory.json';CACHE=CACHE_DIR/'registered-tissue-banks.npz';EMITTER=CACHE_DIR/'prepare_registered_surrounding.py'
REFERENCE=E/'source-route-frame-owner-396/tools/rebind_human_tissue_reference.py'
OUTER=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/candidate.npz'
WITNESS=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/p1-distance-initializer-393/physical-outer-inside-witnesses.json'
PRE=Path('/private/tmp/preflight129-d36-O-import-155-candidate-build-008-functions-only-014.py')
AUDIT=E/'joint-region-roi-014/audit_exact_joint_region_intersections_015.py'
BONE_MAN=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
OUT=E/'source-route-physical-O-397/stable5-route-candidate-001/classification-results'
EXPECTED={
 str(BASE):'8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042',
 str(MAN):'ffe60f996e92325ce21123da86e259a02a9aaed1a874373a0e71721a2ceba07b',
 str(REG):'31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c',
 str(MODEL):'5bd314fa7ceaebc6c0e72a0115befdda7a19e11b081bc2009cd2a1e94ce01483',
 str(CAND):'2faedf570d6c3481c450e7c390934e186a96467e256188b93cd79cb8c3f82e37',
 str(CRECEIPT):'ec5f6525ccbfd2047c6b0aa9e68c8c66b4cc24e7494a149bde0579464014520b',
 str(POSE):'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd',
 str(INV):'b0823cf8f33df2586ba1cd45e5b260cd75d1e35b8d2682b67c15f64ea22bbcc7',
 str(CACHE):'39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792',
 str(EMITTER):'3606321c1265fd1350afb2c1ce765ce7f7afe99d65f0cebfd77a4504c58d235b',
 str(REFERENCE):'7adf9cb321168e859f21ae62cf79ee59d94ce50fffbc0aeac2971feb18defa83',
 str(OUTER):'e70b3e08aa385c8a321ebb0849257ec4afa04d13050e26fc28647d36146d26ba',
 str(WITNESS):'e554b1466b58020c7a27d0ce591e3e2f35b4d597525e0e8b71bec7a155c48c82',
 str(PRE):'16dc19f9331ae5505519e93b218dcc7c937a0ffa408ec8826e7fab8691831501',
 str(AUDIT):'aef02f000fcf7ae4ea85835a4e8599963fc70c3551f8b2706752d6d68630a6f1',
 str(E/'audit_bilateral_skin_retained_poses_002.py'):'5d13ce7a8fff3470b2eb84a298fb7b8652cfa0130bb71cacee3756936b949a86',
 str(E/'source-route-physical-O-397/stable5-route-candidate-001/build_weight_candidate_5.py'):'38ad66666270bd2e46df3e15d3f793a539cbc4395e528c338238db437f566873',
 '/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/cardiac_cavity_intersections.py':'05ba7856fc373f55a1270eb5f3ef887a4eac493ad9fb4590a618caee952d2a32',
 '/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/compiled_quotient_embeddedness.py':'e14793d27e014e23117967a2f486415157d5ae494db342ace0ac701dd1bb5361',
 str(BONE_MAN):'8c744849c2cc76884b7819488389b9706a2d40d9102e9464e8f4b73db10ec5d4',
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for p,w in EXPECTED.items():
 a=sha(p)
 if a!=w: raise SystemExit(f'pin mismatch {p}: {a} != {w}')
if OUT.exists(): raise SystemExit(f'refusing existing output folder {OUT}')
# Exact 191 source-current emission owner.
spec=importlib.util.spec_from_file_location('stable5_reference_396',REFERENCE)
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
base=ref.decode_nhtiss(BASE.read_bytes());cand=ref.decode_nhtiss(CAND.read_bytes())
pose_data=json.loads(POSE.read_text());pose=ref.load_pose_map(pose_data,base['registration'])
inventory=json.loads(INV.read_text());cache=np.load(CACHE,allow_pickle=False)
bankmap=ref.validate_inventory(inventory,sha(BASE),sha(POSE),sha(CACHE),base['nr'])
rotate,bodyworld=ref.load_exact_191_helpers(EMITTER,EXPECTED[str(EMITTER)],pose)
rows={int(r[6]):r for r in base['records']}
if 5 not in rows: raise ValueError('stable 5 missing from historical reconstructed payload')
record=rows[5];fb,nb,fv,nv,fi,ni,sid,layer=map(int,record)
cachemeta=bankmap[5];key=cachemeta['key']
for p,label in ((base,'baseline'),(cand,'385-corrected')):
 if (p['magic'],p['abi'],p['nr'],p['nb'],p['nv'],p['ni'],p['registration'],p['source_id'])!=(base['magic'],base['abi'],base['nr'],base['nb'],base['nv'],base['ni'],base['registration'],base['source_id']):raise ValueError(label+' header mismatch')
 if not np.array_equal(p['indices'],base['indices']) or not np.array_equal(p['vertices']['p'],base['vertices']['p']) or not np.array_equal(p['vertices']['n'],base['vertices']['n']) or not np.array_equal(p['bindings']['transform'],base['bindings']['transform']):raise ValueError(label+' changed geometry/topology/transforms')
f=base['indices'][fi:fi+ni].astype(np.int64)
if f.min()>=fv and f.max()<fv+nv:f=(f-fv).reshape(-1,3)
elif f.min()>=0 and f.max()<nv:f=f.reshape(-1,3)
else:raise ValueError('stable5 face indexing invalid')
vb,nbaseline=ref.emit_current(base,record,pose,rotate,bodyworld)
vc,ncand=ref.emit_current(cand,record,pose,rotate,bodyworld)
cacheV=np.asarray(cache[key+'_vertices'],np.float32);cacheF=np.asarray(cache[key+'_faces'],np.int64)
if not np.array_equal(vb,cacheV):raise ValueError('stable5 baseline not exact current cache replay')
if not np.array_equal(f,cacheF):raise ValueError('stable5 topology/cache mismatch')
def mesh_info(V,F):
 tri=np.asarray(V,np.float32)[:len(vb)][np.asarray(F,np.int64)]
 area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
 specw=importlib.util.spec_from_file_location('bone_preflight_014',PRE)
 pre=importlib.util.module_from_spec(specw);sys.modules[specw.name]=pre;specw.loader.exec_module(pre)
 # build_bones uses this globals map, pinned to the same identity-003 current body states.
 pre.POSE=POSE
 bones=pre.build_bones('right',np.asarray(V,np.float32))
 return {'face_count':int(len(F)),'zero_area_faces':int(np.count_nonzero(area2==0)),'min_double_area_m2':float(area2.min()),'bones':bones,'pre':pre}
# Use the existing closed-bank winding predicate used by the physical-O initializer.
# Stable 5 exact current 191 is expected to be closed; enforce this rather than
# converting an open bank's winding into an inside/outside assertion.
def closed_and_winding(V,F,point):
 # exact Float32 coordinate weld, preserve face order/orientation
 mp={};qv=[];remap=np.empty(len(V),np.int64)
 for i,p in enumerate(np.asarray(V,np.float32)):
  k=tuple(float(x) for x in p)
  j=mp.get(k)
  if j is None:j=len(qv);mp[k]=j;qv.append(k)
  remap[i]=j
 Fw=remap[np.asarray(F,np.int64)]
 edges=np.vstack((Fw[:,[0,1]],Fw[:,[1,2]],Fw[:,[2,0]]));edges.sort(axis=1)
 h=Counter(map(tuple,edges))
 if any(c!=2 for c in Counter(h.values())):return False,None,{'edge_incidence':dict(Counter(h.values()))}
 wind=0.0
 p=np.asarray(point,np.float64);VV=np.asarray(qv,np.float64);tri=VV[Fw]
 a=tri[:,0]-p;b=tri[:,1]-p;c=tri[:,2]-p;la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
 num=np.einsum('ij,ij->i',a,np.cross(b,c));den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
 wind=float(np.arctan2(num,den).sum()/(2*np.pi))
 return True,wind,{'welded_vertices':len(qv),'face_count':len(Fw),'edge_incidence':{'2':len(h)}}
def point_triangle_distance(p,T):
 p=np.asarray(p,np.float64);T=np.asarray(T,np.float64);best=float('inf');bestrow=-1
 for lo in range(0,len(T),4096):
  tri=T[lo:lo+4096];a=tri[:,0];b=tri[:,1];c=tri[:,2];ab=b-a;ac=c-a;ap=p-a
  d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac);d20=np.einsum('ij,ij->i',ap,ab);d21=np.einsum('ij,ij->i',ap,ac);den=d00*d11-d01*d01
  ok=den>np.finfo(float).tiny;v=np.zeros(len(tri));w=np.zeros(len(tri));v[ok]=(d11[ok]*d20[ok]-d01[ok]*d21[ok])/den[ok];w[ok]=(d00[ok]*d21[ok]-d01[ok]*d20[ok])/den[ok];u=1-v-w
  inside=ok&(u>=0)&(v>=0)&(w>=0);proj=a+v[:,None]*ab+w[:,None]*ac
  d2=np.einsum('ij,ij->i',p-proj,p-proj);d2[~inside]=np.inf
  for x,y in ((a,b),(b,c),(c,a)):
   e=y-x;ee=np.einsum('ij,ij->i',e,e);t=np.zeros(len(tri));good=ee>np.finfo(float).tiny;t[good]=np.einsum('ij,ij->i',(p-x[good]),e[good])/ee[good];t=np.clip(t,0,1);q=x+t[:,None]*e;d2=np.minimum(d2,np.einsum('ij,ij->i',p-q,p-q))
  j=int(np.argmin(d2))
  if float(d2[j])<best:best=float(d2[j]);bestrow=lo+j
 return math.sqrt(max(best,0.0)),bestrow
# Load source physical O point and prove the tet edge interpolation witness.
outer=np.load(OUTER,allow_pickle=False);ov=np.asarray(outer['points_world_m'],np.float32)
a=ov[9451].astype(np.float64);b=ov[9473].astype(np.float64);alpha=0.7473772446
point=(alpha*a+(1-alpha)*b)
reported=np.asarray([-0.04637068259,-1.49376714333,0.10053062839],np.float64)
if np.linalg.norm(point-reported)>2e-8:raise ValueError('stable5 witness edge interpolation does not match supplied point')
row2562=np.asarray(vb,np.float32)[f[2562]]
# Load bones from exact identity-003 current body transforms and audit surface intersections.
import sys as _sys
_sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
_sys.path.insert(0,str(E))
aud=runpy.run_path(str(AUDIT))
audit=runpy.run_path(str(E/'audit_bilateral_skin_retained_poses_002.py'))
for name in ('coordinate_quotient','_aabb_candidate_pairs','triangle_intersection_points','_allowed_shared_point'):
 aud[name]=audit[name]
muscle_meta=mesh_info(np.vstack((vb,vc)),f);bones=muscle_meta['bones'];pre=muscle_meta['pre']
boneV=[];boneF=[];bone_owner=[];off=0
for name,(V,Fb) in sorted(bones.items()):
 boneV.append(np.asarray(V,np.float32));boneF.append(np.asarray(Fb,np.int64)+off);bone_owner.extend([name]*len(Fb));off+=len(V)
if boneV:
 Bv=np.vstack(boneV);Bf=np.vstack(boneF)
 def strict_cross_pairs(Va,Fa,Vb,Fb):
  pa,qfa=audit['prep'](np.asarray(Va,np.float32),np.asarray(Fa,np.int64))
  ra,da=audit['records_nonzero'](pa,qfa,np.arange(len(Fa),dtype=np.int64))
  pb,qfb=audit['prep'](np.asarray(Vb,np.float32),np.asarray(Fb,np.int64))
  rb,db=audit['records_nonzero'](pb,qfb,np.arange(len(Fb),dtype=np.int64))
  shift=len(pa);rb=[(r[0],r[1],r[2],r[3],tuple(int(i)+shift for i in r[4])) for r in rb]
  pairs=audit['_aabb_candidate_pairs'](ra,rb,same_surface=False);hits=[];hit_count=0;candidate_count=0
  for aa,bb in pairs:
   candidate_count+=1
   intersections=audit['triangle_intersection_points'](aa[0],bb[0])
   if intersections:
    hit_count+=1
    if len(hits)<32:hits.append({'muscle_face_row':int(aa[3]),'bone_face_row':int(bb[3]),'intersection_point_count':len(intersections)})
  return {'candidate_triangle_pairs':candidate_count,'intersection_pair_count':hit_count,
   'muscle_degenerate_faces':int(da),'bone_degenerate_faces':int(db),'first_witnesses':hits,
   'shared_vertex_exemptions':0,'separate_vertex_id_domains':True}
 exactbone_candidate=strict_cross_pairs(vc,f,Bv,Bf)
 exactbone_baseline=strict_cross_pairs(vb,f,Bv,Bf)
else:
 Bv=np.empty((0,3),np.float32);Bf=np.empty((0,3),np.int64)
 exactbone_candidate=exactbone_baseline={'candidate_triangle_pairs':0,'intersection_pair_count':0,'muscle_degenerate_faces':0,'bone_degenerate_faces':0,'first_witnesses':[],'shared_vertex_exemptions':0,'separate_vertex_id_domains':True}
def surface_point_distances(point,V,F):
 d,row=point_triangle_distance(point,np.asarray(V,np.float32)[np.asarray(F,np.int64)])
 return {'distance_m':d,'nearest_face_row':row}
point_results=[]
for label,V in (('baseline',vb),('corrected385',vc)):
 closed,winding,top=closed_and_winding(V,f,point)
 dist,fr=point_triangle_distance(point,np.asarray(V,np.float32)[f])
 point_results.append({'surface':label,'closed':closed,'winding':winding,'inside':None if winding is None else abs(winding)>0.5,
  'point_to_surface_distance_m':dist,'nearest_face_row':fr,'topology':top,
  'emitted_vertex_count':len(V),'emitted_face_count':len(f)})
# Exact query from O witness point to all collected nearby bone triangles; plus a nearest
# vertex-to-bone query from the muscle surface. Point queries scan every face.
point_bone=[]
for label,V in (('O_edge_witness',point[None,:]),):
 if not len(Bf):
  point_bone.append({'query':label,'status':'no nearby bones selected'})
  continue
 tri=Bv[Bf]
 if label=='O_edge_witness':
  d,fr=point_triangle_distance(point[0] if False else point,tri)
  point_bone.append({'query':label,'nearest_bone_distance_m':d,'bone_face_row':fr,'bone_owner':bone_owner[fr]})
 else:
  # Existing 014 bone-distance query is retained and explicitly described as
  # a centroid-k=64 accelerated vertex probe, not a full triangle-pair clearance.
  d,sd,fr,bary,q,owner,bv_comb,bf_comb=pre.bone_distance(V,bones)
  point_bone.append({'query':label,'method':'existing 014 centroid-k=64 nearest-surface query at each muscle vertex',
   'vertex_count':len(V),'minimum_unsigned_vertex_to_bone_distance_m':float(np.min(d)),
   'minimum_signed_vertex_to_bone_distance_m':float(np.min(sd)),
   'nearest_bone_face_row':int(fr[int(np.argmin(d))]),'nearest_bone_owner':str(owner[int(np.argmin(d))]),
   'inside_bone_vertex_count':int(np.count_nonzero(sd<=0))})
# Compare accelerated vertex-to-bone clearance for both current surfaces.
for label,V in (('baseline_muscle_vertices',vb),('corrected385_muscle_vertices',vc)):
 if not len(Bf):
  point_bone.append({'query':label,'status':'no nearby bones selected'})
 else:
  d,sd,fr,bary,q,owner,bv_comb,bf_comb=pre.bone_distance(V,bones)
  point_bone.append({'query':label,'method':'existing 014 centroid-k=64 nearest-surface query at each muscle vertex',
   'vertex_count':len(V),'minimum_unsigned_vertex_to_bone_distance_m':float(np.min(d)),
   'minimum_signed_vertex_to_bone_distance_m':float(np.min(sd)),
   'nearest_bone_face_row':int(fr[int(np.argmin(d))]),'nearest_bone_owner':str(owner[int(np.argmin(d))]),
   'inside_bone_vertex_count':int(np.count_nonzero(sd<=0))})
# Baseline and candidate bank 5 geometry statistics/route changes.
tri0=vb[f].astype(np.float64);tri1=vc[f].astype(np.float64)
n0=np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]);n1=np.cross(tri1[:,1]-tri1[:,0],tri1[:,2]-tri1[:,0]);area0=np.linalg.norm(n0,axis=1);area1=np.linalg.norm(n1,axis=1);cos=np.einsum('ij,ij->i',n0,n1)/np.maximum(area0*area1,1e-300)
# Compute maximum and quantiles for emitted bank motion.
move=np.linalg.norm((vc-vb).astype(np.float64),axis=1)*1000.0
report={'schema':'numi.human.stable5-soleus-corrected-route-witness-classification.v1',
 'status':'diagnostic-only-no-native-or-physical-admission',
 'pins':{p:{'sha256':sha(p),'expected_sha256':want} for p,want in EXPECTED.items()},
 'pose':{'name':'identity-003 current','pose_sha256':sha(POSE),'emitter_sha256':sha(EMITTER),'emitter_owner':'exact 191 registered current replay'},
 'source_identity':{'stable_id':5,'inventory_key':'bank_0004','member_id':'FJ1437','label':'right soleus','body_bindings':['tibia_r','calcn_r'],'body_indices':[136,138],
  'historical_payload_vertices':nv,'historical_payload_faces':int(len(f)),'raw_source_manifest_member_sha256':next(x['member_sha256'] for x in json.loads(MAN.read_text())['source']['surfaces'] if int(x['stable_id'])==5),
  'raw_source_geometry_substituted':False,'payload_source_xyz_normals_topology_transforms_exact':True},
 'candidate_route_changes':{'changed_vertex_records':2674,'changed_binding_index_rows':506,'max_weight_delta':0.34086715736109396,
  'default_pose_displacement_reported_by_builder_mm':{'p50':9.344096433287454,'p95':16.02770702328939,'max':17.6492249548635},
  'identity003_current_emitted_displacement_mm':{'p50':float(np.quantile(move,.5)),'p95':float(np.quantile(move,.95)),'max':float(move.max())},
  'zero_area_faces_corrected_current':int(np.count_nonzero(area1==0)),
  'faces_reversed_vs_baseline_current':int(np.count_nonzero(cos<=0)),
  'min_current_face_area_ratio_vs_baseline':float(np.min(area1/np.maximum(area0,1e-300)))},
 'physical_o_witness':{'tetrahedron_row':88190,'source_o_node_ids':[9451,9473],
  'edge_barycentric_weights':[alpha,1-alpha],'reported_xyz_m':reported.tolist(),'reconstructed_xyz_m':point.tolist(),
  'interpolated_point_matches_reported_within_20nm':True,
  'historical_obstacle_face_row':2562,'historical_obstacle_face_xyz_m':row2562.astype(float).tolist(),'historical_obstacle_face_centroid_m':row2562.astype(np.float64).mean(axis=0).tolist(),'historical_obstacle_record':['bank_0004','stable5','right soleus']},
 'per_surface_point_classification':point_results,
 'point_to_bone_and_surface_intersection':{'bone_manifest_sha256':sha(BONE_MAN),'bone_surface_names_selected':sorted(bones),
  'combined_bone_faces':int(len(Bf)),'exact_bank5_vs_nearby_bones':{'baseline':exactbone_baseline,'corrected385':exactbone_candidate},'distance_queries':point_bone},
 'bone_distance_limit':'Exact triangle intersections use the existing 264 predicate via the retained 294 surface-audit owner. The edge point-to-bone nearest query scans all selected bone faces. The muscle-vertex clearance is an existing centroid-k64 accelerated nearest-surface diagnostic, not a global all-triangle-pair distance proof.',
 'limitations':['Only the reported stable5 soleus obstacle and two O-edge witnesses were examined; no other bank or the full 689-bank obstacle set was re-audited.','Route reweighting changes the current routed muscle surface substantially while source p/n/faces and body transforms remain fixed. It is not a physical material remesh or admission.','No native application solve or accepted-state run was performed.']}
OUT.mkdir(parents=True,exist_ok=False)
p=OUT/'stable5-soleus-current-witness-classification.json';tmp=OUT/(p.name+'.tmp')
tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');tmp.replace(p)
print(json.dumps({'report':str(p),'sha256':sha(p),'point_results':point_results,'bones':sorted(bones),'bone_face_count':len(Bf),'exact_bone_pairs':exactbone},indent=2),flush=True)
