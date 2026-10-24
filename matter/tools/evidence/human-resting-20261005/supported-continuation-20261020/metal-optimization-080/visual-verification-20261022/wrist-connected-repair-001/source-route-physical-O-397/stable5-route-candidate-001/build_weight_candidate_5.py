from pathlib import Path
import hashlib, importlib.util, json, struct, sys
import numpy as np
ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=ROOT/'source-route-physical-O-397/stable5-route-candidate-001/output'
MODEL=ROOT/'source-route-weighting-owner-385/src/numilab_human/model.py'
ARTIFACT=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/recovered-artifact')
REG=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/brachio109-local810-existing-full-parent-source-853b/assets/candidate-current-source.json')
MANIFEST=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json')
PAYLOAD=MANIFEST.with_name('bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
if OUT.exists(): raise SystemExit(f'refusing existing output directory: {OUT}')
def pinned_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
EXPECTED_INPUTS={
 'PAYLOAD':'8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042',
 'MANIFEST':'ffe60f996e92325ce21123da86e259a02a9aaed1a874373a0e71721a2ceba07b',
 'REG':'31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c',
 'MODEL':'5bd314fa7ceaebc6c0e72a0115befdda7a19e11b081bc2009cd2a1e94ce01483',
}
for name,expected in EXPECTED_INPUTS.items():
 actual=pinned_sha(globals()[name])
 if actual!=expected: raise SystemExit(f'{name} pin mismatch: expected {expected}, got {actual}')
WANTED={5}
spec=importlib.util.spec_from_file_location('route385',MODEL); model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)
reg=json.loads(REG.read_text()); manifest=json.loads(MANIFEST.read_text()); original=PAYLOAD.read_bytes(); outbuf=bytearray(original)
source_sha=reg['source']['myosim']['source']['archive_sha256']
bodies,route_muscles,_=model._myosim_surface_route_context(ARTIFACT,source_sha)
body_local={}
for a in reg['anchors']:
 name=a['target']['name']; body_local.setdefault(name,a['registration']['source_obj_mm_to_core_inertial_body_m'])
G=np.asarray(reg['coordinate_system']['global_source_mm_to_myosim_world_m'],dtype=float)
magic,abi,surface_count,binding_count,vertex_count,index_count,regfp,srcfp=struct.unpack_from('<8s6I32s',original,0)
assert magic==b'NHTISS4\0' and abi==5 and srcfp.hex()==source_sha
rec_off=64; records=[struct.unpack_from('<8I',original,rec_off+32*i) for i in range(surface_count)]
bind_off=rec_off+32*surface_count; bindrecs=[struct.unpack_from('<I8f',original,bind_off+36*i) for i in range(binding_count)]
vert_off=bind_off+36*binding_count; index_off=vert_off+56*vertex_count
indices=np.frombuffer(original,dtype='<u4',count=index_count,offset=index_off)
patched_ranges=[]; row_reports=[]
def pack_weights(weights):
 w=np.asarray(weights,dtype=float)
 if w.ndim!=1 or not np.isfinite(w).all() or np.min(w)<-1e-12: raise ValueError('invalid full binding weights')
 w=np.maximum(w,0.0); nz=np.flatnonzero(w>1e-12)
 if len(nz)==0: raise ValueError('no positive weight')
 order=sorted(nz.tolist(),key=lambda i:(-w[i],i))[:4]
 v=w[order]; v=v/v.sum(); f=np.asarray(v,dtype=np.float32); f=np.asarray(f.astype(float)/f.astype(float).sum(),dtype=np.float32)
 ids=order+[0xffffffff]*(4-len(order)); vals=np.r_[f,np.zeros(4-len(f),dtype=np.float32)]
 if abs(float(vals.sum(dtype=np.float64))-1.0)>2e-7: raise ValueError('packed weights not unit')
 return ids,vals.tolist()
def face_metrics(old,new,faces):
 a=old[faces];b=new[faces]
 no=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]); nn=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
 ao=np.linalg.norm(no,axis=1); an=np.linalg.norm(nn,axis=1)
 ratio=an/np.maximum(ao,1e-300); cos=np.einsum('ij,ij->i',no,nn)/np.maximum(ao*an,1e-300)
 return {'face_count':len(faces),'min_area_ratio':float(ratio.min()),'p01_area_ratio':float(np.quantile(ratio,.01)),'min_normal_cosine':float(cos.min()),'zero_area_faces':int(np.count_nonzero(an==0)),'reversed_faces':int(np.count_nonzero(cos<=0))}
def stats(x):
 x=np.asarray(x,float)
 return {'p50':float(np.quantile(x,.5)),'p95':float(np.quantile(x,.95)),'max':float(x.max())}
for rec in records:
 firstb,nb,firstv,nv,firsti,ni,sid,layer=rec
 if sid not in WANTED: continue
 sm=next(x for x in manifest['source']['surfaces'] if x['stable_id']==sid)
 names=[x['myosim_body'] for x in sm['body_bindings']]
 matched=[x['name'] for x in sm['matched_muscles']]
 routes=[p for name in matched for p in route_muscles[name]['route_points']]
 corrected_routes=model._bodyparts_myosim_route_points_in_source_registration(routes,body_local,bodies,G.tolist(),f'stable {sid} candidate')
 assert len(corrected_routes)==len(routes) and all('source_registration_query_world_m' in p and 'world_m' in p for p in corrected_routes)
 local_v=np.asarray([struct.unpack_from('<3f',original,vert_off+56*j) for j in range(firstv,firstv+nv)],float)
 q_world=(local_v*1000.0)@np.asarray(G[:3,:3],float).T+np.asarray(G[:3,3],float)
 old_w=np.zeros((nv,len(names))); new_w=np.zeros_like(old_w)
 changed_v=changed_i=changed_w=0; max_delta=0.; nearest=[]; active=[]
 for li in range(nv):
  pos=vert_off+56*(firstv+li); vi=struct.unpack_from('<6f4I4f',original,pos)
  for lane in range(4):
   bi=vi[6+lane]; wt=vi[10+lane]
   if bi!=0xffffffff and wt>0:
    if bi>=len(names): raise ValueError(f'{sid}: invalid existing local binding {bi}')
    old_w[li,bi]+=wt
  nw,dist,nactive=model._bodyparts_myosim_route_weights_for_world_point(q_world[li].tolist(),names,corrected_routes,f'stable {sid} vertex {li}')
  nw=np.asarray(nw,dtype=float); new_w[li]=nw; nearest.append(dist);active.append(nactive)
  ids,vals=pack_weights(nw); oldids=list(vi[6:10]); oldvals=list(vi[10:14])
  idxchanged=oldids!=ids
  wtchanged=any(struct.pack('<f',float(a))!=struct.pack('<f',float(b)) for a,b in zip(oldvals,vals))
  changed_v+=int(idxchanged or wtchanged);changed_i+=int(idxchanged);changed_w+=int(wtchanged);max_delta=max(max_delta,float(np.max(np.abs(old_w[li]-nw))))
  struct.pack_into('<4I',outbuf,pos+24,*ids);struct.pack_into('<4f',outbuf,pos+40,*vals)
 per_binding=[]
 for bj,name in enumerate(names):
  tr=bindrecs[firstb+bj]; trans=np.asarray(tr[1:4],float); quat=tr[4:8]; scale=float(tr[8])
  localR=np.asarray(model._myosim_matrix_from_quaternion_xyzw(list(quat)),float)
  body=bodies[name]; bp=np.asarray(body['default_com_position_world_m'],float)
  bodyR=np.asarray(model._myosim_matrix_from_quaternion_xyzw(list(body['default_inertial_quaternion_world_xyzw'])),float)
  per_binding.append(bp[None,:]+(trans[None,:]+scale*(local_v@localR.T))@bodyR.T)
 per_binding=np.stack(per_binding,axis=1)
 render_old=np.einsum('nb,nbc->nc',old_w,per_binding); render_new=np.einsum('nb,nbc->nc',new_w,per_binding)
 disp=np.linalg.norm(render_new-render_old,axis=1)*1000
 if ni%3: raise ValueError(f'{sid}: index count is not triangular')
 ix=indices[firsti:firsti+ni].astype(np.int64)
 if len(ix) and ix.max()<nv: faces=ix.reshape(-1,3)
 elif len(ix) and ix.min()>=firstv and ix.max()<firstv+nv: faces=(ix-firstv).reshape(-1,3)
 else: raise ValueError(f'{sid}: invalid face indices {ix.min()}..{ix.max()}')
 digit={}
 for name in names:
  if any(t in name for t in ('fourthmc','4proxph','midph4','distph4','fifthmc','5proxph','midph5','distph5')):
   k=names.index(name);digit[name]={'old_nonzero':int(np.count_nonzero(old_w[:,k]>1e-7)),'new_nonzero':int(np.count_nonzero(new_w[:,k]>1e-7)),'old_mean':float(old_w[:,k].mean()),'new_mean':float(new_w[:,k].mean())}
 row_reports.append({'stable_id':sid,'member_id':sm['member_id'],'label':sm['label'],'vertex_count':nv,'face_count':len(faces),'body_bindings':names,'route_names':matched,
   'route_query_distance_m':stats(nearest),'active_route_body_count_minmax':[int(min(active)),int(max(active))],
   'changed_vertices':changed_v,'vertices_with_binding_index_changes':changed_i,'vertices_with_weight_changes':changed_w,'max_abs_fullvector_weight_delta':max_delta,
   'default_body_pose_displacement_mm':stats(disp),'default_body_pose_triangle_change':face_metrics(render_old,render_new,faces),'digit_binding_counts':digit})
 patched_ranges.append((vert_off+56*firstv,vert_off+56*(firstv+nv)))
candidate=OUT/'corrected-route-weights-candidate-5.nhtissue'
allowed=bytearray(len(original))
for lo,hi in patched_ranges:
 for j in range(lo,hi,56): allowed[j+24:j+56]=b'\1'*32
for i,(a,b) in enumerate(zip(original,outbuf)):
 if a!=b and not allowed[i]: raise AssertionError(f'unintended byte difference {i}')
checks=[]
for rec in records:
 fb,nb,fv,nv,fi,ni,sid,layer=rec
 if sid not in WANTED: continue
 sm=next(x for x in manifest['source']['surfaces'] if x['stable_id']==sid); nbind=len(sm['body_bindings']); sums=[]
 for j in range(fv,fv+nv):
  v=struct.unpack_from('<6f4I4f',outbuf,vert_off+56*j);ids=v[6:10];ws=v[10:14]
  if abs(sum(ws)-1)>2e-7 or any(w<0 or (w>0 and (i==0xffffffff or i>=nbind)) for i,w in zip(ids,ws)): raise AssertionError(f'invalid candidate row {sid}/{j}')
  sums.append(sum(ws))
 checks.append({'stable_id':sid,'weight_sum_min':min(sums),'weight_sum_max':max(sums)})
report={'schema':'numi.human.nhtiss.route-weight-counterfactual.v1','source_payload_path':str(PAYLOAD),'source_payload_sha256':hashlib.sha256(original).hexdigest(),'source_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
 'registration_sha256':hashlib.sha256(REG.read_bytes()).hexdigest(),'source_archive_sha256':source_sha,'producer_helper_path':str(MODEL),'producer_helper_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
 'candidate_path':str(candidate),'candidate_sha256':hashlib.sha256(outbuf).hexdigest(),'candidate_size_bytes':len(outbuf),'abi':abi,'rows':row_reports,
 'checks':{'all_nonweight_bytes_unchanged':True,'only_selected12surface_lane_records_changed':True,'xyz_normals_faces_binding_transforms_unselected_rows_exact':True,'weight_sums_and_local_binding_indices_valid':checks},
 'scope':'read-only counterfactual candidate for review; only weight lane binding-index/value records of stable row 5 changed. No anatomical, clearance, dynamic or owner qualification is implied.'}
if {int(r['stable_id']) for r in row_reports}!=WANTED: raise AssertionError('selected stable-ID coverage mismatch')
for name,expected in EXPECTED_INPUTS.items():
 if pinned_sha(globals()[name])!=expected: raise SystemExit(f'{name} changed during build; no candidate emitted')
OUT.mkdir(parents=True,exist_ok=False)
candidate.write_bytes(outbuf)
rp=OUT/'candidate-receipt-5.json';rp.write_text(json.dumps(report,indent=2)+'\n')
print('CANDIDATE',candidate,'SHA256',hashlib.sha256(outbuf).hexdigest(),'SIZE',len(outbuf),flush=True)
print('RECEIPT',rp,'SHA256',hashlib.sha256(rp.read_bytes()).hexdigest(),flush=True)
for row in row_reports: print(json.dumps(row,separators=(',',':')),flush=True)

