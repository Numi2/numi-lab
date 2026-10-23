import sys,json,hashlib,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/"finger-profile-069i-left-ring-web-retreat"
P069=E/"finger-profile-069h-right-scale020-preference-waiver"
P060=E/"finger-profile-060"
FAILPOSE=E/"hand-actuation-qualification-001/results/scoped-telemetry004-bilateral-wrist-flexion-reference-initial-80-step10ms/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell-muscle-driven-selected-actuators-source-support-contact-posed.skin-poses.json"
RETAINED=E/"native-pose-checks/repair-005"
sys.path.insert(0,str(E));sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.skin_source_payload_preflight import decode_payload
ns={};exec((E/"audit_bilateral_skin_reweighted_candidate_016.py").read_text().split("def main():")[0],ns)
h={};exec((E/"wrist-loft-rebuild-019/audit_wrist_bones035.py").read_text().split("report={'schema':")[0],h)
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha_file(p):return sha_bytes(Path(p).read_bytes())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def as_pose(p):return json.loads(Path(p).read_text())
raw0=(P060/"replacement-bilateral-hands038.nhskin").read_bytes();rawh=(P069/"replacement-bilateral-hands038.nhskin").read_bytes()
D0=decode_payload(raw0);Dh=decode_payload(rawh)
assert np.array_equal(D0["indices"],Dh["indices"])
assert np.array_equal(D0["bindings_f"],Dh["bindings_f"]) and np.array_equal(D0["bindings_u"],Dh["bindings_u"]) and np.array_equal(D0["full_weights"],Dh["full_weights"])
F=np.asarray(Dh["indices"],int).reshape(-1,3); nvert=Dh["vertex_count"]
changed=np.asarray([494,1069,2411,3973],int)
assert np.all(np.isin(changed,np.asarray(np.load(P069/"replacement-geometry.npz")["patch_vertices"],int)))
leftpatch0=np.load(P069/"left-patch.npz",allow_pickle=False); leftslots=np.asarray(leftpatch0["source_slots"],int)
assert all(np.sum(leftslots==v)==1 for v in changed)
ids_faces=np.asarray(leftpatch0["old_faces"],int)[[3072,7043]]
assert ids_faces.tolist()==[87790,91761]
target_faces=np.flatnonzero(np.any(np.isin(F,changed),axis=1))
assert set(ids_faces).issubset(set(target_faces))
# The eight retained native wrist poses are the exact poses used by the 069h retained audit.
retained=sorted(RETAINED.glob("*/*.skin-poses.json"),key=lambda p:("neutral" not in str(p),str(p)))
assert len(retained)==8, len(retained)
posefiles=[FAILPOSE]+retained
pose_names=['actual-scoped-telemetry004-left-wrist-flexion']+[p.parent.name for p in retained]
poses=[as_pose(p) for p in posefiles]
pose_hashes={Path(p).parent.name:sha_file(p) for p in posefiles}
# Exact source-feature owner mapping and exact old witness comparison.
manifest=json.loads(ns["MANIFEST"].read_text())["source"]["anchors"]
bodyname={int(x["core_body_index"]):x["myosim_body"] for x in manifest}
rows=[]
for fi in ids_faces:
 row={"full_skin_face_id":int(fi),"global_vertices":F[int(fi)].astype(int).tolist(),"dominant_skin_owners":[]}
 for vid in F[int(fi)]:
  vid=int(vid);w=Dh["full_weights"][vid];bind=np.asarray(Dh["bindings_u"][:,0],int);owner=[]
  for j in np.argsort(w)[-4:][::-1]:
   if w[j]>0:owner.append({"body":bodyname.get(int(bind[j]),str(int(bind[j]))),"weight":float(w[j])})
  row["dominant_skin_owners"].append({"vertex":vid,"owners":owner})
 rows.append(row)
bone_pose=as_pose(FAILPOSE);bv,bf,bn=h["bone_mesh"](bone_pose,"left")
# Audit IDs include the 11,205-face arm-patch offset on the bone side.
bone_faces=np.asarray([54435,54434,54427],int)-11205
assert np.all((bone_faces>=0)&(bone_faces<len(bf))) and all(bn[int(i)]=="4proxph_l" for i in bone_faces)
base_compare={"schema":"numi.human.local-hand-bone-witness-comparison.v1","pose_path":str(FAILPOSE),"pose_sha256":sha_file(FAILPOSE),"candidate_payloads":{},"skin_faces":rows,"bone_face_ids_in_audit":[54435,54434,54427],"bone_face_ids_local":bone_faces.tolist(),"bone_owner_names":[str(bn[int(i)]) for i in bone_faces],"test":"exact Float32 triangle-pair intersection predicate, isolated to two skin witnesses versus the three witnessed 4proxph_l bone triangles"}
for name,path in (("060",P060/"replacement-bilateral-hands038.nhskin"),("068",E/"finger-profile-068-whole-hand-minimax/replacement-bilateral-hands038.nhskin"),("069h",P069/"replacement-bilateral-hands038.nhskin")):
 d=decode_payload(Path(path).read_bytes());Q=ns["posed"](d,bone_pose).astype(np.float32);sf=F[ids_faces]
 VV=np.vstack([Q[sf.reshape(-1)],bv[bf[bone_faces].reshape(-1)]])
 FF=np.vstack([np.arange(6).reshape(2,3),6+np.arange(9).reshape(3,3)])
 pts,qF=ns["prep"](VV,FF);q=ns["pair_audit"](pts,qF,[0,1],[2,3,4])
 base_compare["candidate_payloads"][name]={"payload_sha256":sha_file(path),"intersections":int(q["intersection_pairs"]),"witnesses":q["first_witnesses"],"skin_face_centroids_mm":(Q[sf].mean(axis=1)*1000).tolist()}
dump(OUT/"targeted-pair-baseline-comparison.json",base_compare)
# Local checks use Float32 emitted positions and the same exact intersection predicates as the full auditor.
# Broad phase is local: only target faces and skin/bone faces with AABBs within 0.05 mm are materialized.
MARGIN_M=0.00005
ALL_FACE_IDS=np.arange(len(F),dtype=int)
def bbox_candidate_pairs(target_tri,other_tri,pad):
 tlo=target_tri.min(axis=1);thi=target_tri.max(axis=1);olo=other_tri.min(axis=1);ohi=other_tri.max(axis=1)
 # Return target/other pair ids whose AABBs are within pad in Euclidean lower-bound distance.
 pairs=[]
 for i in range(len(target_tri)):
  gap=np.maximum(np.maximum(olo-thi[i],tlo[i]-ohi),0.0)
  d2=np.einsum("ij,ij->i",gap,gap)
  js=np.flatnonzero(d2<=pad*pad+1e-24)
  pairs.extend((i,int(j)) for j in js)
 return pairs
def point_tri_d2(p,a,b,c):
 ab=b-a;ac=c-a;ap=p-a;d1=np.dot(ab,ap);d2=np.dot(ac,ap)
 if d1<=0 and d2<=0:return float(np.dot(ap,ap))
 bp=p-b;d3=np.dot(ab,bp);d4=np.dot(ac,bp)
 if d3>=0 and d4<=d3:return float(np.dot(bp,bp))
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:
  v=d1/(d1-d3);q=a+v*ab;d=p-q;return float(np.dot(d,d))
 cp=p-c;d5=np.dot(ab,cp);d6=np.dot(ac,cp)
 if d6>=0 and d5<=d6:return float(np.dot(cp,cp))
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:
  w=d2/(d2-d6);q=a+w*ac;d=p-q;return float(np.dot(d,d))
 va=d3*d6-d5*d4
 if va<=0 and (d4-d3)>=0 and (d5-d6)>=0:
  w=(d4-d3)/((d4-d3)+(d5-d6));q=b+w*(c-b);d=p-q;return float(np.dot(d,d))
 den=1.0/(va+vb+vc);v=vb*den;w=vc*den;q=a+ab*v+ac*w;d=p-q;return float(np.dot(d,d))
def segseg_d2(p1,q1,p2,q2):
 d1=q1-p1;d2=q2-p2;r=p1-p2;a=np.dot(d1,d1);e=np.dot(d2,d2);f=np.dot(d2,r);eps=1e-30
 if a<=eps and e<=eps:return float(np.dot(p1-p2,p1-p2))
 if a<=eps:s=0.;t=np.clip(f/e,0.,1.)
 else:
  c=np.dot(d1,r)
  if e<=eps:t=0.;s=np.clip(-c/a,0.,1.)
  else:
   b=np.dot(d1,d2);den=a*e-b*b;s=np.clip((b*f-c*e)/den,0.,1.) if den>eps else 0.
   t=(b*s+f)/e
   if t<0:t=0.;s=np.clip(-c/a,0.,1.)
   elif t>1:t=1.;s=np.clip((b-c)/a,0.,1.)
 d=(p1+d1*s)-(p2+d2*t);return float(np.dot(d,d))
def tri_dist(a,b):
 d2=[]
 for p in a:d2.append(point_tri_d2(p,*b))
 for p in b:d2.append(point_tri_d2(p,*a))
 for i in range(3):
  for j in range(3):d2.append(segseg_d2(a[i],a[(i+1)%3],b[j],b[(j+1)%3]))
 return float(np.sqrt(max(0,min(d2))))
def face_angles(Q,face_ids,focus_ids):
 ff=F[np.asarray(face_ids,int)];focus=set(map(int,focus_ids));normals={};edges=defaultdict(list)
 for fid,tri in zip(face_ids,ff):
  t=Q[tri].astype(float);n=np.cross(t[1]-t[0],t[2]-t[0]);ln=np.linalg.norm(n);normals[int(fid)]=n/max(ln,1e-30)
  for k in range(3):
   a=int(tri[k]);b=int(tri[(k+1)%3]);key=(min(a,b),max(a,b));edges[key].append(int(fid))
 vals={}
 for edge,fs in edges.items():
  if len(fs)==2 and (fs[0] in focus or fs[1] in focus):
   vals[edge]=float(np.degrees(np.arccos(np.clip(np.dot(normals[fs[0]],normals[fs[1]]),-1,1))))
 return vals
MARGIN_M=0.00002

def pair_test(V,FA,FB=None):
 G=FA if FB is None else np.vstack((FA,FB))
 pts,qf=ns['prep'](V,G)
 if FB is None:return ns['pair_audit'](pts,qf,np.arange(len(FA),dtype=int))
 return ns['pair_audit'](pts,qf,np.arange(len(FA),dtype=int),np.arange(len(FA),len(G),dtype=int))

def local_skin(Q):
 pv=np.asarray(Q[leftslots],dtype=np.float32);pc=patch_faces_local
 T=pv[pc[target_local]];others=np.setdiff1d(np.arange(len(pc)),target_local,assume_unique=False);OT=pv[pc[others]]
 pairs=bbox_candidate_pairs(T,OT,0.0);ob=np.unique([others[j] for i,j in pairs]).astype(int) if pairs else np.array([],int)
 selfr=pair_test(pv,pc[target_local])
 if len(ob):cross=pair_test(pv,pc[target_local],pc[ob])
 else:cross={'candidate_pairs':0,'intersection_pairs':0,'first_witnesses':[],'allowed_shared_seam_pairs':0}
 return {'self_target':selfr,'target_vs_nearby_skin':cross,'nearby_skin_face_count':int(len(ob))}

def local_bone(Q,pose,with_gap=True):
 bv,bf,bowners=h['bone_mesh'](pose,'left');T=np.asarray(Q[F[target_faces]],dtype=np.float32);BT=np.asarray(bv[bf],dtype=np.float32)
 pad=MARGIN_M if with_gap else 0.0;pairs=bbox_candidate_pairs(T,BT,pad);ids=np.unique([j for i,j in pairs]).astype(int) if pairs else np.array([],int)
 if not len(ids):return {'intersection_pairs':0,'first_witnesses':[],'candidate_pairs':0,'nearby_bone_face_count':0,'min_triangle_gap_mm':None,'crossing_bone_names':{}}
 skin_ids=np.unique(F[target_faces]).astype(int);bone_vids=np.unique(bf[ids]).astype(int);smap=np.full(len(Q),-1,dtype=int);smap[skin_ids]=np.arange(len(skin_ids));bmap=np.full(len(bv),-1,dtype=int);bmap[bone_vids]=np.arange(len(bone_vids))
 fa=smap[F[target_faces]];fb=len(skin_ids)+bmap[bf[ids]];VV=np.vstack((Q[skin_ids],bv[bone_vids])).astype(np.float32);cross=pair_test(VV,fa,fb)
 gaps=[tri_dist(T[i].astype(float),BT[j].astype(float)) for i,j in pairs];mingap=float(min(gaps)) if gaps else None
 hitnames={}
 for a,b in cross.get('all_intersection_pairs',[]):hitnames[str(bowners[int(ids[int(b)])])]=hitnames.get(str(bowners[int(ids[int(b)])]),0)+1
 return {'intersection_pairs':int(cross['intersection_pairs']),'first_witnesses':cross['first_witnesses'],'candidate_pairs':int(cross['candidate_pairs']),'nearby_bone_face_count':int(len(ids)),'min_triangle_gap_mm':None if mingap is None else mingap*1000.0,'crossing_bone_names':hitnames}

def shape_local(Q,Q0):
 T=Q[F[target_faces]].astype(float);T0=Q0[F[target_faces]].astype(float)
 cr=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);cr0=np.cross(T0[:,1]-T0[:,0],T0[:,2]-T0[:,0]);a=np.linalg.norm(cr,axis=1)*.5;a0=np.linalg.norm(cr0,axis=1)*.5
 dot=np.sum(cr*cr0,axis=1)/np.maximum(np.linalg.norm(cr,axis=1)*np.linalg.norm(cr0,axis=1),1e-30)
 ac=face_angles(Q,quality_full_faces,target_faces);a0m=face_angles(Q0,quality_full_faces,target_faces)
 inc=[];new90=[]
 for e,v in ac.items():
  b=a0m[e];inc.append(v-b)
  if v>90.0 and b<=90.0+1e-8:new90.append({'edge':list(e),'candidate_deg':v,'060_deg':b})
 ratio=a/np.maximum(a0,1e-30);minratio=float(np.min(ratio));mindot=float(np.min(dot));maxinc=float(max(inc,default=0.0));maxangle=float(max(ac.values(),default=0.0))
 passed=bool(np.all(a>1e-12) and minratio>=0.5 and mindot>=0.8 and not new90 and maxangle<=130.0)
 return {'passed':passed,'target_face_count':int(len(target_faces)),'quality_face_count':int(len(quality_full_faces)),'tested_hinge_count':int(len(ac)),'min_area_ratio_vs_060':minratio,'min_normal_dot_vs_060':mindot,'max_hinge_increase_vs_060_deg':maxinc,'max_hinge_angle_deg':maxangle,'new_over90_hinges':new90[:8]}

def candidate(alpha):
 vv=np.asarray(Dh['vertices_f']).copy();vv[changed,:3]=(np.asarray(Dh['vertices_f'][changed,:3],float)*(1-alpha)+np.asarray(D0['vertices_f'][changed,:3],float)*alpha).astype('<f4');d=dict(Dh);d['vertices_f']=vv;return d

def check_pose(d,pose,name,pfile):
 Q=ns['posed'](d,pose).astype(np.float32);Q0=ns['posed'](D0,pose).astype(np.float32)
 sk=local_skin(Q);bn=local_bone(Q,pose,True);sh=shape_local(Q,Q0);gap=bn['min_triangle_gap_mm']
 ok=(sk['self_target']['intersection_pairs']==0 and sk['target_vs_nearby_skin']['intersection_pairs']==0 and bn['intersection_pairs']==0 and (gap is None or gap>=0.02) and sh['passed'])
 return {'pose':name,'pose_path':str(pfile),'pose_sha256':sha_file(pfile),'passed':bool(ok),'local_skin':sk,'local_bone':bn,'local_shape':sh}

# Map full source faces/vertices to the exact stored 069h left-patch rows.
leftslots=np.asarray(leftpatch0['source_slots'],int);patch_faces_local=np.asarray(leftpatch0['faces'],int)
local_changed=np.asarray([int(np.flatnonzero(leftslots==v)[0]) for v in changed],int)
target_local=np.flatnonzero(np.any(np.isin(patch_faces_local,local_changed),axis=1))
target_faces=np.asarray(leftpatch0['old_faces'],int)[target_local]
edge_faces=defaultdict(list)
for fi,tri in enumerate(patch_faces_local):
 for k in range(3):edge_faces[tuple(sorted((int(tri[k]),int(tri[(k+1)%3]))))].append(int(fi))
target_edges=set()
for fi in target_local:
 tri=patch_faces_local[int(fi)]
 for k in range(3):target_edges.add(tuple(sorted((int(tri[k]),int(tri[(k+1)%3])))))
quality_local=set(map(int,target_local))
for edge,fs in edge_faces.items():
 if edge in target_edges:quality_local.update(map(int,fs))
quality_full_faces=np.asarray(sorted(np.asarray(leftpatch0['old_faces'],int)[list(quality_local)]),int)

def trial(alpha):
 d=candidate(alpha);Q=ns['posed'](d,poses[0]).astype(np.float32);Q0=ns['posed'](D0,poses[0]).astype(np.float32)
 sk=local_skin(Q);bn=local_bone(Q,poses[0],True);sh=shape_local(Q,Q0);gap=bn['min_triangle_gap_mm']
 passed=(sk['self_target']['intersection_pairs']==0 and sk['target_vs_nearby_skin']['intersection_pairs']==0 and bn['intersection_pairs']==0 and (gap is None or gap>=0.02) and sh['passed'])
 return {'alpha':float(alpha),'passed':bool(passed),'bone_intersections':bn['intersection_pairs'],'bone_min_gap_mm':gap,'bone_candidate_pairs':bn['candidate_pairs'],'skin_self_intersections':sk['self_target']['intersection_pairs']+sk['target_vs_nearby_skin']['intersection_pairs'],'shape_pass':sh['passed'],'shape':sh}

trials=[];bracket=None
for alpha in [i/10. for i in range(11)]:
 r=trial(alpha);trials.append(r)
 if r['passed']:
  prev=trials[-2]['alpha'] if len(trials)>1 else 0.;bracket=(prev,alpha);break
if bracket is None:
 dump(OUT/'failed-alpha-search-069i.json',{'trials':trials,'changed_ids':changed.tolist(),'target_faces':target_faces.tolist()})
 print(json.dumps({'status':'no_passing_alpha','trials':trials},indent=2))
 raise RuntimeError('four-vertex local retreat did not pass through full 060 target positions; candidate not emitted')
lo,hi=bracket
for _ in range(7):
 mid=(lo+hi)*.5;r=trial(mid);trials.append(r)
 if r['passed']:hi=mid
 else:lo=mid
alpha=hi;D=candidate(alpha)
results=[check_pose(D,p,n,fp) for p,n,fp in zip(poses,pose_names,posefiles)]
if not all(x['passed'] for x in results):raise RuntimeError('local gates failed on final nine-pose check; no payload emitted: '+json.dumps([(x['pose'],x['passed'],x['local_bone']['intersection_pairs'],x['local_bone']['min_triangle_gap_mm'],x['local_shape']['passed']) for x in results]))
# Prepare exact NHSKIN with only four changed rest positions and affected display-normal rows.
raw=bytearray(rawh);arr=np.ndarray((nvert,14),dtype='<f4',buffer=raw,offset=60+36*Dh['binding_count']);arr[changed,:3]=D['vertices_f'][changed,:3]
neutral_file=next(p for p in posefiles if p.parent.name=='neutral');neutral_pose=as_pose(neutral_file);Qn=ns['posed'](D,neutral_pose).astype(float)
affected=np.unique(F[np.any(np.isin(F,changed),axis=1)]).astype(int);fn=np.cross(Qn[F[:,1]]-Qn[F[:,0]],Qn[F[:,2]]-Qn[F[:,0]]);wn=np.zeros_like(Qn);np.add.at(wn,F.ravel(),np.repeat(fn,3,axis=0));wn/=np.maximum(np.linalg.norm(wn,axis=1)[:,None],1e-30)
W=np.asarray(D['full_weights'],float);BF=np.asarray(D['bindings_f'],float);bindings=np.asarray(D['bindings_u'][:,0],int);state={int(x['body_index']):x for x in neutral_pose['bodies']};N=np.zeros((len(affected),3,3),float)
for j,b in enumerate(bindings):
 cur=state[int(b)]['current'];rest=state[int(b)]['rest'];rc=ns['mat'](cur['orientation_world_xyzw']);rr=ns['mat'](rest['orientation_world_xyzw']);N+=W[affected,j,None,None]*(rc@rr.T)[None]
ln=np.linalg.solve(N,wn[affected,:,None])[:,:,0];ln/=np.maximum(np.linalg.norm(ln,axis=1)[:,None],1e-30);arr[affected,3:6]=ln.astype('<f4');payload=bytes(raw);Dout=decode_payload(payload)
for k in ('indices','bindings_f','bindings_u','full_weights'):assert np.array_equal(Dout[k],Dh[k]),k
unchanged=np.setdiff1d(np.arange(nvert),changed);assert np.array_equal(Dout['vertices_f'][unchanged,:3],Dh['vertices_f'][unchanged,:3])
assert np.array_equal(Dout['vertices_f'][changed,:3],D['vertices_f'][changed,:3])
# Emit only after gates. World arrays and left patch inherit exact full topology/source metadata.
(OUT/'replacement-bilateral-hands038.nhskin').write_bytes(payload);Qemit=ns['posed'](Dout,neutral_pose).astype(float)
g0=np.load(P069/'replacement-geometry.npz',allow_pickle=False);g={k:g0[k].copy() for k in g0.files};g['world']=Qemit;np.savez(OUT/'replacement-geometry.npz',**g)
lpf={k:leftpatch0[k].copy() for k in leftpatch0.files};lpf['world']=Qemit[leftslots];np.savez_compressed(OUT/'left-patch.npz',**lpf);shutil.copy2(P069/'right-patch.npz',OUT/'right-patch.npz')
report={'schema':'numi.human.finger-local-patch-audit.v1','candidate':'069i-left-ring-web-retreat','status':'diagnostic_candidate_emitted_pending_parent_full_audit','admission':'not physically qualified; parent full-surface and layer audits remain required','inputs':{'069h_payload_sha256':sha_bytes(rawh),'060_payload_sha256':sha_bytes(raw0)},'candidate_payload_sha256':sha_bytes(payload),'index_sha256':sha_bytes(np.asarray(Dout['indices'],dtype='<u4').tobytes()),'witness_baseline_comparison':base_compare,'alpha_search':{'criterion':'zero exact target-skin/bone and target-skin/nearby-skin intersections; at least 0.02 mm minimum target-face to nearby left-bone triangle gap; local area/orientation/hinge gates','trials':trials,'chosen_alpha':float(alpha)},'pose_local_results':results,'integrity':{'indices_bindings_full_weights_bit_exact_to_069h':True,'all_nonchanged_positions_bit_exact_to_069h':True,'changed_global_vertex_ids':changed.tolist(),'normal_rows_recomputed':int(len(affected)),'right_patch_file_sha256_069h':sha_file(P069/'right-patch.npz'),'right_patch_file_sha256_candidate':sha_file(OUT/'right-patch.npz'),'right_patch_files_bit_exact':sha_file(P069/'right-patch.npz')==sha_file(OUT/'right-patch.npz'),'wrist_scope':'No wrist or forearm positions/weights were edited.'},'limits':['local target-face collision checks only; full-surface audit remains parent-owned','not physical or functional qualification','retained poses are discrete snapshots, not continuous swept contact','source fit is not certain']}
dump(OUT/'local-audit-069i.json',report)
provenance={'schema':'numi.human.finger-profile-local-repair-provenance.v1','candidate':'069i-left-ring-web-retreat','status':report['status'],'parent_payload_sha256':sha_bytes(rawh),'baseline_060_payload_sha256':sha_bytes(raw0),'candidate_payload_sha256':sha_bytes(payload),'global_index_sha256':report['index_sha256'],'changed_global_vertex_ids':changed.tolist(),'alpha_toward_060':float(alpha),'rest_vertex_deltas_mm':{str(int(v)):((Dout['vertices_f'][int(v),:3].astype(float)-Dh['vertices_f'][int(v),:3].astype(float))*1000).tolist() for v in changed},'actual_failing_pose_path':str(FAILPOSE),'actual_failing_pose_sha256':sha_file(FAILPOSE),'tested_pose_hashes':{n:sha_file(fp) for n,fp in zip(pose_names,posefiles)},'source_faces_full_ids':target_faces.astype(int).tolist(),'witness_faces_full_ids':ids_faces.astype(int).tolist(),'bone_owner':'4proxph_l','method':'Smallest accepted interpolation on the four shared witness controls toward exact 060 positions. Display normals recomputed for their adjacent vertices. No weights, IDs, topology, or wrist positions changed.','limits':report['limits']}
dump(OUT/'generation-provenance.json',provenance)
print(json.dumps({'status':report['status'],'chosen_alpha':alpha,'payload_sha256':report['candidate_payload_sha256'],'tested_pose_count':len(results),'all_local_pass':all(x['passed'] for x in results),'actual_min_bone_gap_mm':results[0]['local_bone']['min_triangle_gap_mm'],'changed_ids':changed.tolist(),'target_face_count':len(target_faces)},indent=2))
