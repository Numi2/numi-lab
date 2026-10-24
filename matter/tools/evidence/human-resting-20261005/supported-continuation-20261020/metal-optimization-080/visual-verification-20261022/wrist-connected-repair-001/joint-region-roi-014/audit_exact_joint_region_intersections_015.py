#!/usr/bin/env python3
"""Exact selected skin self/cross-surface and nearby-bone tests for static joint snapshots.

The inputs are native registration snapshots and their own registered-skin CSV
coordinates. This is a geometric diagnostic, not a physics/FEM/contact pass.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, struct, sys, time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=ROOT/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
CAP=E/'native-registered-skin-pose-clean-014'
OUTDIR=E/'joint-region-roi-014'
PAYLOAD=E/'finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin'
MAN=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
PAIRS=[
 ('shoulder-elevation','shoulder','construction-neutral-014','shoulder-elevation-050-014'),
 ('shoulder-plane-at-elevation','shoulder','shoulder-elevation-050-014','shoulder-abduction-at-elevation-050-015'),
 ('elbow-flexion','elbow','construction-neutral-014','elbow-flexion-080-014'),
 ('knee-flexion','knee','construction-neutral-014','knee-flexion-025-014'),
 ('ankle-dorsiflexion','ankle','construction-neutral-014','ankle-dorsiflexion-020-014'),
 ('ankle-plantarflexion','ankle','construction-neutral-014','ankle-plantarflexion-020-014'),
 ('mtp-flexion','mtp','construction-neutral-014','mtp-flexion-020-014'),
 ('mtp-extension','mtp','construction-neutral-014','mtp-extension-020-014'),
]
GROUPS={
 'shoulder':{'shoulder_axilla','upper_arm','elbow_forearm','wrist_carpals','palm_metacarpals','digits'},
 'elbow':{'upper_arm','elbow_forearm','wrist_carpals','palm_metacarpals','digits'},
 'knee':{'thigh','knee','lower_leg'},
 'ankle':{'lower_leg','foot'},
 'mtp':('toes','metatars'),
}
BONE_KEYS={
 'shoulder':('clavicle','scapula','humerus'),
 'elbow':('humerus','ulna','radius'),
 'knee':('femur','patella','tibia','fibula'),
 'ankle':('tibia','fibula','talus','calcn'),
 # The pinned bone manifest carries an aggregate toes mesh, not separate toe phalanges.
 'mtp':('toes','metatars'),
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mat(q):
 x,y,z,w=map(float,q);s=x*x+y*y+z*z+w*w
 if not math.isfinite(s) or s<=1e-24:raise ValueError('invalid quaternion')
 x,y,z,w=x/math.sqrt(s),y/math.sqrt(s),z/math.sqrt(s),w/math.sqrt(s)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)
def skin_region(n):
 n=n.lower()
 if 'clavicle' in n or 'scapula' in n:return 'shoulder_axilla'
 if 'humerus' in n:return 'upper_arm'
 if 'ulna' in n or 'radius' in n:return 'elbow_forearm'
 if any(k in n for k in ('scaphoid','lunate','triquetrum','pisiform','capitate','trapezium','trapezoid','hamate','carpal')):return 'wrist_carpals'
 if any(k in n for k in ('metacarp','firstmc','secondmc','thirdmc','fourthmc','fifthmc')):return 'palm_metacarpals'
 if any(k in n for k in ('thumb','proxph','midph','distph','phal')):return 'digits'
 if 'femur' in n:return 'thigh'
 if 'patella' in n:return 'knee'
 if 'tibia' in n or 'fibula' in n:return 'lower_leg'
 if any(k in n for k in ('talus','calcn','calcaneus','tarsal','metatars','toes','toe_')):return 'foot'
 return 'other'
def side(n):
 if n.endswith('_l'):return 'left'
 if n.endswith('_r'):return 'right'
 return 'none'
def majority(x):
 return Counter(x).most_common(1)[0][0]
def exact_points(V):
 rat=[tuple(Fraction.from_float(float(x)) for x in row) for row in V]
 den=max(x.denominator for row in rat for x in row)
 return [tuple(x.numerator*(den//x.denominator) for x in row) for row in rat]
def read_csv(path):
 with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=61950:raise ValueError(f'{path} row count {len(rows)}')
 ids=np.fromiter((int(r['vertex']) for r in rows),dtype=np.int64,count=len(rows))
 if not np.array_equal(ids,np.arange(61950)):raise ValueError('native CSV vertex ordering mismatch')
 return np.asarray([[float(r[k]) for k in ('wx','wy','wz')] for r in rows],dtype=np.float32)
def exact_scene(V,F):
 qv,qf,_=coordinate_quotient(np.asarray(V,dtype=np.float32).tolist(),np.asarray(F,dtype=np.int64).tolist())
 qf=np.asarray(qf,dtype=np.int64)
 return exact_points(np.asarray(qv,dtype=np.float64)),qf
def exact_pair_audit(pts,qf,ids_a,ids_b=None,label_a=None,label_b=None,same_surface=False,keep=12):
 aa,da=records_nonzero(pts,qf,ids_a)
 if ids_b is None:
  bb=aa;db=0;pairs=_aabb_candidate_pairs(aa,aa,same_surface=True)
 else:
  bb,db=records_nonzero(pts,qf,ids_b)
  pairs=_aabb_candidate_pairs(aa,bb,same_surface=False)
 counts=Counter();seams=Counter();witnesses=defaultdict(list);candidate=0
 for a,b in pairs:
  candidate+=1;hits=triangle_intersection_points(a[0],b[0])
  if not hits:continue
  shared=set(a[4])&set(b[4]);common={a[0][a[4].index(k)] for k in shared}
  ka=label_a(a[3]) if label_a else 'selected'
  kb=label_b(b[3]) if label_b else ka
  key=f'{ka}|{kb}'
  if len(shared) in (1,2) and all(_allowed_shared_point(x,common) for x in hits):
   seams[key]+=1;continue
  counts[key]+=1
  if len(witnesses[key])<keep:witnesses[key].append({'face_a':int(a[3]),'face_b':int(b[3]),'intersection_points':len(hits)})
 return {'candidate_pairs':candidate,'intersection_pairs':int(sum(counts.values())),'allowed_shared_seam_pairs':int(sum(seams.values())),'degenerate_faces_excluded':[int(da),int(db)],'intersections_by_owner_pair':dict(counts),'allowed_seams_by_owner_pair':dict(seams),'witnesses_by_owner_pair':dict(witnesses)}
def load_bones():
 m=json.loads(MAN.read_text());raw_path=MAN.parent/m['payload']['file'];raw=raw_path.read_bytes()
 hsz=struct.calcsize('<8s5I32s');_,abi,bc,vc,ic,*_=struct.unpack_from('<8s5I32s',raw,0)
 rsz=struct.calcsize('<6I8fI');records=[struct.unpack_from('<6I8fI',raw,hsz+i*rsz) for i in range(bc)]
 off=hsz+bc*rsz;bv=np.asarray([struct.unpack_from('<6f',raw,off+i*24)[:3] for i in range(vc)],dtype=np.float32);off+=vc*24
 bi=np.frombuffer(raw,dtype='<u4',count=ic,offset=off).reshape(-1,3)
 anchors=m['source']['anchors']
 if len(records)!=len(anchors):raise ValueError('bone record/anchor count mismatch')
 return m,raw_path,records,bv,bi,anchors
def bone_scene(pose,joint,which_side,records,bv,bi,anchors):
 st={int(x['body_index']):x for x in pose['bodies']};suffix='_l' if which_side=='left' else '_r';keys=BONE_KEYS[joint]
 vv=[];ff=[];owners=[];vb=0
 for r,a in zip(records,anchors):
  name=str(a['myosim_body']).lower();bi_body,fv,nv,fi,ni,stable,tx,ty,tz,qx,qy,qz,qw,scale,_=r
  if not name.endswith(suffix) or not any(k in name for k in keys):continue
  local=(bv[fv:fv+nv].astype(np.float64)*scale)@mat((qx,qy,qz,qw)).T+np.array((tx,ty,tz),dtype=np.float64)
  cur=st[int(bi_body)]['current'];world=(local@mat(cur['orientation_world_xyzw']).T+np.asarray(cur['position_world_m'],dtype=np.float64)).astype(np.float32)
  faces=bi[fi//3:(fi+ni)//3].astype(np.int64)-int(fv)+vb
  vv.append(world);ff.append(faces);owners.extend([name]*(len(faces)));vb+=len(world)
 if not vv:return np.empty((0,3),np.float32),np.empty((0,3),np.int64),[]
 return np.vstack(vv),np.vstack(ff),owners
def main():
 global coordinate_quotient,_aabb_candidate_pairs,triangle_intersection_points,_allowed_shared_point,records_nonzero
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--pair',action='append',choices=[x[0] for x in PAIRS],help='run one matched pair; repeat to select several')
 parser.add_argument('--output',type=Path,help='new report path; a sibling partial file is written after each pair')
 args=parser.parse_args()
 selected=[x for x in PAIRS if not args.pair or x[0] in args.pair]
 output_path=args.output or OUTDIR/'exact-joint-region-intersection-audit-015.json'
 partial_path=output_path.with_name(output_path.stem+'.partial'+output_path.suffix)
 if output_path.exists() or partial_path.exists():raise SystemExit(f'refusing overwrite existing report: {output_path} or {partial_path}')
 sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
 sys.path.insert(0,str(E))
 from numilab_human.skin_source_payload_preflight import decode_payload
 from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
 from numilab_human.cardiac_cavity_intersections import _aabb_candidate_pairs,triangle_intersection_points,_allowed_shared_point
 from audit_bilateral_skin_retained_poses_002 import records_nonzero
 d=decode_payload(PAYLOAD.read_bytes());F=np.asarray(d['indices'],dtype=np.int64).reshape(-1,3);W=np.asarray(d['full_weights'],dtype=np.float64);bind=np.asarray(d['bindings_u'][:,0],dtype=np.int64)
 if W.shape!=(61950,86) or len(bind)!=86:raise SystemExit(f'unexpected skin payload dimensions {W.shape} {bind.shape}')
 manifest=json.loads(MAN.read_text());name_by_id={int(a['core_body_index']):str(a['myosim_body']) for a in manifest['source']['anchors']}
 if len(name_by_id)!=86:raise SystemExit(f'expected 86 unique donor names, got {len(name_by_id)}')
 vertex_owner=bind[np.argmax(W,axis=1)];owner_name=[name_by_id.get(int(x),'') for x in vertex_owner]
 if any(not x for x in owner_name):raise SystemExit('unmapped skin source donor')
 vreg=np.asarray([skin_region(x) for x in owner_name]);vside=np.asarray([side(x) for x in owner_name])
 fside=np.asarray([majority(vside[t]) for t in F]);freg=np.asarray([majority(vreg[t]) for t in F])
 manifest_bone, bonepath, brec,bv,bi,anchors=load_bones()
 # Validate native input identity from a complete capture before any geometry work.
 first=json.loads((CAP/'construction-neutral-014'/'invocation.json').read_text())
 if first['skin_payload_sha256']!=sha(PAYLOAD):raise SystemExit('skin payload hash disagrees with native invocation')
 posecache={}
 def get_pose(folder):
  if folder not in posecache:
   p=CAP/folder
   r=json.loads((p/'result.json').read_text());snap=json.loads((p/'registered-skin-poses.json').read_text())
   if r.get('registered_rows')!=61950 or not r.get('paired_pose_snapshot_valid') or len(snap.get('bodies',[]))!=86:raise SystemExit(f'incomplete capture: {folder}')
   posecache[folder]=(read_csv(p/'registered-skin.csv'),snap,r,json.loads((p/'invocation.json').read_text()))
  return posecache[folder]
 scenes={}
 def scene_for(folder):
  if folder not in scenes:
   V,snap,_,_=get_pose(folder);t0=time.monotonic();pts,qf=exact_scene(V,F);scenes[folder]=(pts,qf,time.monotonic()-t0)
   print(json.dumps({'event':'prepared_exact_skin_scene','pose':folder,'vertices':len(pts),'faces':len(qf),'seconds':scenes[folder][2]}),flush=True)
  return scenes[folder]
 report={'schema':'numi.human.static-joint-region-exact-intersection-audit.v1','boundary':'Exact predicates on static native registration CSV positions and static registered bone meshes. No physics-step, muscle actuation, FEM admission/transfer, contact response, accepted/replay, or clinical ROM qualification. Native captures exited 1 after complete registration dumps at the existing FEM/layer gate.','skin_payload':str(PAYLOAD),'skin_payload_sha256':sha(PAYLOAD),'source_bone_manifest':str(MAN),'source_bone_manifest_sha256':sha(MAN),'bone_mesh_payload':str(bonepath),'bone_mesh_sha256':sha(bonepath),'native_binary_sha256':first['binary_sha256'],'candidate_rigid_sha256':first['candidate_rigid_sha256'],'exact_predicate_modules':{},'skin_owner_counts':{c:int(np.sum(vreg==c)) for c in sorted(set(vreg))},'face_owner_counts':{c:int(np.sum(freg==c)) for c in sorted(set(freg))},'pose_pairs':{},'selected_pairs':[x[0] for x in selected],'method':{'skin_faces':'face class and side are majority vote over each triangle vertices, each vertex class from argmax of exact registered 86-body source weights and pinned source bone manifest names','self':'exact Float32 coordinate quotient + exact triangle predicates inside each selected side/joint ROI','other_skin':'selected joint ROI against the complement full-body skin, exact shared edge/vertex seams exempted','nearby_bone':'selected skin ROI against pinned source bone meshes transformed by the same native body snapshot; only listed bone members are included','roi_groups':{k:sorted(v) for k,v in GROUPS.items()},'nearby_bone_name_tokens':{k:list(v) for k,v in BONE_KEYS.items()}}}
 for mod in ('/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/skin_source_payload_preflight.py','/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/compiled_quotient_embeddedness.py','/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/cardiac_cavity_intersections.py',str(E/'audit_bilateral_skin_retained_poses_002.py')):
  report['exact_predicate_modules'][mod]=sha(Path(mod))
 for label,joint,basefolder,activefolder in selected:
  print(json.dumps({'event':'start_pair','pair':label,'joint':joint,'baseline':basefolder,'active':activefolder}),flush=True)
  pair={'joint':joint,'baseline':{},'active':{}}
  for kind,folder in (('baseline',basefolder),('active',activefolder)):
   V,snap,res,inv=get_pose(folder);pts,qf,scenetime=scene_for(folder)
   side_reports={};groups=GROUPS[joint]
   for sd in ('left','right'):
    roi=np.flatnonzero((fside==sd)&np.isin(freg,list(groups)))
    other=np.setdiff1d(np.arange(len(F),dtype=np.int64),roi,assume_unique=True)
    if len(roi)==0:raise ValueError(f'empty ROI {joint}/{sd}')
    t0=time.monotonic()
    selfres=exact_pair_audit(pts,qf,roi,None,same_surface=True)
    crossres=exact_pair_audit(pts,qf,roi,other,label_a=lambda fid:f'{sd}/{freg[fid]}',label_b=lambda fid:f'{fside[fid]}/{freg[fid]}')
    bV,bF,bown=bone_scene(snap,joint,sd,brec,bv,bi,anchors)
    if len(bF):
     u=np.unique(F[roi].ravel());remap=np.full(len(V),-1,dtype=np.int64);remap[u]=np.arange(len(u));skinF=remap[F[roi]]
     combinedV=np.vstack((V[u],bV)).astype(np.float32);combinedF=np.vstack((skinF,bF+len(u))).astype(np.int64)
     cpts,cqf=exact_scene(combinedV,combinedF);nbskin=len(skinF)
     boneres=exact_pair_audit(cpts,cqf,np.arange(nbskin,dtype=np.int64),np.arange(nbskin,len(combinedF),dtype=np.int64),label_a=lambda fid:f'{sd}/{freg[int(roi[fid])]}',label_b=lambda fid:bown[fid-nbskin])
    else:
     boneres={'candidate_pairs':0,'intersection_pairs':0,'allowed_shared_seam_pairs':0,'degenerate_faces_excluded':[0,0],'intersections_by_owner_pair':{},'allowed_seams_by_owner_pair':{},'witnesses_by_owner_pair':{}}
    side_reports[sd]={'roi_face_count':int(len(roi)),'roi_unique_vertex_count':int(len(np.unique(F[roi]))),'bone_face_count':int(len(bF)),'bone_mesh_parts':dict(Counter(bown)),'skin_self':selfres,'skin_roi_vs_other_fullbody_skin':crossres,'skin_roi_vs_nearby_bone':boneres,'wall_seconds':time.monotonic()-t0}
    print(json.dumps({'event':'finished_side','pair':label,'kind':kind,'side':sd,'roi_faces':len(roi),'bones':len(bF),'self_hits':selfres['intersection_pairs'],'other_skin_hits':crossres['intersection_pairs'],'bone_hits':boneres['intersection_pairs'],'seconds':side_reports[sd]['wall_seconds']}),flush=True)
   pair[kind]={'pose_folder':folder,'requested_q':inv.get('requested_q'),'native_exit_code':res.get('native_exit_code'),'registered_csv_sha256':res.get('registered_csv_sha256'),'pose_snapshot_sha256':res.get('paired_pose_snapshot_sha256'),'construction_pose_lines':res.get('construction_pose_lines'),'skin_scene_prepare_seconds':scenetime,'sides':side_reports}
  pair['matched_inputs']={'same_binary':get_pose(basefolder)[3]['binary_sha256']==get_pose(activefolder)[3]['binary_sha256'],'same_rigid':get_pose(basefolder)[3]['candidate_rigid_sha256']==get_pose(activefolder)[3]['candidate_rigid_sha256'],'same_skin':get_pose(basefolder)[3]['skin_payload_sha256']==get_pose(activefolder)[3]['skin_payload_sha256'],'same_environment_except_output_paths':{k:v for k,v in get_pose(basefolder)[3]['environment_overrides'].items() if k not in ('NUMI_HUMAN_SKIN_REFERENCE_DUMP','NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP')}=={k:v for k,v in get_pose(activefolder)[3]['environment_overrides'].items() if k not in ('NUMI_HUMAN_SKIN_REFERENCE_DUMP','NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP')},'same_reference_root_transform':get_pose(basefolder)[3]['construction_root_lift_m']==get_pose(activefolder)[3]['construction_root_lift_m'],'no_ongoing_lock':not get_pose(basefolder)[3]['ongoing_q_lock'] and not get_pose(activefolder)[3]['ongoing_q_lock']}
  report['pose_pairs'][label]=pair
  partial_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'event':'pair_complete','pair':label,'out':str(partial_path)}),flush=True)
 output_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print('REPORT='+str(output_path),flush=True)
if __name__=='__main__':
 main()

