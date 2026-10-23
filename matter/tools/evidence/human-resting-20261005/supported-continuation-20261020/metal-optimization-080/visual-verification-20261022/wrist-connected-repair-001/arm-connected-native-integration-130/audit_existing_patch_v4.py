from pathlib import Path
import sys,json,hashlib,time,argparse,numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001';P=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001')
sys.path.insert(0,str(R/'tools'));import rebind_human_skin_reference as rb
from human_skin_witness_identity import bind_skin_witness_identities
sys.path.insert(0,str(E/'joint-region-roi-014'));import run_ankle_tibia_weight_counterfactual_018 as wa
audit=wa.audit
parser=argparse.ArgumentParser(description='Exact emitted arm geometry audit; pin the payload instead of copying pose scripts.')
parser.add_argument('emission_directory',type=Path)
parser.add_argument('--payload',type=Path)
parser.add_argument('--payload-sha256')
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--roi-payload',type=Path)
parser.add_argument('--roi-payload-sha256')
parser.add_argument('--all-poses',action='store_true',help='Diagnostic only: continue after a failed pose')
args=parser.parse_args()
D=args.emission_directory;out=args.output
if out.exists():raise SystemExit('refusing to overwrite audit evidence: '+str(out))
base_payload=D/'candidate.nhskin';base=rb.decode(base_payload.read_bytes())
a=np.load(D/'candidate.npz',allow_pickle=False)
receipt=json.loads((D/'emission-receipt.json').read_text())
assert receipt['emitted_payload_sha256']==wa.sha(base_payload)
assert receipt['candidate_sha256']==wa.sha(D/'candidate.npz')
if (args.payload is None)!=(args.payload_sha256 is None):raise SystemExit('override requires both payload and exact SHA-256')
payload=args.payload or base_payload
if args.payload_sha256 and wa.sha(payload)!=args.payload_sha256:raise SystemExit('payload SHA-256 mismatch')
p=rb.decode(payload.read_bytes());F=p['inds'].astype(np.int64)
if p['nv']!=base['nv'] or p['nb']!=base['nb'] or p['fingerprint']!=base['fingerprint'] or p['source_sha256']!=base['source_sha256'] or p['binds'].tobytes()!=base['binds'].tobytes() or p['inds'].tobytes()!=base['inds'].tobytes():raise SystemExit('override is not the same registered topology/source')
def same_layout(q):
 return q['nv']==base['nv'] and q['nb']==base['nb'] and q['fingerprint']==base['fingerprint'] and q['source_sha256']==base['source_sha256'] and q['binds'].tobytes()==base['binds'].tobytes() and q['inds'].tobytes()==base['inds'].tobytes()
def changed_rows(q):
 return np.flatnonzero(np.any(q['weights']!=base['weights'],axis=1)|np.any(q['verts'][:,:3]!=base['verts'][:,:3],axis=1))
if (args.roi_payload is None)!=(args.roi_payload_sha256 is None):raise SystemExit('ROI override requires payload and exact SHA-256')
roi_payload=args.roi_payload or payload
if args.roi_payload_sha256 and wa.sha(roi_payload)!=args.roi_payload_sha256:raise SystemExit('ROI payload SHA-256 mismatch')
rp=rb.decode(roi_payload.read_bytes())
if not same_layout(rp):raise SystemExit('ROI payload is not the same registered topology/source')
changed=changed_rows(p);roi_changed=changed_rows(rp);affected=np.union1d(changed,roi_changed)
incident=np.any(np.isin(F,affected),axis=1)
patch=np.flatnonzero((a['face_origin_codes']!=0)|incident);other=np.setdiff1d(np.arange(len(F)),patch)

out.mkdir(parents=True)
report={'payload_path':str(payload.resolve()),'payload_sha256':wa.sha(payload),'base_payload_sha256':wa.sha(base_payload),'candidate_npz_sha256':wa.sha(D/'candidate.npz'),'scope':'Experimental emitted Float32 arm patch, matched static poses only. No physical admission.','physical_admission':False,'patch_faces':len(patch),'retained_faces':len(other),'roi_payload_sha256':wa.sha(roi_payload),'roi_payload_path':str(roi_payload.resolve()),'roi_changed_rows_count':len(roi_changed),'actual_changed_position_or_weight_rows_count':len(changed),'affected_rows_count':len(affected),'affected_retained_face_count':int(np.count_nonzero(incident & (a['face_origin_codes']==0))),'patch_rows_sha256':hashlib.sha256(patch.astype('<u4').tobytes()).hexdigest(),'changed_weight_rows_count':int(np.count_nonzero(np.any(p['weights']!=base['weights'],axis=1))),'changed_position_or_weight_rows_sha256':hashlib.sha256(changed.astype('<u4').tobytes()).hexdigest(),'audit_script_sha256':wa.sha(Path(__file__)),'poses':{}}
(out/'audit-inputs.json').write_text(json.dumps(report,indent=2)+'\n')
print('payload_sha256',report['payload_sha256'],'output',out,flush=True)
bm,bpath,brec,bv,bi,anchors=audit.load_bones();report['bone_payload_sha256']=wa.sha(bpath);report['bone_manifest_sha256']=wa.sha(audit.MAN)
audit.BONE_KEYS['full_arm']=('clavicle','scapula','humerus','ulna','radius')
neutral=json.loads((P/'neutral.json').read_text())
snap={'bodies':[{'body_index':x['core_body'],'current':{'position_world_m':(np.asarray(x['position'])+[0,0,.35]).tolist(),'orientation_world_xyzw':x['quaternion_xyzw']}} for x in neutral['source_poses']]}
cases=[('reference',neutral,[0,0,.35],snap,P/'neutral.json')]
for label in ['shoulder-elevation-050-014','shoulder-abduction-at-elevation-050-015','elbow-flexion-080-014']:
 path=E/'native-registered-skin-pose-clean-014'/label/'registered-skin-poses.json';s=json.loads(path.read_text());pose={'source_poses':[{'core_body':x['body_index'],'position':x['current']['position_world_m'],'quaternion_xyzw':x['current']['orientation_world_xyzw']} for x in s['bodies']]}
 cases.append((label,pose,[0,0,0],s,path))
for label,pose,lift,s,path in cases:
 t=time.monotonic();mat,off=rb.reference_transforms(p,pose,lift);V=rb.positions(p,mat,off).astype(np.float32)
 pts,qf=audit.exact_scene(V,F);checks={}
 for name,pa,pb in [('patch_self',patch,None),('patch_vs_retained',patch,other)]:
  checks[name]=bind_skin_witness_identities(audit.exact_pair_audit(pts,qf,pa,pb,same_surface=pb is None),F,face_origins=a['face_origin_codes']);print(label,name,checks[name]['intersection_pairs'],flush=True)
 u=np.unique(F[patch].ravel());remap=np.full(len(V),-1,dtype=np.int64);remap[u]=np.arange(len(u));sf=remap[F[patch]]
 for side in ['left','right']:
  Vb,Fb,owners=audit.bone_scene(s,'full_arm',side,brec,bv,bi,anchors)
  cv=np.vstack((V[u],Vb)).astype(np.float32);cf=np.vstack((sf,Fb+len(u)));cp,cq=audit.exact_scene(cv,cf);n=len(sf)
  checks['patch_vs_'+side+'_bones']=bind_skin_witness_identities(audit.exact_pair_audit(cp,cq,np.arange(n),np.arange(n,len(cf)),label_a=lambda f:'arm_patch',label_b=lambda f:owners[int(f)-n]),F,patch_rows=patch,bone_owners=owners,face_origins=a['face_origin_codes'])
  print(label,side+'_bones',checks['patch_vs_'+side+'_bones']['intersection_pairs'],flush=True)
 report['poses'][label]={'pose_sha256':wa.sha(path),'checks':checks,'wall_s':time.monotonic()-t}
 (out/'emitted-exact-static.partial.json').write_text(json.dumps(report,indent=2)+'\n')
 if not args.all_poses and any(v['intersection_pairs'] for v in checks.values()):
  report['stopped_at_first_failed_pose']=label;break
report['all_requested_poses_pass']=len(report['poses'])==len(cases) and all(v['intersection_pairs']==0 for s in report['poses'].values() for v in s['checks'].values())
(out/'emitted-exact-static.json').write_text(json.dumps(report,indent=2)+'\n')

