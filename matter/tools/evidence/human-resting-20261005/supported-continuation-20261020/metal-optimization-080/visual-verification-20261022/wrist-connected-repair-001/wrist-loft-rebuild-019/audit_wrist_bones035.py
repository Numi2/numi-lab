import sys,json,struct,hashlib,time,math
from pathlib import Path
from collections import Counter
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BASE=E/'candidate-bilateral-arm-hand-weights-016.nhskin'
CAND=E/'wrist-loft-rebuild-019/palm-fairing035.nhskin'
MANIFEST=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
sys.path.insert(0,str(E))
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
from numilab_human.cardiac_cavity_intersections import _aabb_candidate_pairs,triangle_intersection_points,_allowed_shared_point
from audit_bilateral_skin_retained_poses_002 import posed,region,prep,records_nonzero

def mat(q):
 x,y,z,w=map(float,q);s=1/math.sqrt(x*x+y*y+z*z+w*w);x*=s;y*=s;z*=s;w*=s
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Load authored skin and bind columns, exact bone payload records.
D0=decode_payload(BASE.read_bytes());D1=decode_payload(CAND.read_bytes());F=np.asarray(D0['indices'],np.int64).reshape(-1,3);W=np.asarray(D0['full_weights'],float);bind=np.asarray(D0['bindings_u'][:,0],int)
m=json.loads(MANIFEST.read_text());bonepath=MANIFEST.parent/m['payload']['file'];raw=bonepath.read_bytes();header=struct.calcsize('<8s5I32s');magic,abi,bc,vc,ic,fingerprint,source_sha=struct.unpack_from('<8s5I32s',raw)
if abi!=3: raise RuntimeError(f'expected bone ABI3, got {abi}')
if source_sha.hex()!=m['source']['myosim_source_archive_sha256']: raise RuntimeError('bone payload source archive hash mismatch')
recsize=struct.calcsize('<6I8fI');records=[struct.unpack_from('<6I8fI',raw,header+i*recsize) for i in range(bc)];off=header+bc*recsize
BV=np.asarray([struct.unpack_from('<6f',raw,off+i*24)[:3] for i in range(vc)],np.float32);off+=vc*24;BI=np.frombuffer(raw,dtype='<u4',count=ic,offset=off).reshape(-1,3)
names={int(a['core_body_index']):a['myosim_body'] for a in m['source']['anchors']}
owner=bind[np.argmax(W,axis=1)];reg=np.asarray([region(names.get(int(b),'')) for b in owner]);sides=np.asarray(['right' if names.get(int(b),'').endswith('_r') else 'left' if names.get(int(b),'').endswith('_l') else 'none' for b in owner]);tri_side=[];tri_reg=[]
for tri in F:
 tri_side.append(Counter(sides[tri]).most_common(1)[0][0]);tri_reg.append(Counter(reg[tri]).most_common(1)[0][0])
tri_side=np.asarray(tri_side);tri_reg=np.asarray(tri_reg);regions=('shoulder_axilla','upper_arm','elbow_forearm','wrist_carpals','palm_metacarpals','digits')
changed=np.any(D0['vertices_f'][:,:3]!=D1['vertices_f'][:,:3],axis=1);armfaces={s:np.flatnonzero((tri_side==s)&np.isin(tri_reg,regions)&np.any(changed[F],axis=1)) for s in ('left','right')}
sys.path.insert(0,'/tmp')
from wrist_sections_20261022 import P as neutral,axes
for side in ('left','right'):
 ax=np.array(axes[side]['radius_axis_toward_capitate_world']);cc=np.array(axes[side]['distal_radius_center_world_m']);st=(neutral-cc)@ax;ff=armfaces[side];armfaces[side]=ff[np.any(abs(st[F[ff]])<.060,axis=1)]
def bone_mesh(pose,side):
 state={int(b['body_index']):b for b in pose['bodies']};suf='_l' if side=='left' else '_r';vv=[];ff=[];owners=[];offset=0
 for r,a in zip(records,m['source']['anchors']):
  name=a['myosim_body'].lower()
  if not name.endswith(suf) or not any(k in name for k in ('clavicle','scapula','humerus','ulna','radius','carpal','scaphoid','lunate','triquetrum','pisiform','capitate','hamate','trapezium','trapezoid','mc_','metacarp','phalan','thumb','proxph','midph','distph')):continue
  bi,fv,nv,fi,ni,stable,tx,ty,tz,qx,qy,qz,qw,scale,source_rec=r
  rb=mat((qx,qy,qz,qw));local=(BV[fv:fv+nv].astype(float)*scale)@rb.T+np.array((tx,ty,tz));cur=state[int(bi)]['current'];world=(local@mat(cur['orientation_world_xyzw']).T+np.asarray(cur['position_world_m'])).astype(np.float32)
  indices=BI[fi//3:(fi+ni)//3].astype(np.int64)-fv+offset
  vv.append(world);ff.append(indices);owners.extend([name]*len(indices));offset+=nv
 return np.vstack(vv),np.vstack(ff),owners

def pair_audit(pts,qF,ids_a,ids_b=None):
 aa,da=records_nonzero(pts,qF,ids_a)
 if ids_b is None:
  bb=aa;db=0;pairs=_aabb_candidate_pairs(aa,aa,same_surface=True)
 else:
  bb,db=records_nonzero(pts,qF,ids_b);pairs=_aabb_candidate_pairs(aa,bb,same_surface=False)
 defects=[];seams=0;candidates=0
 for a,b in pairs:
  candidates+=1;hits=triangle_intersection_points(a[0],b[0])
  if not hits:continue
  shared=set(a[4])&set(b[4]);common={a[0][a[4].index(k)] for k in shared}
  if len(shared) in (1,2) and all(_allowed_shared_point(x,common) for x in hits):seams+=1;continue
  defects.append((a[3],b[3]))
 return {"candidate_pairs":candidates,"intersection_pairs":len(defects),"allowed_shared_seam_pairs":seams,"degenerate_faces_excluded":[da,db],"first_witnesses":[list(x) for x in defects[:16]],"all_intersection_pairs":[list(x) for x in defects]}
def cross(d,pose,side):
 skinv=posed(d,pose).astype(np.float32);sids=armfaces[side];sv_ids=np.unique(F[sids]);sv=skinv[sv_ids];sremap=np.full(len(skinv),-1,dtype=np.int64);sremap[sv_ids]=np.arange(len(sv_ids));sf=sremap[F[sids]]
 bv,bf,bowners=bone_mesh(pose,side);allv=np.vstack((sv,bv));bonesf=bf+len(sv);allf=np.vstack((sf,bonesf));qv,qf,_=coordinate_quotient(allv.tolist(),allf.tolist());
 # Exact tests use both surfaces in one welded Float32 coordinate quotient; faces do not share connectivity by construction.
 from audit_bilateral_skin_retained_poses_002 import exact_points
 pts=[]
 from fractions import Fraction
 rat=[tuple(Fraction.from_float(float(x)) for x in row) for row in qv];den=max(x.denominator for row in rat for x in row);pts=[tuple(x.numerator*(den//x.denominator) for x in row) for row in rat]
 ids_a=np.arange(len(sf),dtype=np.int64);ids_b=np.arange(len(sf),len(allf),dtype=np.int64)
 result=pair_audit(pts,np.asarray(qf,dtype=np.int64),ids_a,ids_b)
 result.update({'skin_face_count':int(len(sf)),'bone_face_count':int(len(bf)),'bone_member_count':int(len(bowners)),'bone_members':sorted(set(bowners)),'skin_vertex_count':int(len(sv_ids)),'bone_vertex_count':int(len(bv))})
 result['crossing_bone_names']=dict(Counter(bowners[b-len(sf)] for a,b in result['all_intersection_pairs']))
 result['crossing_skin_source_faces']=[int(sids[a]) for a,b in result['all_intersection_pairs']]
 return result
report={'schema':'numi.human.bilateral-arm-skin-to-bone-crossing-audit.v1','skin_source_sha256':sha(BASE),'candidate_sha256':sha(CAND),'bone_payload_sha256':sha(bonepath),'bone_manifest_sha256':sha(MANIFEST),'bone_source_archive_sha256':source_sha.hex(),'pose_frame':'each bone member record is transformed into its core-body local frame, then by the same native current body pose used by the skin LBS snapshot','method':'exact rational triangle intersection after native Float32 pose emission; arm-side skin triangles checked against same-side clavicle/scapula/humerus/radius/ulna/carpal/metacarpal/phalange bone surfaces','cases':{}}
for folder in sorted((E/'native-pose-checks/repair-005').iterdir()):
 if not folder.is_dir() or folder.name!='neutral':continue
 jp=next(folder.glob('*.skin-poses.json'),None)
 if jp is None:continue
 pose=json.loads(jp.read_text());out={}
 for label,d in (('source',D0),('candidate',D1)):
  out[label]={s:cross(d,pose,s) for s in ('left','right')}
 report['cases'][folder.name]=out
 E.joinpath('wrist-loft-rebuild-019/wrist-bone-audit035.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'pose':folder.name,'source':{s:out['source'][s]['intersection_pairs'] for s in ('left','right')},'candidate':{s:out['candidate'][s]['intersection_pairs'] for s in ('left','right')},'candidates':{s:out['candidate'][s]['candidate_pairs'] for s in ('left','right')}}),flush=True)
print('REPORT='+str(E/'wrist-loft-rebuild-019/wrist-bone-audit035.json'),flush=True)
