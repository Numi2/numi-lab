import sys,json,hashlib,math,time
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
BASE=E/'candidate-bilateral-arm-hand-connected-repair-012.nhskin'
CAND=E/'candidate-bilateral-arm-hand-weights-016.nhskin'
MANIFEST=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.cardiac_cavity_intersections import _records,_aabb_candidate_pairs,triangle_intersection_points,_allowed_shared_point
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mat(q):
 x,y,z,w=map(float,q);s=x*x+y*y+z*z+w*w;scale=1/math.sqrt(s);x,y,z,w=x*scale,y*scale,z*scale,w*scale
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def posed(d,p):
 V=np.asarray(d['vertices_f'][:,:3],float);W=np.asarray(d['full_weights'],float);B=np.asarray(d['bindings_u'][:,0],int);BF=np.asarray(d['bindings_f'],float);state={int(x['body_index']):x for x in p['bodies']};out=np.zeros_like(V);done=set()
 for j,b in enumerate(B):
  b=int(b)
  if b in done:continue
  done.add(b);cur=state[b]['current'];Rc=mat(cur['orientation_world_xyzw']);t=np.asarray(cur['position_world_m']);Rb=mat(BF[j,4:8]);local=(V@Rb.T)*BF[j,8]+BF[j,1:4];out+=W[:,j,None]*(local@Rc.T+t)
 return out.astype(np.float32)
def region(n):
 n=n.lower()
 if 'clavicle' in n or 'scapula' in n:return 'shoulder_axilla'
 if n in ('humerus_r','humerus_l'):return 'upper_arm'
 if n.startswith(('ulna_','radius_')):return 'elbow_forearm'
 if any(k in n for k in ('scaphoid','lunate','triquetrum','pisiform','capitate','trapezium','trapezoid','hamate')):return 'wrist_carpals'
 if 'mc_' in n or 'metacarp' in n:return 'palm_metacarpals'
 if 'thumb' in n or any(k in n for k in ('proxph','midph','distph','phal')):return 'digits'
 return 'other'
def edge_metrics(V,F,ids):
 f=F[ids];T=V[f].astype(float);C=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);L=np.linalg.norm(C,axis=1);A=L*.5;N=C/np.maximum(L[:,None],1e-30);ed=defaultdict(list)
 for i,t in enumerate(f):
  for k in range(3):ed[(min(int(t[k]),int(t[(k+1)%3])),max(int(t[k]),int(t[(k+1)%3])))].append(i)
 ang=np.array([math.degrees(math.acos(float(np.clip(N[o[0]]@N[o[1]],-1,1)))) for o in ed.values() if len(o)==2])
 return {'face_count':len(f),'degenerate_faces':int(np.sum(A<=1e-12)),'area_mm2':{'p05':float(np.percentile(A,5)*1e6),'p50':float(np.percentile(A,50)*1e6),'p95':float(np.percentile(A,95)*1e6),'max':float(A.max()*1e6)},'normal_turn_deg':{'edge_count':len(ang),'p50':float(np.percentile(ang,50)) if len(ang) else None,'p95':float(np.percentile(ang,95)) if len(ang) else None,'p99':float(np.percentile(ang,99)) if len(ang) else None,'over_90':int(np.sum(ang>90)),'over_120':int(np.sum(ang>120)),'over_150':int(np.sum(ang>150)),'max':float(ang.max()) if len(ang) else None}}
def exact_points(V):
 rat=[tuple(Fraction.from_float(float(x)) for x in row) for row in V];den=max(x.denominator for row in rat for x in row);return [tuple(x.numerator*(den//x.denominator) for x in row) for row in rat]
def prep(V,F):
 qv,qf,_=coordinate_quotient(np.asarray(V,dtype=np.float32).tolist(),F.tolist());return exact_points(np.asarray(qv,dtype=np.float64)),np.asarray(qf,dtype=np.int64)
def records_nonzero(pts,qF,ids):
 kept=[];deg=0
 for fid in map(int,ids):
  a,b,c=[pts[int(i)] for i in qF[fid]];u=[b[k]-a[k] for k in range(3)];v=[c[k]-a[k] for k in range(3)];cr=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
  if cr==(0,0,0):deg+=1
  else:kept.append(fid)
 rs=_records(pts,qF[kept].tolist()) if kept else []
 rs=[(r[0],r[1],r[2],kept[r[3]],r[4]) for r in rs]
 return rs,deg
def pair_audit(pts,qF,ids_a,ids_b=None):
 aa,da=records_nonzero(pts,qF,ids_a)
 if ids_b is None:bb=aa;db=0;pairs=_aabb_candidate_pairs(aa,aa,same_surface=True)
 else:bb,db=records_nonzero(pts,qF,ids_b);pairs=_aabb_candidate_pairs(aa,bb,same_surface=False)
 defects=[];seams=0;candidates=0
 for a,b in pairs:
  candidates+=1;hits=triangle_intersection_points(a[0],b[0])
  if not hits:continue
  shared=set(a[4])&set(b[4]);common={a[0][a[4].index(k)] for k in shared}
  if len(shared) in (1,2) and all(_allowed_shared_point(x,common) for x in hits):seams+=1;continue
  defects.append((a[3],b[3]))
 return {'candidate_pairs':candidates,'intersection_pairs':len(defects),'allowed_shared_seam_pairs':seams,'degenerate_faces_excluded':[da,db],'first_witnesses':[list(x) for x in defects[:16]]}
def main():
 D0=decode_payload(BASE.read_bytes());D1=decode_payload(CAND.read_bytes());F=np.asarray(D0['indices'],int).reshape(-1,3);W=np.asarray(D0['full_weights'],float);bind=np.asarray(D0['bindings_u'][:,0],int);names={int(x['core_body_index']):x['myosim_body'] for x in json.loads(MANIFEST.read_text())['source']['anchors']};owner=bind[np.argmax(W,axis=1)];regions=np.asarray([region(names.get(int(x),'')) for x in owner]);sides=np.asarray(['right' if names.get(int(x),'').endswith('_r') else 'left' if names.get(int(x),'').endswith('_l') else 'none' for x in owner]);fs=[];fr=[]
 for tri in F:fs.append(Counter(sides[tri]).most_common(1)[0][0]);fr.append(Counter(regions[tri]).most_common(1)[0][0])
 fs=np.asarray(fs);fr=np.asarray(fr);rgs=('shoulder_axilla','upper_arm','elbow_forearm','wrist_carpals','palm_metacarpals','digits');arm={s:np.flatnonzero((fs==s)&np.isin(fr,rgs)) for s in ('right','left')};byreg={(s,r):np.flatnonzero((fs==s)&(fr==r)) for s in ('right','left') for r in rgs};sums=W.sum(axis=1)
 report={'schema':'numi.human.bilateral-arm-hand-native-pose-audit.v3','source_sha256':sha(BASE),'candidate_sha256':sha(CAND),'native_binary_sha256':json.loads((E/'native-pose-checks/repair-005/neutral/result.json').read_text())['binary_sha256'],'source_vertices':len(W),'source_triangles':len(F),'weight_sums':{'min':float(sums.min()),'max':float(sums.max()),'negative_count':int(np.sum(W<0))},'pose_cases':{},'method':{'pose':'native articulated transform snapshots, both payloads evaluated by same registered LBS transform','geometry':'posed Float32 skin coordinates; regional edge/area metrics; exact AABB triangle predicates','crossing':'each side arm versus every nonarm skin face, with exact-coordinate seam welding and valid shared edge/vertex exemption','limitations':['does not qualify coupled skin/fat FEM transfer, layer-cell Jacobians, regional physical mass, contact, or accepted-state replay','skin-to-skin intersections do not classify bone, tendon, or other tissue contact','the eight cases below are deterministic wrist flexion/extension probes, not an existing external eight-pose qualification set']}}
 for folder in sorted((E/'native-pose-checks/repair-005').iterdir()):
  if not folder.is_dir():continue
  jp=next(folder.glob('*.skin-poses.json'),None)
  if not jp:continue
  pose=json.loads(jp.read_text());rr={}
  for label,d in (('source',D0),('candidate',D1)):
   t0=time.monotonic();V=posed(d,pose);pts,qF=prep(V,F);entry={'sides':{},'posed_coordinate_count':len(pts)};allf=np.arange(len(F))
   for side in ('right','left'):
    ids=arm[side];bg=np.setdiff1d(allf,ids,assume_unique=True);row={'complete_arm_self':pair_audit(pts,qF,ids),'arm_vs_nonarm_skin':pair_audit(pts,qF,ids,bg),'regions':{r:edge_metrics(V,F,byreg[(side,r)]) for r in rgs if len(byreg[(side,r)])}}
    entry['sides'][side]=row
   entry['wall_seconds']=time.monotonic()-t0;rr[label]=entry
  V0=posed(D0,pose);V1=posed(D1,pose);delta=np.linalg.norm(V1.astype(float)-V0.astype(float),axis=1);armv=np.unique(np.concatenate([F[arm[s]].ravel() for s in ('right','left')]))
  argv=json.loads((folder/'result.json').read_text())['argv'];qargs=[argv[i:i+3] for i,x in enumerate(argv) if x=='--pose-q']
  rr['comparison']={'pose_q_arguments':qargs,'arm_vertex_displacement_mm':{'p50':float(np.percentile(delta[armv],50)*1000),'p95':float(np.percentile(delta[armv],95)*1000),'p99':float(np.percentile(delta[armv],99)*1000),'max':float(delta[armv].max()*1000)}};report['pose_cases'][folder.name]=rr
  out=E/'bilateral-arm-hand-native-pose-audit-016.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'pose':folder.name,'self_pairs':{s:{k:rr[k]['sides'][s]['complete_arm_self']['intersection_pairs'] for k in ('source','candidate')} for s in ('left','right')},'cross_skin_pairs':{s:{k:rr[k]['sides'][s]['arm_vs_nonarm_skin']['intersection_pairs'] for k in ('source','candidate')} for s in ('left','right')},'seconds':{k:rr[k]['wall_seconds'] for k in ('source','candidate')}}),flush=True)
 out=E/'bilateral-arm-hand-native-pose-audit-016.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print('REPORT='+str(out),flush=True)
if __name__=='__main__':main()
