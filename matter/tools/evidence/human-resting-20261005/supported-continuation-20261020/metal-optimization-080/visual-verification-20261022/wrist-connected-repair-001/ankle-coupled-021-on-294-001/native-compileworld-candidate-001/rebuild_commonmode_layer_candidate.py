#!/usr/bin/env python3
"""Rebuild the 294 O/I/D common-mode layer candidate from pinned inputs.

No optimization is run. Candidate-minus-base 86-binding LBS displacement at
the matched reference pose is applied in common mode to existing 294 O/I/D.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
import numpy as np

ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=ROOT/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
OUT=E/'ankle-coupled-021-on-294-001'
BASE=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin')
RED=Path('/Users/n/numi-human-separation-20261018/C/wholebody-integration-295/build-wholebody-016/source/apps/NumiHumanWholeBodySkinReduction.hpp')
CSV=Path('/Users/n/numi-human-separation-20261018/C/wholebody-integration-295/native-wholebody-001/registered-skin.csv')
POSE=E/'native-registered-skin-pose-clean-014/registered-reference-elevation-010-014/registered-skin-poses.json'
CAND=OUT/'ankle-coupled-021-on-294.nhskin'
EMISSION=OUT/'emission-receipt.json'
MAP=E/'ankle-coupled-geometry-weight-021/ankle-patch-069i-to-294-exact-index-map.npz'
RAW021=E/'ankle-coupled-geometry-weight-021/ankle-coupled-021-candidate.npz'
SOURCE069=E/'finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin'
EXPECTED_HASH={BASE:'d4353d5379f552909f8035da2603cfe9d2dc15cd02f4da78a367caab985f1a04',RED:'a0e6d69706611257556c74ab7815ca2605f1f3c5f5530d9dd3472ecf482f5200',CSV:'4dd63097a035f5f385712307f2336ef6fd46c964d1b93a2e85d84193ac12d327',POSE:'f4b375a3c1476b9e4dcf506fc4001afa8ab4079070644af7b7d473cb43366818',CAND:'d07b27e7d9d6952a3bc33d41535693b45f107f629e69520c2afbfb990ed1237f',MAP:'39011d487170e7b535295887aa96c73d485db20f96a0edf1999faf912749192d',RAW021:'ea5184dcc4652c318f0ad818dfcb135f95ed569953719727625881d6ce3a0aec',SOURCE069:'a848a41368aeb0dda8af579fe0ea417bbf7ec7db641033b5ba917302189fe1bc'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ints(text,name):
 m=re.search(r'\b'+re.escape(name)+r'\s*=\s*\{(.*?)\}\s*;',text,re.S)
 if not m: raise ValueError('missing '+name)
 return np.asarray([int(x) for x in re.findall(r'(?<![A-Za-z_])(\d+)(?:u)?',m.group(1))],np.int64)
def faces(text,name):
 m=re.search(r'\b'+re.escape(name)+r'\s*=\s*\{(.*?)\}\s*;',text,re.S)
 if not m: raise ValueError('missing '+name)
 a=np.asarray([[int(x) for x in r] for r in re.findall(r'Face\s*\{\s*(\d+)u?\s*,\s*(\d+)u?\s*,\s*(\d+)u?\s*\}',m.group(1))],np.int64)
 if a.ndim!=2 or a.shape[1]!=3: raise ValueError('bad face array '+name)
 return a
def vecs(text,name):
 p=text.index(name);e=text.index('};',p)
 a=np.asarray([[float(x.strip().rstrip('f')) for x in r.split(',')] for r in re.findall(r'std::array<double,3>\{([^}]*)\}',text[p:e])],float)
 if a.ndim!=2 or a.shape[1]!=3: raise ValueError('bad vector '+name)
 return a
def scalars(text,name):
 p=text.index(name);e=text.index('};',p);block=text[text.index('{',p)+1:e]
 return np.asarray([float(x.strip().rstrip('f')) for x in block.replace('\n','').split(',') if x.strip()],float)
def frames(text):
 pat=r'WholeBodySkinDonorFrameV1\{(\d+)u,\{([^}]*)\},\{([^}]*)\},\{([^}]*)\},\{([^}]*)\}\}'
 nums=lambda s:np.asarray([float(x.strip().rstrip('f')) for x in s.split(',')],float)
 out=[]
 for m in re.finditer(pat,text):
  out.append((int(m.group(1)),nums(m.group(2)),nums(m.group(3)).reshape(3,3),nums(m.group(4))))
 return out
def rot(q):
 x,y,z,w=map(float,q);n=np.sqrt(x*x+y*y+z*z+w*w)
 if not np.isfinite(n) or n<1e-12: raise ValueError('invalid pose quaternion')
 x,y,z,w=x/n,y/n,z/n,w/n
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def posed(d,p):
 v=np.asarray(d['vertices_f'][:,:3],float);w=np.asarray(d['full_weights'],float);bodies=np.asarray(d['bindings_u'][:,0],int);bf=np.asarray(d['bindings_f'],float)
 state={int(x['body_index']):x for x in p['bodies']};o=np.zeros_like(v);done=set()
 for j,b0 in enumerate(bodies):
  b=int(b0)
  if b in done:continue
  done.add(b);c=state[b]['current'];rc=rot(c['orientation_world_xyzw']);t=np.asarray(c['position_world_m'],float);rb=rot(bf[j,4:8]);local=(v@rb.T)*float(bf[j,8])+bf[j,1:4];o+=w[:,j,None]*(local@rc.T+t)
 return o.astype(np.float32)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('/tmp/numi-ankle-commonmode-rebuilt.npz'));ap.add_argument('--report',type=Path,default=OUT/'layer-rebuild-receipt.json');ap.add_argument('--expected',type=Path,default=OUT/'main294-commonmode-layer-candidate-001/candidate-layers.npz');args=ap.parse_args()
 for p,h in EXPECTED_HASH.items():
  if sha(p)!=h:raise ValueError(f'input hash mismatch: {p}')
 em=json.loads(EMISSION.read_text())
 if em['candidate_payload']['sha256']!=sha(CAND) or em['inputs']['map_sha256']!=sha(MAP) or em['inputs']['021_candidate_npz_sha256']!=sha(RAW021) or em['inputs']['069i_source_sha256']!=sha(SOURCE069):raise ValueError('emission receipt provenance mismatch')
 sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
 from numilab_human.skin_source_payload_preflight import decode_payload
 t=RED.read_text();sv=ints(t,'kWholeBodySkinSourceVertices');sp=ints(t,'kWholeBodySkinSourceFaces');F=faces(t,'kWholeBodySkinFaces');fr=frames(t)
 if len(sv)!=2332 or len(sp)!=4619 or F.shape!=(4619,3) or len(fr)!=2332:raise ValueError('clean 294 shape changed')
 donor=np.asarray([x[0] for x in fr],np.int64);O0=np.asarray([x[3] for x in fr])+vecs(t,'kWholeBodySkinMechanicalWorldDeltas');OL0=np.asarray([x[1] for x in fr])+vecs(t,'kWholeBodySkinMechanicalDonorLocalDeltas');I0=vecs(t,'kWholeBodySkinInnerWorld');IL0=vecs(t,'kWholeBodySkinInnerLocal');D0=vecs(t,'kWholeBodyFatDeepWorld');DL0=vecs(t,'kWholeBodyFatDeepLocal')
 if any(x.shape!=(2332,3) for x in (O0,OL0,I0,IL0,D0,DL0)):raise ValueError('layer field shape changed')
 base=decode_payload(BASE.read_bytes());cand=decode_payload(CAND.read_bytes())
 if base['binding_count']!=86 or cand['binding_count']!=86 or not np.array_equal(base['indices'],cand['indices']) or not np.array_equal(base['bindings_u'],cand['bindings_u']) or not np.array_equal(base['bindings_f'],cand['bindings_f']):raise ValueError('candidate source/topology/bindings mismatch')
 if np.any(sv<0) or np.any(sv>=cand['vertex_count']):raise ValueError('invalid direct source ID')
 with np.load(MAP,allow_pickle=False) as z:
  mapped=np.asarray(z['changed_source_rows_294'],np.int64); mapped_faces=np.asarray(z['changed_source_faces_294'],np.int64)
  if len(np.unique(mapped))!=len(mapped) or np.any(mapped<0) or np.any(mapped>=cand['vertex_count']):raise ValueError('mapped 294 source-vertex IDs are invalid or nonunique')
  if len(np.unique(mapped_faces))!=len(mapped_faces) or np.any(mapped_faces<0) or np.any(mapped_faces>=len(cand['indices'])//3):raise ValueError('mapped 294 source-face IDs are invalid or nonunique')
 delta=(posed(cand,json.loads(POSE.read_text()))[sv]-posed(base,json.loads(POSE.read_text()))[sv]).astype(np.float32).astype(float)
 R=np.asarray([x[2] for x in fr]);ld=np.einsum('nij,nj->ni',R,delta);O=O0+delta;I=I0+delta;D=D0+delta;OL=OL0+ld;IL=IL0+ld;DL=DL0+ld
 data={'outer_world_m':O,'skin_inner_world_m':I,'fat_deep_world_m':D,'outer_donor_local_m':OL,'skin_inner_donor_local_m':IL,'fat_deep_donor_local_m':DL,'faces':F,'source_vertex_ids':sv,'source_face_parents':sp,'donor_body_indices':donor,'skin_thickness_m':scalars(t,'kWholeBodySkinThicknessM'),'fat_thickness_m':np.linalg.norm(I-D,axis=1),'reference_source_delta_m':delta}
 args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**data)
 checks=[]
 with np.load(args.expected,allow_pickle=False) as exp,np.load(args.output,allow_pickle=False) as got:
  if set(exp.files)!=set(got.files):raise ValueError('NPZ keys differ')
  for k in sorted(exp.files):
   a=np.asarray(exp[k]);b=np.asarray(got[k]);eq=bool(np.array_equal(a,b));md=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
   checks.append({'key':k,'shape':list(a.shape),'dtype':str(a.dtype),'exact':eq,'max_abs_diff':md})
   if not eq:raise ValueError(f'{k} differs: {md}')
 module=Path('/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/skin_source_payload_preflight.py')
 report={'schema':'numi.human.main294.ankle-commonmode-layer-rebuild.v1','status':'arrays_exact_to_retained_candidate','method':'At matched reference pose, evaluate base/candidate 86-binding LBS in Float32; apply their mapped 2332-point difference in common mode to clean 294 O/I/D in world and frozen donor-local frames. No optimizer.','inputs':{str(p):{'sha256':sha(p)} for p in [BASE,RED,CSV,POSE,CAND,MAP,RAW021,SOURCE069,module,EMISSION]},'outputs':{'rebuilt_npz':str(args.output),'rebuilt_npz_sha256':sha(args.output),'expected_npz':str(args.expected),'expected_npz_sha256':sha(args.expected),'report':str(args.report)},'source_rows':len(sv),'faces':len(F),'changed_reference_source_rows':np.flatnonzero(np.any(delta!=0,axis=1)).astype(int).tolist(),'max_reference_source_delta_mm':float(np.linalg.norm(delta,axis=1).max()*1000),'exact_array_count':sum(x['exact'] for x in checks),'array_count':len(checks),'comparison':checks,'limits':['Static layer reconstruction only; no 021 optimization is rerun.','No deformed FEM/contact/dynamic runtime/admission claim.']}
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');json.loads(args.report.read_text())
 print(json.dumps({'status':report['status'],'npz':str(args.output),'npz_sha256':sha(args.output),'receipt':str(args.report),'exact_arrays':report['exact_array_count'],'array_count':report['array_count'],'changed_rows':report['changed_reference_source_rows'],'max_delta_mm':report['max_reference_source_delta_mm']},sort_keys=True))
if __name__=='__main__':main()
