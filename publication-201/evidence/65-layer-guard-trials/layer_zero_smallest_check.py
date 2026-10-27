from pathlib import Path
from collections import Counter
import hashlib,json,importlib.util,sys
import numpy as np
B=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027')
C=B/'b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz'
L=B/'physical-layer-diagnostic-001/guarded-65-quarter/skin-fat-layers-541.npz'
FTR=Path(__file__).resolve().parent/'fit-run-003'
R=FTR/'report.json';D=FTR/'lp-direction.npz'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
CPIN='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db'
LPIN='78916a0c05eb130e5efb02174342295c2d8d4444eae963c7b36fc43a2a1b17fc'
RPIN='0b4a9bb8f46b15a2033b80637d2a297e68347cb01912ed40d5258fc51c2ffb4b'
DPIN='8304a756036e9e36133e14ac11beb6eeb8b26fae4748506daa6e69da87118641'
ALPHA=1.0/65536.0
TOL=1e-9

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def import_owner(path):
 sp=importlib.util.spec_from_file_location('layer_gate_bank_helper',path);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
def boundary_nodes(faces,rows):
 cnt=Counter()
 for r in rows:
  a,b,c=map(int,faces[int(r)])
  for u,v in ((a,b),(b,c),(c,a)):cnt[tuple(sorted((u,v)))]+=1
 if any(v not in (1,2) for v in cnt.values()):raise ValueError('patch nonmanifold')
 edges=[e for e,n in cnt.items() if n==1];deg=Counter()
 for a,b in edges:deg[a]+=1;deg[b]+=1
 if not edges or any(n!=2 for n in deg.values()):raise ValueError('bad patch boundary')
 return np.asarray(sorted(deg),np.int64)
def one_tet(points,sign):
 p0,p1,p2,p3=points
 return float(sign*np.dot(p1-p0,np.cross(p2-p0,p3-p0))/6.0)
def main():
 for p,h in [(C,CPIN),(L,LPIN),(R,RPIN),(D,DPIN)]:
  if sha(p)!=h:raise RuntimeError(f'pin mismatch {p}: {sha(p)}')
 report=json.loads(R.read_text())
 with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
 P=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
 patch_rows=np.asarray(report['topology']['patch_face_rows'],np.int64);nodes=np.unique(F[patch_rows].ravel());bd=boundary_nodes(F,patch_rows)
 anchors=np.asarray(Z['typed_coarse_anchor_output_rows_018'],np.int64);movable=np.setdiff1d(nodes,np.union1d(bd,np.intersect1d(nodes,anchors)))
 affected=np.flatnonzero(np.any(np.isin(F,movable),axis=1)).astype(np.int64)
 with np.load(D,allow_pickle=False) as q:step=np.asarray(q['displacement_mm'],np.float64).reshape(-1,3)
 if len(step)!=len(movable):raise ValueError('direction/movable node size mismatch')
 helper=import_owner(HELPER);T,ids,g,b,M,off,pos,keys,pins,unused=helper.read_bank_inputs(BANKDIR,E)
 base=P.copy();zero=base.copy();small=base.copy()
 small[movable]=(base[movable].astype(np.float64)+(ALPHA*step@M[65])/1000.0).astype(np.float32)
 world_delta=small.astype(np.float64)-base.astype(np.float64)
 with np.load(L,allow_pickle=False) as l:
  if not np.array_equal(l['outer_faces'],F[:,[0,2,1]]):raise ValueError('layer topology mismatch')
  if not np.array_equal(l['outer_points_f32'],base):raise ValueError('layer outer baseline not exact candidate F32')
  faces=np.asarray(l['outer_faces'],np.int64)
  outer0=np.asarray(l['outer_points_f32'],np.float32).astype(np.float64)*1000
  inner0=np.asarray(l['inner_points_f32'],np.float32).astype(np.float64)*1000
  deep0=np.asarray(l['deep_points_f32'],np.float32).astype(np.float64)*1000
  innerf=np.asarray(l['inner_points_f64'],np.float64);deepf=np.asarray(l['deep_points_f64'],np.float64)
  inner1=((innerf+world_delta).astype(np.float32).astype(np.float64))*1000
  deep1=((deepf+world_delta).astype(np.float32).astype(np.float64))*1000
 outer_zero=zero.astype(np.float64)*1000
 inner_zero=(innerf.astype(np.float32).astype(np.float64))*1000
 deep_zero=(deepf.astype(np.float32).astype(np.float64))*1000
 def check(outer,inner,deep,label):
  vals=[]
  for li,(o,i) in enumerate(((outer,inner),(inner,deep))):
   for fi in map(int,affected):
    orig=faces[fi];a,b,c=sorted(map(int,orig));parity=(int(orig[0]>orig[1])+int(orig[0]>orig[2])+int(orig[1]>orig[2]))&1;e=1.0 if parity else -1.0
    tets=[([o[a],o[b],o[c],i[c]],e),([o[a],o[b],i[b],i[c]],-e),([o[a],i[a],i[b],i[c]],e)]
    for ti,(pts,sgn) in enumerate(tets):
     vals.append({'layer':'skin' if li==0 else 'fat','face_row':fi,'tet_index':ti,'signed_volume_mm3':one_tet(pts,sgn)})
  return vals
 base_vals=check(outer0,inner0,deep0,'base');zero_vals=check(outer_zero,inner_zero,deep_zero,'zero');small_vals=check(small.astype(np.float64)*1000,inner1,deep1,'smallest')
 base_by={(x['layer'],x['face_row'],x['tet_index']):x['signed_volume_mm3'] for x in base_vals}
 def assess(vals):
  out=[]
  for x in vals:
   old=base_by[(x['layer'],x['face_row'],x['tet_index'])];new=x['signed_volume_mm3'];floor=.2*old if old>0 else old;margin=new-floor
   out.append({**x,'baseline_signed_volume_mm3':old,'required_min_mm3':floor,'delta_from_baseline_mm3':new-old,'margin_mm3':margin,'baseline_positive':old>0,'pass':margin>=-TOL})
  return out
 zero_assess=assess(zero_vals);small_assess=assess(small_vals)
 failures=[x for x in small_assess if not x['pass']]
 zero_fail=[x for x in zero_assess if not x['pass']]
 report_out={'schema':'numi.human.b65_99.layer-zero-vs-smallest-f32-diagnostic.v1','candidate_path':str(C),'candidate_sha256':sha(C),'layer_path':str(L),'layer_sha256':sha(L),'fit_report_path':str(R),'fit_report_sha256':sha(R),'lp_direction_path':str(D),'lp_direction_sha256':sha(D),'emission_alpha':ALPHA,'emission_rule':'pinned base Float32 world coordinates plus (alpha*local_step_mm @ M[65])/1000, then Float32; inner/deep are the pinned transferred f64 offsets plus the emitted outer-node delta, then Float32.','affected_face_count':len(affected),'affected_tetra_count':len(base_vals),'movable_node_count':len(movable),'smallest_step_changed_node_count':int(np.count_nonzero(np.any(small.view(np.uint32)!=base.view(np.uint32),axis=1))),'smallest_step_max_node_motion_mm':float(np.linalg.norm(world_delta,axis=1).max()*1000),'base_and_zero_inner_f32_equal':bool(np.array_equal(inner0,inner_zero)),'base_and_zero_deep_f32_equal':bool(np.array_equal(deep0,deep_zero)),'zero_displacement_gate':{'failure_count':len(zero_fail),'pass':not zero_fail,'first_failures':zero_fail[:20]},'smallest_step_gate':{'failure_count':len(failures),'pass':not failures,'minimum_margin_mm3':float(min(x['margin_mm3'] for x in small_assess)),'worst_failure':min(failures,key=lambda x:x['margin_mm3']) if failures else None,'failure_identities_and_deltas':failures},'all_baseline_tetrahedra':base_vals,'all_zero_tetrahedra':zero_assess,'all_smallest_step_tetrahedra':small_assess,'geometry_or_native_admission':False}
 OUT=Path(__file__).resolve().parent/'layer-zero-smallest-diagnostic.json';OUT.write_text(json.dumps(report_out,indent=2,sort_keys=True)+'\n');print(json.dumps({k:report_out[k] for k in ['schema','candidate_sha256','layer_sha256','affected_face_count','affected_tetra_count','smallest_step_changed_node_count','smallest_step_max_node_motion_mm','base_and_zero_inner_f32_equal','base_and_zero_deep_f32_equal','zero_displacement_gate','smallest_step_gate']},indent=2))
if __name__=='__main__':main()
