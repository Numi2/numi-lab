
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib,importlib.util,json,sys,traceback
import numpy as np

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'per-bank-workers-20261027/combined-d071-b109-b61-001/candidate.npz'
A=C.parent/'audit-159/report.json'
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
TOPO=S/'per-bank-workers-20261027/b65-99/attempt-002/fan-hole-ancestry-check.json'
PREFLIGHT=S/'per-bank-workers-20261027/b65-99/attempt-004-bone-aware-connected-patch-prep-001/witness-preflight.json'
FIT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/attempt-026-806-connected-witness-fit-001/fit-run-001/report.json')
DIR=FIT.parent/'lp-direction.npz'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=D/'publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
OUT=Path(__file__).resolve().parent/'fit-run-001'
C_PIN='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
A_PIN='4cc8fd5d5e8fb651733fd380001e2b9847a0040960d7c8865c37f25fe5b8a09d'
FIT_PIN='bcc1e4b00d968816fc68aeb0e97fafb55de523c7c9a47ffa753c7f4c92254fc5'
DIR_PIN='8304a756036e9e36133e14ac11beb6eeb8b26fae4748506daa6e69da87118641'
PREFLIGHT_PIN='579ab33b2a1922af8075d7825b93a2a132e0f84aaf072b85c56a2196f9083855'
TOPO_PIN='2cf84019b3c58e9300b063e61d9b3001c3111e8dfa84985ca773b8bc020c0281'
ALPHA=1.0
SCALE_M=.001

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def frac(x):return Fraction.from_float(float(x))
def patch_topology(F,rows):
 use=Counter();adj=defaultdict(set)
 for r in rows:
  f=F[int(r)]
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
   a=int(a);b=int(b);e=tuple(sorted((a,b)));use[e]+=1;adj[a].add(b);adj[b].add(a)
 boundary=[e for e,n in use.items() if n==1];bn=defaultdict(set);deg=Counter()
 for a,b in boundary:bn[a].add(b);bn[b].add(a);deg[a]+=1;deg[b]+=1
 return set(bn),{'faces':len(rows),'boundary_edges':len(boundary),'boundary_nodes':len(bn),'boundary_degree_counts':dict(Counter(deg.values()))}
def mesh_records(points,faces,face_ids,exact):
 pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)];rec=[]
 for fid,face in zip(face_ids,faces):
  inds=tuple(map(int,face));tri=tuple(pts[i] for i in inds);n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
  if not any(n):raise RuntimeError(f'emitted degenerate face {fid}')
  lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
  rec.append((tri,lo,hi,int(fid),inds))
 return rec
def main():
 if sha(C)!=C_PIN or sha(A)!=A_PIN or sha(FIT)!=FIT_PIN or sha(DIR)!=DIR_PIN or sha(PREFLIGHT)!=PREFLIGHT_PIN or sha(TOPO)!=TOPO_PIN:
  raise RuntimeError('pinned input changed')
 fit=json.loads(FIT.read_text());roundrec=fit['rounds'][0]
 trial=next(t for t in roundrec['backtracking_trials'] if float(t['alpha'])==ALPHA)
 if not trial.get('emitted_shape_pass'):raise RuntimeError('selected saved direction alpha does not pass emitted Float32 shape')
 with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
 P32=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64);parent=np.asarray(Z['refinement_face_parent_face_row'],np.int64)
 baserows=json.loads(TOPO.read_text());pref=json.loads(PREFLIGHT.read_text())
 base_rows=sorted(set(map(int,baserows['two_ring_patch_summary'][0]['face_rows']))|set(map(int,baserows['fan_expansion'][0]['fan_face_rows'])))
 rows=sorted(set(base_rows)|set(map(int,pref['measured_expansion']['added_face_rows'])))
 boundary_nodes,topometa=patch_topology(F,rows)
 patchnodes=np.unique(F[np.asarray(rows,np.int64)].reshape(-1))
 anchor_rows=np.asarray(Z['typed_coarse_anchor_output_rows_018'],np.int64)
 fixed=np.asarray(sorted(set(map(int,boundary_nodes))|set(map(int,anchor_rows))),np.int64)
 movable=np.asarray(sorted(set(map(int,patchnodes))-set(map(int,fixed))),np.int64)
 vmap={int(n):i for i,n in enumerate(movable)}
 affected_rows=np.flatnonzero(np.any(np.isin(F,movable),axis=1)).astype(np.int64);affected_set=set(map(int,affected_rows));shape_faces=F[affected_rows]
 if len(movable)!=94 or len(affected_rows)!=235:raise RuntimeError('support changed since the fit report')
 helper=loadmod('screen65_helper',HELPER);aud=loadmod('screen65_audit',AUDIT_TOOL);exact=aud.exact_module()
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
 bmask=(ids[:,0]==65)&(ids[:,1]==99);bone=np.asarray(T[bmask],np.float64);bone_ids=np.asarray(ids[bmask,3],np.int64)
 bmap={int(v):i for i,v in enumerate(bone_ids)}
 local=(P32.astype(np.float64)-pos[65])@M[65].T-off[65]
 delta=np.asarray(np.load(DIR,allow_pickle=False)['displacement_mm'],np.float64).reshape(len(movable),3)*ALPHA
 W=P32.copy();W[movable]=(P32[movable].astype(np.float64)+(delta@M[65])/1000.0).astype(np.float32);W[fixed]=P32[fixed]
 emitted_local=(W.astype(np.float64)-pos[65])@M[65].T-off[65]
 if not np.array_equal(W[fixed].view(np.uint32),P32[fixed].view(np.uint32)):raise RuntimeError('fixed boundary drift')
 candidate=OUT/'diagnostic-alpha-1-shape-only.npz';Z['points_world_m']=W;np.savez_compressed(candidate,**Z)
 bone_records=aud.records(bone,bone_ids,exact)
 patch_records=mesh_records(emitted_local,shape_faces,affected_rows,exact)
 bgate=exact._audit_pair(patch_records,bone_records,same_surface=False)
 local_pairs={(int(f),int(b)) for f,b in bgate['triangle_pairs']}
 audit=json.loads(A.read_text());bank=next(b for b in audit['banks'] if (int(b['body']),int(b['stable_bone_id']))==(65,99))
 original={(int(f),int(b)) for f,b in bank['result']['triangle_pairs']}
 unchanged={pair for pair in original if pair[0] not in affected_set}
 current=unchanged|local_pairs
 baseline_mapped={(int(parent[f]),b) for f,b in original};cur_mapped={(int(parent[f]),b) for f,b in current}
 added=sorted(cur_mapped-baseline_mapped);removed=sorted(baseline_mapped-cur_mapped)
 # Update inside classifications on all patch nodes using the exact current bank.
 bn=np.cross(bone[:,1]-bone[:,0],bone[:,1]-bone[:,0])
 # Recreate consistently oriented exact bank records and point-location index.
 bnorm=np.cross(bone[:,1]-bone[:,0],bone[:,2]-bone[:,0]);vol6=float(sum(float(a@np.cross(b,c)) for a,b,c in bone));bnorm*=1 if vol6>0 else -1
 prepared=exact.prepare_point_location(aud.records(bone,bone_ids,exact))
 def locs(points):return sorted(int(n) for n in patchnodes if exact.point_location_prepared(tuple(frac(x) for x in points[n]),prepared)['location'] in ('inside','boundary'))
 baseline_inside=locs(local);final_inside=locs(emitted_local);new_inside=sorted(set(final_inside)-set(baseline_inside));resolved=sorted(set(baseline_inside)-set(final_inside))
 strict=len(cur_mapped)<len(baseline_mapped) and len(added)==0 and len(removed)>0
 selfgate=None
 if strict:
  comp=np.asarray([r for r in range(len(F)) if r not in affected_set],np.int64)
  tri=emitted_local[shape_faces];lo=tri.min(axis=(0,1));hi=tri.max(axis=(0,1));ct=emitted_local[F[comp]];cl=ct.min(axis=1);ch=ct.max(axis=1)
  active=np.flatnonzero(np.all(ch>=lo,axis=1)&np.all(cl<=hi,axis=1));comp_rows=comp[active]
  comp_records=mesh_records(emitted_local,F[comp_rows],comp_rows,exact)
  inside_hits=exact._audit_pair(patch_records,patch_records,same_surface=True)
  outside_hits=exact._audit_pair(patch_records,comp_records,same_surface=True)
  selfgate={'internal_pair_count':int(inside_hits['count']),'internal_pairs':inside_hits['triangle_pairs'],
   'complement_aabb_active_face_count':int(len(comp_rows)),'complement_pair_count':int(outside_hits['count']),
   'complement_pairs':outside_hits['triangle_pairs'],'pass':int(inside_hits['count'])==0 and int(outside_hits['count'])==0}
 report={'schema':'numi.human.b65_99.saved-fit-direction-shape-only-exact-screen.v1','status':'REJECTED_LAYER_GATE_FAILED_DIAGNOSTIC_TARGET_SCREEN',
  'candidate_path':str(candidate),'candidate_sha256':sha(candidate),'candidate_private_diagnostic_only':True,
  'input_pins':{'baseline_candidate_sha256':sha(C),'baseline_audit159_sha256':sha(A),'fit_report_sha256':sha(FIT),'lp_direction_sha256':sha(DIR),
   'exact_endpoint_receipt_sha256':sha(Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99/attempt-026-806-connected-witness-fit-001/current-exact-endpoints.json'))},
  'selected_backtrack_alpha':ALPHA,'fit_trial_layer_gate':{'pass':trial['emitted_layer_pass'],'violation_count':trial['emitted_layer_nonregression']['violation_count'],
   'minimum_baseline_positive_new_to_old_ratio':trial['emitted_layer_nonregression']['minimum_baseline_positive_new_to_old_ratio'],
   'worst_baseline_nonpositive_change_mm3':trial['emitted_layer_nonregression']['worst_baseline_nonpositive_change_mm3']},
  'target_bank':{'bank':[65,99],'baseline_raw_child_pairs':len(original),'candidate_updated_raw_child_pairs':len(current),
   'raw_added':sorted(current-original),'raw_removed':sorted(original-current),'baseline_mapped_parent_pairs':len(baseline_mapped),
   'candidate_mapped_parent_pairs':len(cur_mapped),'mapped_added':[list(p) for p in added],'mapped_removed':[list(p) for p in removed],
   'mapped_net_delta':len(cur_mapped)-len(baseline_mapped),'strict_mapped_reduction_no_new_mapped_identities':strict,
   'affected_local_pair_count':int(bgate['count']),'affected_local_pairs':bgate['triangle_pairs']},
  'inside_nodes':{'baseline_inside_or_boundary':baseline_inside,'candidate_inside_or_boundary':final_inside,'new_inside_or_boundary':new_inside,
   'resolved_inside_or_boundary':resolved,'count_before':len(baseline_inside),'count_after':len(final_inside)},
  'local_self_gate':selfgate,'shape_trial':trial['emitted_f32_shape'],
  'limits':['This exact target/self screen is diagnostic for alpha=1 only; layer gate failed, so no fit step is admitted.',
   'Self screen runs only if target mapped identities strictly reduce with no added target identities; otherwise no self claim is made.',
   'Other-bank identities were not screened here.'],
  'geometry_or_native_admission':False}
 path=OUT/'saved-direction-alpha-1-screen.json';path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\\n')
 print(json.dumps({'path':str(path),'sha256':sha(path),'candidate':str(candidate),'candidate_sha256':sha(candidate),
  'status':report['status'],'raw_before_after':[len(original),len(current)],'mapped_before_after':[len(baseline_mapped),len(cur_mapped)],
  'mapped_added':len(added),'mapped_removed':len(removed),'inside_before_after':[len(baseline_inside),len(final_inside)],
  'resolved_inside':resolved,'new_inside':new_inside,'self_gate':selfgate,'layer_gate':report['fit_trial_layer_gate']},indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:print(type(e).__name__,e);raise
