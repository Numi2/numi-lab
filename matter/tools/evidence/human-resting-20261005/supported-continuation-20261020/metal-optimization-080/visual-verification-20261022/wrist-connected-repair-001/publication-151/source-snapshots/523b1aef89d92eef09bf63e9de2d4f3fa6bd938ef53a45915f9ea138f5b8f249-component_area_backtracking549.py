"""Independent-region line search of a retained construction proposal.

No model, anatomy identity, material or physical acceptance rule changes.
The graph includes complete face and self-pair stencils, not their possibly
zero linear derivatives. Exact 545 checks remain required after this proposal.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def components(n,faces,free,pairs):
 f=np.asarray(faces);free=np.asarray(free,dtype=np.int64);pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
 if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=n):raise ValueError('invalid faces')
 if len(np.unique(free))!=len(free) or np.any(free<0) or np.any(free>=n):raise ValueError('invalid free rows')
 if np.any(pairs<0) or np.any(pairs>=len(f)):raise ValueError('invalid self stencils')
 if not len(free):return [],{'components':0,'free_nodes':0}
 ix=np.full(n,-1,dtype=np.int64);ix[free]=np.arange(len(free))
 # Face stencil cliques and complete paired-face cliques: an edge connects
 # all variables appearing in a nonlinear energy/guard, including zero derivatives.
 edge=[] 
 for stencil in (f, f[pairs].reshape(-1,6)):
  v=ix[stencil]
  for a in range(v.shape[1]):
   for b in range(a):
    z=v[:,[a,b]];edge.append(z[(z>=0).all(1)])
 e=np.concatenate(edge) if edge else np.empty((0,2),dtype=np.int64)
 nc,lab=connected_components(sparse.coo_matrix((np.ones(len(e),dtype=np.int8),(e[:,0],e[:,1])),shape=(len(free),len(free))).tocsr(),directed=False)
 groups=sorted([np.sort(free[lab==i]) for i in range(nc)],key=lambda a:int(a[0]))
 # Independently verify every full stencil has at most one mutable owner.
 owner=np.full(n,-1,dtype=np.int64)
 for i,g in enumerate(groups):owner[g]=i
 for stencil in (f,f[pairs].reshape(-1,6)):
  z=owner[stencil]
  lo=np.where(z>=0,z,nc).min(1);hi=z.max(1)
  if np.any((hi>=0)&(lo!=hi)):raise AssertionError('nonlinear stencil crosses components')
 return groups,{'components':nc,'free_nodes':len(free),'sizes':[len(g) for g in groups],'complete_face_stencils':len(f),'complete_self_pair_stencils':len(pairs),'all_stencils_single_mutable_owner':True}

def line_search(base,target,free,groups,gate,max_halvings=32):
 if base.dtype!=np.float32 or target.dtype!=np.float32 or target.shape!=base.shape or base.ndim!=2 or base.shape[1]!=3:raise ValueError('invalid Float32 point domain')
 if not np.isfinite(base).all() or not np.isfinite(target).all():raise ValueError('nonfinite points')
 if not gate(base)[0]:raise ValueError('baseline does not pass gates')
 if sorted(np.concatenate(groups).tolist())!=sorted(np.asarray(free).tolist()):raise ValueError('component partition mismatch')
 x=base.copy();bm=base.astype(np.float64)*1000.;delta=(target.astype(np.float64)-base.astype(np.float64))*1000.;records=[]
 for ci,g in enumerate(groups):
  attempts=[];selected=None
  for k in range(max_halvings+1):
   alpha=2.**(-k);trial=x.copy()
   trial[g]=((bm[g]+alpha*delta[g])/1000.).astype(np.float32)
   if same(trial[g],x[g]):break
   ok,details=gate(trial);attempts.append({'alpha':alpha,'pass':bool(ok),**details})
   if ok:x=trial;selected=alpha;break
  records.append({'component':ci,'nodes':g.tolist(),'selected_alpha':selected,'trials':attempts})
 outside=np.setdiff1d(np.arange(len(base)),np.asarray(free))
 if not same(x[outside],base[outside]):raise AssertionError('outside rows changed')
 return x,records

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,required=True);ap.add_argument('--context',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--trust-mm',type=float,required=True);ap.add_argument('--exact-self',action='store_true');a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 if not np.isfinite(a.trust_mm) or a.trust_mm<=0:raise ValueError('invalid trust')
 root=Path('/Users/numi/Work/numi-human-connected-surface-069')
 own=root/'D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
 if sha(own)!='5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2':raise ValueError('owner pin mismatch')
 spec=importlib.util.spec_from_file_location('owner408_component549',own);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 rr=json.loads((a.raw_dir/'report.json').read_text());cp=a.raw_dir/'candidate.npz';bp=Path(rr['base_candidate'])
 if sha(bp)!=rr['base_sha256'] or sha(cp)!=rr['candidate_sha256']:raise ValueError('raw/base hashes')
 b=read_npz(bp);r=read_npz(cp);x=b['points_world_m'];f=b['source_O_faces'];target=r['points_world_m'];free=np.asarray(rr['free_nodes'],dtype=np.int64)
 if set(b)!=set(r) or any(not same(b[k],r[k]) for k in b if k!='points_world_m'):raise ValueError('raw changes nonpoint payload')
 anchors=b['typed_coarse_anchor_output_rows_018']
 if np.intersect1d(free,anchors).size or not same(x[anchors],target[anchors]):raise ValueError('free anchors')
 ctx=json.loads(a.context.read_text())
 if ctx['candidate_sha256']!=sha(bp):raise ValueError('context/base mismatch')
 pairpath=Path(ctx['seed_pairs_path']);pairs=np.load(pairpath,allow_pickle=False)
 groups,proof=components(len(x),f,free,pairs)
 # Every bone guard names one complete skin-face stencil already in the graph.
 with np.load(ctx['guard_npz'],allow_pickle=False) as z:
  gf=z['skin_face_rows']
  if gf.dtype.kind not in 'iu' or np.any(gf<0) or np.any(gf>=len(f)):raise ValueError('bone guard stencil domain')
 proof['fixed_bone_guard_face_stencils']=len(gf)
 bm=x.astype(np.float64)*1000.;ref=b['area_reference_037_points_world_m_f64']*1000.
 EG,rhs,_=m.build_edge_gradient_constraints(bm,f,free,.5)
 R0=root/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
 if sha(R0)!='70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31':raise ValueError('R0 pin')
 with np.load(bp,allow_pickle=False) as z:source,_=m.compose_source_reference(z,R0)
 predicate=None;self_state=None;self_trials=[];self_baseline_report=None;predicate_path=None
 if a.exact_self:
  E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
  predicate_path=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
  if sha(predicate_path)!='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d':raise ValueError('exact predicate pin')
  sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
  ps=importlib.util.spec_from_file_location('numilab_human._component549',predicate_path);predicate=importlib.util.module_from_spec(ps);sys.modules[ps.name]=predicate;ps.loader.exec_module(predicate)
  records,_=predicate.float32_surface_records(x,f,source_seam_groups=())
  self_state,self_baseline_report=predicate._prepare_complete_self_pair_baseline(records)
  if self_state.triangle_pairs:raise ValueError('baseline has exact self intersections')
 def gate(p):
  nonlocal self_state
  area=m.full_f32_area_gate(p,bm,ref,f,.1)
  step=float(np.linalg.norm(p.astype(np.float64)-x.astype(np.float64),axis=1).max(initial=0)*1000)
  distance=float(np.linalg.norm(p.astype(np.float64)-source,axis=1).max(initial=0)*1000)
  slack=float((EG@((p.astype(np.float64)*1000-bm)[free].ravel())-rhs).min(initial=1e30))
  ok=area['pass'] and step<=a.trust_mm+1e-6 and distance<=12+1e-6 and slack>=-1e-7 and same(p[anchors],x[anchors])
  self_info={}
  if ok and predicate is not None:
   records,_=predicate.float32_surface_records(p,f,source_seam_groups=())
   proposed,ss=predicate._update_complete_self_pair_baseline(self_state,records)
   self_info={'exact_self_pairs':ss['count'],'selected_face_count':ss['selected_face_count'],'exact_incremental_reuse':ss['exact_incremental_reuse']}
   self_trials.append({**self_info,'failed_pairs_first64':ss['triangle_pairs'][:64]})
   if ss['count']:ok=False
   else:self_state=proposed
  return bool(ok),{**self_info,'area_pass':area['pass'],'minimum_current_oriented_ratio':area['minimum_current_oriented_ratio'],'minimum_fixed_037_reference_magnitude_ratio':area['minimum_fixed_037_reference_magnitude_ratio'],'maximum_step_mm':step,'maximum_source_distance_mm':distance,'minimum_edge_slack_mm':slack}
 out,records=line_search(x,target,free,groups,gate)
 if same(out,x):raise ValueError('no representable construction progress')
 a.output.mkdir(parents=True);payload=dict(b);payload['points_world_m']=out;op=a.output/'candidate.npz';np.savez_compressed(op,**payload)
 pins={str(p):sha(p) for p in [Path(__file__),own,bp,cp,a.raw_dir/'report.json',a.context,pairpath,Path(ctx['guard_npz']),R0]+([predicate_path] if predicate_path else [])}
 report={'status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED','candidate_path':str(op),'candidate_sha256':sha(op),'base_candidate':str(bp),'base_sha256':sha(bp),'free_nodes':free.tolist(),'independence_proof':proof,'components':records,'full_final_gate':gate(out)[1],'source_pins':pins,'all_nonpoint_payload_bitwise_unchanged':True,'native_admission':False,'accepted_physical_steps':0,'exact_self_predicate_315':{'baseline':self_baseline_report,'trial_summaries':self_trials,'final_pairs':len(self_state.triangle_pairs) if self_state else None},'exact_self_and_bone_admission':'EXACT_SELF_SCREEN_COMPLETE_BONE_NOT_RUN' if a.exact_self else 'NOT_RUN','scope':'offline retained raw LP direction, independent nonlinear-stencil component alpha; unchanged actual545 gates'}
 (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'candidate':str(op),'components':len(groups),'alphas':[r['selected_alpha'] for r in records],'native_admission':False}),flush=True)
if __name__=='__main__':main()
