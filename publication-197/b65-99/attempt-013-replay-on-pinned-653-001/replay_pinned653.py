from __future__ import annotations
import hashlib, importlib.util, json, sys, traceback
from pathlib import Path
from collections import defaultdict
import numpy as np

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
OUT=S/'per-bank-workers-20261027/b65-99/attempt-013-replay-on-pinned-653-001'
BASE=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
INPUT=S/'per-bank-workers-20261027/b65-99/attempt-008-witness-direction-backtrack-prepared-001/candidate-private-iteration-01.npz'
INPUT_REPORT=INPUT.parent/'report.json'
PREFLIGHT=S/'per-bank-workers-20261027/b65-99/attempt-012-topology-flip-batch-preflight-shape-ref-001/preflight.json'
PARENT=S/'fit-from-663-001/seed-retry-663-001/fraction-small-001/candidate.npz'
OWNER=D/'local-two-parent-retriangulation-402/two_parent_patch_flips_402.py'
REPLAY405=D/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
REPLAY534=D/'connected-quality-patch-534/replay_surviving_flips_534.py'
REFRESH=S/'edge-refinement-661-threebank-001/refresh_guards_multi_parent_generation002.py'
PREP=S/'per-bank-workers-20261027/b65-99/attempt-012-topology-flip-batch-preflight-shape-ref-001/prepare_topology_flip_batch.py'
EXPECTED={
 'pinned653':'7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af',
 'backtrack':'8931f3f240f595f2df17e91cb4f4c82688721e778e81bcbc4acc7f6264d61d70',
 'preflight':'6169a812653299153a6a3ad205e6305a263b4d614f414fe1a8eff4581c0a6dff',
 'parent':'beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35',
 'owner':'f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764',
 'replay405':'e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967',
 'replay534':'4559fd425a3300528f49e5211afbd633674910ecaa0d6671408ee0e9d145e504',
 'refresh':'90bafb6a51e9d7a2dd9b49570fcd977423616e18e992c2d4a4306feea5bd5eab',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
 a=np.ascontiguousarray(a); b=np.ascontiguousarray(b)
 return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def mod(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m); return m
def csr(ptr,vals,n):
 p=np.asarray(ptr);v=np.asarray(vals)
 if p.shape!=(n+1,) or int(p[0])!=0 or int(p[-1])!=len(v) or np.any(p[1:]<p[:-1]): raise ValueError('bad CSR')
 return [tuple(map(int,v[int(p[i]):int(p[i+1])])) for i in range(n)]
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 rep={'schema':'numi.human.bank65_99.pinned653-eight-flip-topology-replay.v1',
      'status':'STARTED','candidate_modified':False,'geometry_or_native_admission':False,
      'exact_bone_screen':'NOT_RUN','exact_self_screen':'NOT_RUN',
      'scope':'Replay the eight retained 402 owner flips against pinned 653, rechecking operation eligibility, ancestry unions, topology and emitted-F32 shape only.'}
 try:
  for label,path,key in [('pinned653',BASE,'pinned653'),('backtrack',INPUT,'backtrack'),('preflight',PREFLIGHT,'preflight'),('parent',PARENT,'parent'),('owner',OWNER,'owner'),('replay405',REPLAY405,'replay405'),('replay534',REPLAY534,'replay534'),('refresh',REFRESH,'refresh')]:
   if sha(path)!=EXPECTED[key]: raise ValueError(f'{label} SHA mismatch: {sha(path)}')
  doc=json.loads(INPUT_REPORT.read_text()); pf=json.loads(PREFLIGHT.read_text())
  with np.load(BASE,allow_pickle=False) as z: base={k:z[k] for k in z.files}
  with np.load(INPUT,allow_pickle=False) as z: back={k:z[k] for k in z.files}
  with np.load(PARENT,allow_pickle=False) as z: parent={k:z[k] for k in z.files}
  if not np.array_equal(base['source_O_faces'],back['source_O_faces']): raise ValueError('base topology differs from backtrack input')
  moved=np.flatnonzero(np.any(base['points_world_m']!=back['points_world_m'],axis=1))
  if len(moved)!=75: raise ValueError(f'expected 75 moved nodes in rejected backtrack, got {len(moved)}')
  maxdelta=float(np.linalg.norm(np.asarray(back['points_world_m'],np.float64)-np.asarray(base['points_world_m'],np.float64),axis=1).max())
  if maxdelta>1e-3: raise ValueError(f'unexpected backtrack delta {maxdelta}')
  if not np.array_equal(base['source_O_faces'],base['area_reference_037_source_O_faces']): raise ValueError('base 037 connectivity mismatch')
  if str(np.asarray(base['refinement_input_candidate_sha256']).reshape(-1)[0])!=EXPECTED['parent']: raise ValueError('pinned 653 immediate-parent hash mismatch')
  rep['input_pins']={'pinned653':{'path':str(BASE),'sha256':sha(BASE)},
   'rejected_backtrack':{'path':str(INPUT),'sha256':sha(INPUT),'report':str(INPUT_REPORT),'report_sha256':sha(INPUT_REPORT)},
   'prior_flip_manifest':{'path':str(PREFLIGHT),'sha256':sha(PREFLIGHT)},
   'immediate_refinement_parent':{'path':str(PARENT),'sha256':sha(PARENT)},
   'owner402':{'path':str(OWNER),'sha256':sha(OWNER)},'replay534':{'path':str(REPLAY534),'sha256':sha(REPLAY534)},
   'refresh546_generation002':{'path':str(REFRESH),'sha256':sha(REFRESH)}}
  r0=doc['rounds'][0]
  rep['rejected_backtrack_origin']={'status':doc['status'],'source_method':doc['scope'],
   'solver':r0.get('solver'),'solver_status':r0.get('solver_status'),
   'selected_backtrack_factor':r0.get('selected_backtrack_factor'),
   'changed_coordinate_count_from_pinned653':int(len(moved)),
   'max_node_delta_m':maxdelta,'reported_max_cumulative_displacement_mm':r0.get('max_cumulative_displacement_mm'),
   'local_65_99_exact_pairs':r0.get('exact_local_bone_child_pair_count'),
   'parent_identity_improved':r0.get('parent_identity_improved'),
   'complete_all_bank_audit':'NOT_PRESENT_IN_ATTEMPT008_REPORT',
   'candidate_not_used_as_replay_base':True}
  rows=sorted(set(map(int,doc['topology']['patch_face_rows'])))
  if len(rows)!=248: raise ValueError(f'expected 248 patch rows, got {len(rows)}')
  ops=pf['selected_operations']; chosen_ids=[0,1,2,3,4,6,7,8]
  owner=mod('p653_owner402',OWNER); compose=mod('p653_replay534',REPLAY534)
  replay=mod('p653_replay405',REPLAY405); refresh=mod('p653_refresh546',REFRESH)
  shapehelper=mod('p653_shapehelper',PREP)
  owner.validate_candidate(base)
  F=np.asarray(base['source_O_faces'],dtype=np.int64); n=len(F); em=owner.edge_map(F)
  src=csr(base['face_source_ancestry_indptr'],base['face_source_ancestry_face_rows'],n)
  sub=csr(base['face_fine_subface_ancestry_indptr'],base['face_fine_subface_ancestry_rows'],n)
  det=csr(base['face_fine_detail_ancestry_indptr'],base['face_fine_detail_ancestry_rows'],n)
  par=csr(base['refinement_face_parent_ancestry_indptr'],base['refinement_face_parent_ancestry_face_rows'],n)
  local=defaultdict(list)
  for f in rows:
   a,b,c=map(int,F[f])
   for u,v in ((a,b),(b,c),(c,a)): local[tuple(sorted((u,v)))].append((f,u,v))
  outer={e for e,inc in local.items() if len(inc)==1}
  seam={e for e,inc in local.items() if len(inc)==2 and (
   det[inc[0][0]]!=det[inc[1][0]] or sub[inc[0][0]]!=sub[inc[1][0]] or
   int(base['face_kind'][inc[0][0]])!=int(base['face_kind'][inc[1][0]]))}
  locked=outer|seam
  requested=set(map(int,np.asarray(base['parent_fine_detail_face_row'])[rows]))
  accepted=[];checks=[]
  for oi in chosen_ids:
   old=ops[oi]; fr=tuple(map(int,old['face_rows'])); edge=tuple(map(int,old['edge']))
   c={'prior_operation_index':oi,'face_rows':list(fr),'edge':list(edge)}
   if edge not in em: c.update(status='INCOMPATIBLE',reason='edge_absent'); checks.append(c); continue
   if not np.array_equal(F[list(fr)],np.asarray(old['old_faces'],dtype=F.dtype)):
    c.update(status='INCOMPATIBLE',reason='old_faces_mismatch'); checks.append(c); continue
   proposal,why=owner.candidate_flip(base,edge,em[edge],requested,locked,0.0,0.0,em)
   if proposal is None: c.update(status='INCOMPATIBLE',reason=str(why)); checks.append(c); continue
   if tuple(proposal['face_rows'])!=fr or not np.array_equal(np.asarray(proposal['new_faces'],dtype=F.dtype),np.asarray(old['new_faces'],dtype=F.dtype)):
    c.update(status='INCOMPATIBLE',reason='owner_rows_or_new_faces_differ',
      owner_rows=proposal['face_rows'],owner_new_faces=proposal['new_faces']); checks.append(c); continue
   x=dict(old); x.update(proposal); x['face_rows']=list(fr); x['edge']=list(edge); x['old_faces']=F[list(fr)].tolist()
   x['immediate_parent_union']=sorted(set(par[fr[0]])|set(par[fr[1]]))
   x['source_ancestry_union']=sorted(set(src[fr[0]])|set(src[fr[1]]))
   x['subface_ancestry_union']=sorted(set(sub[fr[0]])|set(sub[fr[1]]))
   x['detail_ancestry_union']=sorted(set(det[fr[0]])|set(det[fr[1]]))
   accepted.append(x)
   c.update(status='REVALIDATED',old_min_angle_deg=float(proposal['old_min_angle_deg']),
    new_min_angle_deg=float(proposal['new_min_angle_deg']),angle_gain_deg=float(proposal['angle_gain_deg']),
    immediate_parent_union=x['immediate_parent_union']); checks.append(c)
  seenf=set(); seenn=set()
  for x in accepted:
   fr=set(map(int,x['face_rows'])); nn=set(map(int,F[list(fr)].ravel()))
   if fr&seenf or nn&seenn: raise ValueError(f'operation conflict at {sorted(fr)}')
   seenf|=fr; seenn|=nn
  out=compose.apply_ops_batch(base,accepted,replay)
  owner.validate_candidate(out)
  changed=sorted(map(int,np.flatnonzero(np.any(out['source_O_faces']!=F,axis=1))))
  expect=sorted({int(r) for op in accepted for r in op['face_rows']})
  if changed!=expect: raise ValueError(f'changed rows mismatch: {changed} != {expect}')
  if not same(out['points_world_m'],base['points_world_m']): raise ValueError('coordinates changed')
  outside=sorted(set(range(n))-set(changed))
  if not np.array_equal(out['source_O_faces'][outside],F[outside]): raise ValueError('outside topology changed')
  if not np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']): raise ValueError('037 connectivity stale')
  b0,m0=compose.topology(F); b1,m1=compose.topology(out['source_O_faces'])
  if m0 or m1 or b0!=b1: raise ValueError('whole-surface boundary/nonmanifold changed')
  if shapehelper.patch_boundary(F,rows)!=shapehelper.patch_boundary(out['source_O_faces'],rows): raise ValueError('patch boundary changed')
  bad=[]
  for k,v in base.items():
   if k not in compose.ALLOWED_CHANGED_KEYS and (k not in out or not same(v,out[k])): bad.append(k)
  if bad: raise ValueError(f'unapproved array mutations {bad[:10]}')
  if set(out)-set(base)-compose.ALLOWED_CHANGED_KEYS: raise ValueError('unexpected array keys added')
  osrc=csr(out['face_source_ancestry_indptr'],out['face_source_ancestry_face_rows'],n)
  osub=csr(out['face_fine_subface_ancestry_indptr'],out['face_fine_subface_ancestry_rows'],n)
  odet=csr(out['face_fine_detail_ancestry_indptr'],out['face_fine_detail_ancestry_rows'],n)
  opar=csr(out['refinement_face_parent_ancestry_indptr'],out['refinement_face_parent_ancestry_face_rows'],n)
  lineage_rows=[]
  for f in changed:
   op=next(x for x in accepted if f in x['face_rows']); fr=list(map(int,op['face_rows']))
   want={'source':sorted(set(src[fr[0]])|set(src[fr[1]])),
    'subface':sorted(set(sub[fr[0]])|set(sub[fr[1]])),
    'detail':sorted(set(det[fr[0]])|set(det[fr[1]])),
    'parent':sorted(set(par[fr[0]])|set(par[fr[1]]))}
   got={'source':list(osrc[f]),'subface':list(osub[f]),'detail':list(odet[f]),'parent':list(opar[f])}
   if want!=got: raise ValueError(f'CSR union mismatch face {f}')
   wantp=want['parent'][0] if len(want['parent'])==1 else -1
   if int(out['refinement_face_parent_face_row'][f])!=wantp: raise ValueError(f'parent scalar mismatch face {f}')
   if len(want['parent'])>1 and not np.isnan(out['refinement_face_parent_barycentric'][f]).all():
    raise ValueError(f'multi-parent barycentric marker mismatch face {f}')
   lineage_rows.append({'face_row':f,'operation_rows':fr,'unions':want})
  lineage=refresh.validate_ancestry(out,parent,{'banks':[]})
  p0=shapehelper.quality_after_topology(base['points_world_m'],base['points_world_m'],F,F,rows,[])
  p1=shapehelper.quality_after_topology(out['points_world_m'],base['points_world_m'],F,out['source_O_faces'],rows,accepted)
  opt=doc['optimizer_parameters']; area=float(opt['area_floor']); ang=float(opt['shape_gate_min_angle_deg']); asp=float(opt['shape_gate_max_aspect'])
  shape_pass=bool(p1['orientation_positive'] and p1['minimum_unsigned_parent_area_ratio']>=area and
   p1['minimum_angle_degrees']>=ang and p1['maximum_aspect_measure']<=asp)
  dst=OUT/'candidate-pinned653-eight-flip.npz'; np.savez_compressed(dst,**out)
  rep.update({'status':'TOPOLOGY_PREFLIGHT_PASSED_PENDING_EXACT_AUDIT' if shape_pass else 'PRIVATE_REJECTED_SHAPE_GATE',
   'revalidated_operation_count':len(accepted),'incompatible_operation_count':len(chosen_ids)-len(accepted),
   'operations':checks,'changed_face_rows':changed,'changed_face_row_count':len(changed),
   'coordinates_bitwise_unchanged_from_pinned653':True,'topology_identical_outside_changed_rows':True,
   'whole_surface_boundary_edge_count':len(b0),'whole_surface_boundary_preserved':True,
   'whole_surface_nonmanifold_edges_before':len(m0),'whole_surface_nonmanifold_edges_after':len(m1),
   'patch_boundary_edge_count':len(shapehelper.patch_boundary(F,rows)),'patch_boundary_preserved':True,
   'unexpected_array_mutations':bad,'lineage_refresh':lineage,'changed_row_ancestry_receipts':lineage_rows,
   'shape_gate':{'pass':shape_pass,'area_ratio_floor':area,'minimum_angle_floor_deg':ang,'maximum_aspect_cap':asp,
    'pinned653_baseline':p0,'replayed_candidate':p1},
   'quality_improvement':{'per_operation_angle_gains_deg':[float(x['angle_gain_deg']) for x in accepted],
    'mean_operation_min_angle_gain_deg':float(np.mean([x['angle_gain_deg'] for x in accepted])) if accepted else None,
    'max_operation_min_angle_gain_deg':float(max([x['angle_gain_deg'] for x in accepted])) if accepted else None},
   'candidate_path':str(dst),'candidate_sha256':sha(dst),
   'incremental_audit_scope':'Same pinned 653 coordinates; exact all-bank intersections can be updated by replacing changed-face records after screening changed faces against all bank triangles. Exact skin self-pairs must replace every pair incident to changed rows after screening them against the complete skin. Not run here.',
   'candidate_modified':False,'geometry_or_native_admission':False,
   'exact_bone_screen':'NOT_RUN_PENDING_COORDINATION','exact_self_screen':'NOT_RUN_PENDING_COORDINATION'})
 except Exception as e:
  rep.update({'status':'PREFLIGHT_FAILED','failure':repr(e),'traceback':traceback.format_exc(),'candidate_modified':False})
 rep['script_path']=str(Path(__file__)); rep['script_sha256']=sha(Path(__file__))
 rp=OUT/'report.json'; rp.write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+'\n')
 rep['report_path']=str(rp); rep['report_sha256']=sha(rp)
 print(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__': main()

