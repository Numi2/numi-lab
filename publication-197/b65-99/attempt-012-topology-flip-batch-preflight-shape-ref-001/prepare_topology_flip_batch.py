from __future__ import annotations
import hashlib, importlib.util, json, sys, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
INPUT=S/'per-bank-workers-20261027/b65-99/attempt-008-witness-direction-backtrack-prepared-001/candidate-private-iteration-01.npz'
INPUT_REPORT=INPUT.parent/'report.json'
PARENT=S/'fit-from-663-001/seed-retry-663-001/fraction-small-001/candidate.npz'
OWNER402=ROOT/'local-two-parent-retriangulation-402/two_parent_patch_flips_402.py'
REPLAY405=ROOT/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
REPLAY534=ROOT/'connected-quality-patch-534/replay_surviving_flips_534.py'
REFRESH=ROOT/'surface-clearance-546/component-sat-selector-638/edge-refinement-661-threebank-001/refresh_guards_multi_parent_generation002.py'
OUT=Path(__file__).resolve().parent
PINS={
 'input_candidate':'8931f3f240f595f2df17e91cb4f4c82688721e778e81bcbc4acc7f6264d61d70',
 'input_report':'5ce52aa8e793a0ff8cf202e9efa00f9dee01823f6cf62144426725612c4a3d0e',
 'parent_candidate':'beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35',
 'owner402':'f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764',
 'replay405':'e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967',
 'replay534':'4559fd425a3300528f49e5211afbd633674910ecaa0d6671408ee0e9d145e504',
 'refresh546':'90bafb6a51e9d7a2dd9b49570fcd977423616e18e992c2d4a4306feea5bd5eab'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError(f'cannot load {path}')
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def csr_rows(ptr,vals,n):
 p=np.asarray(ptr);v=np.asarray(vals)
 if p.shape!=(n+1,) or int(p[0])!=0 or int(p[-1])!=len(v) or np.any(p[1:]<p[:-1]):raise ValueError('invalid CSR')
 return [tuple(map(int,v[int(p[i]):int(p[i+1])])) for i in range(n)]
def patch_boundary(faces,rows):
 use=defaultdict(int)
 for r in rows:
  a,b,c=map(int,faces[int(r)])
  for e in ((a,b),(b,c),(c,a)):use[tuple(sorted(e))]+=1
 if any(n not in (1,2) for n in use.values()):raise ValueError('patch has nonmanifold edge incidence')
 return {e for e,n in use.items() if n==1}
def quality(points,refpoints,faces,reffaces):
 tri=np.asarray(points,np.float64)[faces];rt=np.asarray(refpoints,np.float64)[reffaces]
 pn=np.cross(rt[:,1]-rt[:,0],rt[:,2]-rt[:,0]);cn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 p2=np.einsum('ij,ij->i',pn,pn);plen=np.sqrt(np.maximum(p2,1e-300))
 ur=np.linalg.norm(cn,axis=1)/plen;sr=np.einsum('ij,ij->i',pn,cn)/np.maximum(p2,1e-300)
 angles=[];aspects=[]
 for t in tri:
  e=[float(np.linalg.norm(t[(i+1)%3]-t[i])) for i in range(3)]
  area=float(np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))*.5)
  aspects.append(max(e)**2/(4*np.sqrt(3)*max(area,1e-30)))
  aa=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
   aa.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-30),-1,1)))))
  angles.append(min(aa))
 return {'face_count':len(faces),'minimum_unsigned_parent_area_ratio':float(ur.min()),
  'minimum_signed_orientation_ratio':float(sr.min()),'orientation_positive':bool(np.all(sr>0)),
  'minimum_angle_degrees':float(min(angles)),'p05_angle_degrees':float(np.percentile(angles,5)),
  'maximum_aspect_measure':float(max(aspects))}


def quality_after_topology(points,basepoints,basefaces,newfaces,rows,operations):
 # Keep the Float32 point field fixed. Unchanged faces use their original
 # emitted-F32 triangle as reference; each changed 2-to-2 pair uses its old
 # oriented quadrilateral area vector and total area as the local parent chart.
 p=np.asarray(points,np.float64);bp=np.asarray(basepoints,np.float64)
 nf=np.asarray(newfaces,np.int64);bf=np.asarray(basefaces,np.int64)
 op_by_row={int(r):op for op in operations for r in op['face_rows']}
 area_ratios=[];signed=[];angles=[];aspects=[]
 for r in map(int,rows):
  tri=nf[r];q=p[tri];cn=np.cross(q[1]-q[0],q[2]-q[0]);cl=float(np.linalg.norm(cn))
  if cl<=0:raise ValueError(f'Float32 face {r} is degenerate')
  if r in op_by_row:
   pair=list(map(int,op_by_row[r]['face_rows']))
   oldcross=[]
   for oldr in pair:
    t=bf[oldr];x=bp[t];oldcross.append(np.cross(x[1]-x[0],x[2]-x[0]))
   refn=np.sum(oldcross,axis=0);refarea=sum(float(np.linalg.norm(n)) for n in oldcross)
   if float(np.linalg.norm(refn))<=0 or refarea<=0:raise ValueError(f'flip pair {pair} has degenerate reference area')
   area_ratios.append(cl/refarea)
   signed.append(float(refn@cn)/(float(np.linalg.norm(refn))*cl))
  else:
   t=bf[r];x=bp[t];rn=np.cross(x[1]-x[0],x[2]-x[0]);rl=float(np.linalg.norm(rn))
   if rl<=0:raise ValueError(f'baseline face {r} is degenerate')
   area_ratios.append(cl/rl);signed.append(float(rn@cn)/(rl*cl))
  e=[float(np.linalg.norm(q[(i+1)%3]-q[i])) for i in range(3)]
  area=cl*.5;aspects.append(max(e)**2/(4*np.sqrt(3)*max(area,1e-30)))
  aa=[]
  for i in range(3):
   u=q[(i+1)%3]-q[i];v=q[(i+2)%3]-q[i]
   aa.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-30),-1,1)))))
  angles.append(min(aa))
 return {'face_count':len(rows),'minimum_unsigned_parent_area_ratio':float(min(area_ratios)),
  'minimum_signed_orientation_cosine':float(min(signed)),'orientation_positive':bool(min(signed)>0),
  'minimum_angle_degrees':float(min(angles)),'p05_angle_degrees':float(np.percentile(angles,5)),
  'maximum_aspect_measure':float(max(aspects)),
  'area_orientation_reference':'unchanged rows use their pre-flip emitted F32 triangle; changed rows use old paired quadrilateral aggregate'}

def run():
 report={'schema':'numi.human.bank65_99.topology-flip-batch-preflight.v1','status':'STARTED',
  'candidate_modified':False,'geometry_or_native_admission':False,
  'scope':'One private deterministic, node-disjoint 2-to-2 flip batch in the 248-face 65/99 patch; use 402 shape/topology proposal guards, 534 batch ancestry replay and 546 lineage refresh. No exact bone or self screen in this preflight.'}
 try:
  for name,path in [('input_candidate',INPUT),('input_report',INPUT_REPORT),('parent_candidate',PARENT),('owner402',OWNER402),('replay405',REPLAY405),('replay534',REPLAY534),('refresh546',REFRESH)]:
   got=sha(path)
   if got!=PINS[name]:raise RuntimeError(f'{name} pin mismatch: {got} != {PINS[name]}')
  report['input_pins']={name:{'path':str(path),'sha256':sha(path)} for name,path in [('input_candidate',INPUT),('input_report',INPUT_REPORT),('parent_candidate',PARENT),('owner402',OWNER402),('replay405',REPLAY405),('replay534',REPLAY534),('refresh546',REFRESH)]}
  srcdoc=json.loads(INPUT_REPORT.read_text());round0=srcdoc['rounds'][0]
  rows=sorted(set(map(int,srcdoc['topology']['patch_face_rows'])))
  if len(rows)!=248:raise RuntimeError(f'expected 248 patch faces; got {len(rows)}')
  exact_pairs=sorted(set((int(f),int(b)) for f,b in round0['exact_local_bone_pairs']))
  if len(exact_pairs)!=148:raise RuntimeError(f'expected 148 saved exact pairs; got {len(exact_pairs)}')
  with np.load(INPUT,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
  immediate_hash=str(np.asarray(base['refinement_input_candidate_sha256']).reshape(-1)[0])
  archived_generation002_hash=str(np.asarray(base['prior_input_parent_546_generation_002__refinement_input_candidate_sha256']).reshape(-1)[0])
  if immediate_hash!=PINS['parent_candidate']:raise RuntimeError(f'immediate parent hash mismatch: {immediate_hash}')
  with np.load(PARENT,allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
  owner=loadmod('b65_99_topology_owner402',OWNER402)
  replay=loadmod('b65_99_topology_replay405',REPLAY405)
  compose=loadmod('b65_99_topology_composer534',REPLAY534)
  refresh=loadmod('b65_99_topology_refresh546',REFRESH)
  F=np.asarray(base['source_O_faces'],np.int64);P=np.asarray(base['points_world_m'],np.float32)
  nface=len(F);patch=set(rows);cross={f for f,_ in exact_pairs}
  src=csr_rows(base['face_source_ancestry_indptr'],base['face_source_ancestry_face_rows'],nface)
  sub=csr_rows(base['face_fine_subface_ancestry_indptr'],base['face_fine_subface_ancestry_rows'],nface)
  det=csr_rows(base['face_fine_detail_ancestry_indptr'],base['face_fine_detail_ancestry_rows'],nface)
  par=csr_rows(base['refinement_face_parent_ancestry_indptr'],base['refinement_face_parent_ancestry_face_rows'],nface)
  patch_edges=defaultdict(list)
  for r in rows:
   a,b,c=map(int,F[r])
   for u,v in ((a,b),(b,c),(c,a)):
    patch_edges[tuple(sorted((u,v)))].append((int(r),u,v))
  outer={e for e,inc in patch_edges.items() if len(inc)==1}
  seam={e for e,inc in patch_edges.items() if len(inc)==2 and
    (det[inc[0][0]]!=det[inc[1][0]] or sub[inc[0][0]]!=sub[inc[1][0]] or
     int(base['face_kind'][inc[0][0]])!=int(base['face_kind'][inc[1][0]]))}
  locked=outer|seam
  em=owner.edge_map(F)
  requested=set(map(int,np.asarray(base['parent_fine_detail_face_row'])[rows]))
  eligible=[];rejected={};checked=0
  for edge,inc in em.items():
   if len(inc)!=2 or not all(int(x[0]) in patch for x in inc):continue
   if not any(int(x[0]) in cross for x in inc):continue
   checked+=1
   i,j=map(lambda x:int(x[0]),inc)
   # Never cross a source-detail/subface/material seam even when scalar labels are -1.
   if det[i]!=det[j] or sub[i]!=sub[j] or int(base['face_kind'][i])!=int(base['face_kind'][j]):
    rejected['ancestry_or_material_seam']=rejected.get('ancestry_or_material_seam',0)+1;continue
   op,why=owner.candidate_flip(base,edge,inc,requested,locked,0.0,0.0,em)
   if op is None:rejected[why]=rejected.get(why,0)+1;continue
   fr=list(map(int,op['face_rows']));
   if set(fr)-patch:raise RuntimeError('owner proposed a flip outside the local patch')
   if op['immediate_parent_union']!=sorted(set(par[fr[0]])|set(par[fr[1]])):
    raise RuntimeError('owner immediate-parent union differs from live CSR')
   op=dict(op);op['old_faces']=F[fr].tolist()
   op['crossing_pairs_touched']=sorted([list(q) for q in exact_pairs if q[0] in set(fr)])
   op['crossing_pair_count_touched']=len(op['crossing_pairs_touched'])
   op['source_ancestry_union']=sorted(set(src[fr[0]])|set(src[fr[1]]))
   op['detail_ancestry_union']=sorted(set(det[fr[0]])|set(det[fr[1]]))
   op['subface_ancestry_union']=sorted(set(sub[fr[0]])|set(sub[fr[1]]))
   eligible.append(op)
  eligible.sort(key=lambda q:(-q['crossing_pair_count_touched'],-float(q['angle_gain_deg']),q['face_rows'],q['edge']))
  chosen=[];usedf=set();usedn=set()
  for op in eligible:
   fr=set(map(int,op['face_rows']));nodes={int(x) for t in op['old_faces'] for x in t}
   if fr&usedf or nodes&usedn:continue
   chosen.append(op);usedf|=fr;usedn|=nodes
  if not chosen:raise RuntimeError('no nonconflicting owner-accepted batch operations')
  out=compose.apply_ops_batch(base,chosen,replay)
  owner_report={'helper':'connected-quality-patch-534/replay_surviving_flips_534.apply_ops_batch','applied_operation_count':len(chosen),'array_updates':'source, detail, subface and immediate-parent CSR unions; scalar -1 and NaN barycentrics for multi-parent rows'}
  if not np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']):raise RuntimeError('037 connectivity stale')
  if not np.array_equal(out['points_world_m'].view(np.uint32),P.view(np.uint32)):raise RuntimeError('topology batch changed emitted Float32 points')
  changed=sorted({int(r) for op in chosen for r in op['face_rows']})
  if sorted(map(int,np.flatnonzero(np.any(out['source_O_faces']!=F,axis=1))))!=changed:raise RuntimeError('changed face rows differ from batch manifest')
  if compose.topology(out['source_O_faces'])[1] or compose.topology(F)[0]!=compose.topology(out['source_O_faces'])[0]:raise RuntimeError('full-surface boundary/nonmanifold guard failed')
  if patch_boundary(F,rows)!=patch_boundary(out['source_O_faces'],rows):raise RuntimeError('local patch boundary changed')
  if len({tuple(sorted(map(int,t))) for t in out['source_O_faces']})!=nface:raise RuntimeError('duplicate face introduced')
  allow=set(compose.ALLOWED_CHANGED_KEYS)
  unexpected=[]
  for k,v in base.items():
   if k not in allow and (k not in out or v.dtype!=out[k].dtype or v.shape!=out[k].shape or v.tobytes()!=out[k].tobytes()):unexpected.append(k)
  if unexpected:raise RuntimeError(f'unapproved array changes: {unexpected[:10]}')
  # Verify changed-face CSR unions and explicit multi-parent representation.
  changes=[]
  op_by_row={int(r):op for op in chosen for r in op['face_rows']}
  osrc=csr_rows(out['face_source_ancestry_indptr'],out['face_source_ancestry_face_rows'],nface)
  osub=csr_rows(out['face_fine_subface_ancestry_indptr'],out['face_fine_subface_ancestry_rows'],nface)
  odet=csr_rows(out['face_fine_detail_ancestry_indptr'],out['face_fine_detail_ancestry_rows'],nface)
  opar=csr_rows(out['refinement_face_parent_ancestry_indptr'],out['refinement_face_parent_ancestry_face_rows'],nface)
  for f in changed:
   op=op_by_row[f];fr=list(map(int,op['face_rows']));union_parent=sorted(set(par[fr[0]])|set(par[fr[1]]))
   union_source=sorted(set(src[fr[0]])|set(src[fr[1]]));union_sub=sorted(set(sub[fr[0]])|set(sub[fr[1]]));union_detail=sorted(set(det[fr[0]])|set(det[fr[1]]))
   if list(opar[f])!=union_parent or list(osrc[f])!=union_source or list(osub[f])!=union_sub or list(odet[f])!=union_detail:raise RuntimeError(f'CSR union not regenerated for face {f}')
   if len(union_parent)>1 and (int(out['refinement_face_parent_face_row'][f])!=-1 or not np.isnan(out['refinement_face_parent_barycentric'][f]).all()):raise RuntimeError(f'multi-parent scalar/NaN guard failed at face {f}')
   changes.append({'face_row':f,'operation_face_rows':fr,'immediate_parent_union':union_parent,'source_union':union_source,'detail_union':union_detail,'subface_union':union_sub,'is_multi_parent':len(union_parent)>1})
  owner.validate_candidate(out)
  # Run lineage refresh on all rows and the unchanged saved crossings; omit only flipped rows
  # because their exact new intersection identities are pending the authorized screen.
  keep_pairs=[[f,b] for f,b in exact_pairs if f not in set(changed)]
  surface={'banks':[{'body':65,'stable_bone_id':99,'result':{'triangle_pairs':keep_pairs}}]}
  lineage=refresh.validate_ancestry(out,parent,surface)
  floor=float(srcdoc['optimizer_parameters']['shape_gate_min_angle_deg'])
  cap=float(srcdoc['optimizer_parameters']['shape_gate_max_aspect'])
  area_floor=float(srcdoc['optimizer_parameters']['area_floor'])
  shape=quality_after_topology(out['points_world_m'],base['points_world_m'],F,out['source_O_faces'],rows,chosen)
  shape_pass=bool(shape['orientation_positive'] and shape['minimum_unsigned_parent_area_ratio']>=area_floor and
    shape['minimum_angle_degrees']>=floor and shape['maximum_aspect_measure']<=cap)
  if not shape_pass:raise RuntimeError(f'batch emitted Float32 shape gate failed: {shape}')
  multi_ops=[op for op in chosen if len(op['immediate_parent_union'])>1]
  report.update({'status':'SHAPE_AND_LINEAGE_PREFLIGHT_PASSED_PENDING_EXACT_LOCAL_SCREEN',
   'input_candidate_path':str(INPUT),'input_candidate_sha256':PINS['input_candidate'],
   'lineage_trace':{'current_refinement_input_candidate_sha256':immediate_hash,'immediate_parent_path':str(PARENT),
     'archived_generation002_input_candidate_sha256':archived_generation002_hash,
     'parent_domain_source':'the current refinement_input_candidate_sha256 pin; archived generation-002 hash remains historical lineage'},
   'saved_exact_pair_count':len(exact_pairs),'saved_crossing_face_count':len(cross),
   'patch_face_count':len(rows),'patch_boundary_edge_count':len(outer),
   'patch_interior_crossing_neighborhood_edge_count':checked,
   'owner_eligible_operations':len(eligible),'owner_rejection_reasons':rejected,
   'deterministic_sort':'descending saved exact pair count touched; descending 402 angle gain; then face_rows and edge',
   'selected_operation_count':len(chosen),'selected_multi_parent_operation_count':len(multi_ops),
   'selected_operations':chosen,'changed_face_csr_receipts':changes,
   'owner_batch_report':owner_report,'whole_surface_boundary_preserved':True,
   'local_patch_boundary_preserved':True,'coordinates_bitwise_unchanged':True,
   'shape_gate':{'pass':shape_pass,'area_ratio_floor':area_floor,'minimum_angle_floor_deg':floor,'maximum_aspect_cap':cap,'measured':shape},
   'refresh_546_lineage_guard':lineage,'refresh_guard_crossing_scope':'Existing saved 65/99 pairs on unchanged face rows only; changed rows are explicitly omitted until exact screen.',
   'exact_bone_screen':'NOT_RUN_PENDING_AUTHORIZED_SLOT','exact_self_screen':'NOT_RUN_PENDING_AUTHORIZED_SLOT',
   'canonical_identity_improvement':'NOT_CLAIMED_PENDING_EXACT_SCREEN','candidate_modified':False})
  dst=OUT/'candidate-private.npz';np.savez_compressed(dst,**out)
  report['private_candidate_path']=str(dst);report['private_candidate_sha256']=sha(dst)
 except Exception as e:
  report.update({'status':'PREFLIGHT_FAILED','failure':repr(e),'traceback':traceback.format_exc(),'candidate_modified':False})
 rp=OUT/'preflight.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+chr(10))
 report['script_path']=str(Path(__file__));report['script_sha256']=sha(Path(__file__))
 # Refresh with script pin included.
 rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+chr(10))
 print(json.dumps({k:v for k,v in report.items() if k not in ('traceback',)},indent=2,sort_keys=True,allow_nan=False))
 print('preflight_path',rp,'preflight_sha256',sha(rp))

if __name__=='__main__':run()
