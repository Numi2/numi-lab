#!/usr/bin/env python3
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import numpy as np

BASE=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-inner-p1-edge-support-403/build-002')
CANDIDATE=BASE/'candidate.npz'
REPORT=BASE/'float32-alternative-diagnostic.json'
FLOOR=1e-5

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def quality(points,tets):
 q=np.asarray(points,dtype=np.float64)[np.asarray(tets,dtype=np.int64)]
 det=np.linalg.det(np.stack((q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]),axis=1))
 edge=np.zeros(len(q),dtype=np.float64)
 for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
  edge=np.maximum(edge,np.linalg.norm(q[:,i]-q[:,j],axis=1))
 return det/np.maximum(edge,1e-300)**3,det

def floor_ceil(value):
 nearest=np.float32(value); nearest64=float(nearest)
 if nearest64<value:
  return nearest,np.nextafter(nearest,np.float32(np.inf),dtype=np.float32)
 if nearest64>value:
  return np.nextafter(nearest,np.float32(-np.inf),dtype=np.float32),nearest
 return nearest,nearest

def main():
 if sha(CANDIDATE)!='9c8c23c8229c46494d19f6fc12d6131667b901864dfab21cd1ef7de6a2fc6744':
  raise SystemExit('frozen build-002 candidate hash mismatch')
 with np.load(CANDIDATE,allow_pickle=False) as z:
  X=z['points_world_m'];T=z['tetrahedra'];parents=z['parent_398_tet_row']
  edges=z['edge_support_parent_endpoints'];edge_rows=z['edge_support_midpoint_node_rows']
  face_vertices=z['face_support_parent_vertices'];face_rows=z['face_support_centroid_node_rows']
  shell=z['physical_shell_faces']; outer=z['physical_outer_faces']
  Q,_=quality(X,T);bad=np.flatnonzero(Q<FLOOR)
  edge_by_node={int(n):tuple(map(int,e)) for n,e in zip(edge_rows,edges)}
  face_by_node={int(n):tuple(map(int,f)) for n,f in zip(face_rows,face_vertices)}
  bad_rows=[];node_ids=set()
  for ti in bad:
   supports=[]
   for n in T[ti]:
    n=int(n)
    if n in edge_by_node: supports.append({'node':n,'kind':'edge_midpoint','parents':list(edge_by_node[n])});node_ids.add(n)
    elif n in face_by_node: supports.append({'node':n,'kind':'face_centroid','parents':list(face_by_node[n])});node_ids.add(n)
   bad_rows.append({'output_tet_row':int(ti),'parent_398_tet_row':int(parents[ti]),'vertices':T[ti].astype(int).tolist(),'normalized_determinant':float(Q[ti]),'support_nodes':supports})
  shell_nodes=set(np.asarray(shell,dtype=np.int64).reshape(-1).tolist())
  outer_nodes=set(np.asarray(outer,dtype=np.int64).reshape(-1).tolist())
  node_reports=[]
  star_sets={}
  for node in sorted(node_ids):
   if node not in edge_by_node:
    node_reports.append({'node':node,'kind':'face_centroid','result':'not-enumerated; task requested midpoint floor/ceil search'})
    continue
   a,b=edge_by_node[node]; ideal=(X[a].astype(np.float64)+X[b].astype(np.float64))*0.5
   axes=[floor_ceil(x) for x in ideal]
   raw=list(itertools.product(*axes))
   candidates={tuple(float(v) for v in c) for c in raw}
   candidates.add(tuple(float(v) for v in X[node]))
   incident=np.flatnonzero(np.any(T==node,axis=1))
   star_sets[node]=set(map(int,incident))
   local_tets=T[incident]
   node_positions=np.argwhere(local_tets==node)
   if len(node_positions)!=len(incident): raise AssertionError('unexpected duplicate node within one tet')
   local_coords=X[local_tets].copy()
   flat=node_positions[:,0]*4+node_positions[:,1]
   current_q,_=quality(X,T[incident])
   rows=[]
   for cand in sorted(candidates):
    value=np.asarray(cand,dtype=np.float32)
    trial=local_coords.copy();trial.reshape(-1,3)[flat]=value
    q=quality(trial.reshape(-1,3),np.arange(len(incident)*4,dtype=np.int64).reshape(-1,4))[0]
    # The local flattened quality call uses the four vertices per incident tet.
    off=np.linalg.norm(np.cross(value.astype(np.float64)-X[a].astype(np.float64),X[b].astype(np.float64)-X[a].astype(np.float64)))/np.linalg.norm(X[b].astype(np.float64)-X[a].astype(np.float64))
    rows.append({'coordinate_f32':value.tolist(),'min_star_normalized_determinant':float(q.min()),'below_1e_5_count_in_star':int(np.count_nonzero(q<FLOOR)),'nonpositive_count_in_star':int(np.count_nonzero(q<=0)),'movement_from_frozen_node_um':float(np.linalg.norm(value.astype(np.float64)-X[node].astype(np.float64))*1e6),'distance_from_exact_edge_m':float(off),'delta_from_exact_edge_midpoint_um':((value.astype(np.float64)-ideal)*1e6).tolist(),'is_frozen_coordinate':bool(np.array_equal(value,X[node]))})
   rows.sort(key=lambda r:(r['below_1e_5_count_in_star'],r['nonpositive_count_in_star'],-r['min_star_normalized_determinant'],r['movement_from_frozen_node_um']))
   node_reports.append({'node':node,'kind':'edge_midpoint','parent_edge':[a,b],'incident_output_tet_rows':incident.astype(int).tolist(),'incident_parent_398_rows':parents[incident].astype(int).tolist(),'incident_tet_count':int(len(incident)),'current_min_star_normalized_determinant':float(current_q.min()),'current_below_1e_5_count_in_star':int(np.count_nonzero(current_q<FLOOR)),'exact_midpoint_f64_m':ideal.tolist(),'frozen_coordinate_f32_m':X[node].tolist(),'floor_ceil_values_per_axis':[[float(v) for v in ax] for ax in axes],'raw_floor_ceil_combinations':len(raw),'unique_candidates_including_frozen':len(candidates),'is_physical_shell_node':node in shell_nodes,'is_physical_outer_node':node in outer_nodes,'candidate_evaluations':rows,'any_candidate_clears_full_star':any(r['below_1e_5_count_in_star']==0 and r['nonpositive_count_in_star']==0 for r in rows)})
  star_overlaps=[{'node_a':int(a),'node_b':int(b),'shared_output_tet_count':len(star_sets[a]&star_sets[b])} for i,a in enumerate(sorted(star_sets)) for b in sorted(star_sets)[i+1:]]
  result={'support_star_pairwise_intersections':star_overlaps,'schema':'numi.global-inner-p1-edge-support-403.float32-alternative-diagnostic.v1','candidate_path':str(CANDIDATE),'candidate_sha256':sha(CANDIDATE),'quality_floor':FLOOR,'input_geometry_mutated':False,'candidate_emitted':False,'bad_cell_count':int(len(bad)),'bad_cells':bad_rows,'affected_support_node_count':len(node_ids),'support_nodes':node_reports,'selection':'For each edge midpoint appearing in a failing cell, enumerate the Float32 floor/ceil product around its exact F64 endpoint midpoint and the frozen coordinate; evaluate every coordinate on the full incident tet star. No coordinate was written.','result':'No tested one-node floor/ceil alternative clears all cells in its incident star.' if all(not x.get('any_candidate_clears_full_star',False) for x in node_reports if x.get('kind')=='edge_midpoint') else 'At least one one-node floor/ceil alternative clears its incident star; multi-node interactions remain untested.'}
 REPORT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'report_path':str(REPORT),'report_sha256':sha(REPORT),'bad_cell_count':len(bad),'nodes':[{k:x[k] for k in ('node','parent_edge','current_min_star_normalized_determinant','current_below_1e_5_count_in_star','raw_floor_ceil_combinations','unique_candidates_including_frozen','any_candidate_clears_full_star')} for x in node_reports],'result':result['result']},indent=2))
if __name__=='__main__': main()
