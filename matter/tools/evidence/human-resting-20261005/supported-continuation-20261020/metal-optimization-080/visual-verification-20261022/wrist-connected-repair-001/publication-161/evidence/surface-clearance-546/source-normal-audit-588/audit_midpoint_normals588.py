#!/usr/bin/env python3
"""Cached arithmetic audit of 585 appended midpoint normals and direct-depth exits."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
PILOT=ROOT/'D/surface-clearance-546/edge-only-refinement-585/pilot-008'
CAND=PILOT/'candidate.npz'; REFINE=PILOT/'report.json'; PARENT=ROOT/'D/surface-clearance-546/connected-topology-580/run-003/lineage/candidate.npz'
TARGET_REPORT=PILOT/'union-targets/nearest-union-exits.json'; TARGET_NPZ=PILOT/'union-targets/nearest-union-exits.npz'
ORIENT=ROOT/'publication-095/evidence/target-004/source-path-orientation-audit-112.json'
BANKS=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/all-bank-containment.csv.banks.csv')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(v):
 n=float(np.linalg.norm(v)); return (np.asarray(v,dtype=np.float64)/n,n) if n>0 else (np.full(3,np.nan),0.)
def edge_uses(F,a,b):
 out=[]
 for fi,tri in enumerate(F):
  for u,v in zip(tri,np.roll(tri,-1)):
   if (int(u)==a and int(v)==b) or (int(u)==b and int(v)==a): out.append({'face_row':int(fi),'from':int(u),'to':int(v)})
 return out
def main(out=None):
 z=np.load(CAND,allow_pickle=False); par=np.load(PARENT,allow_pickle=False); tz=np.load(TARGET_NPZ,allow_pickle=False)
 refine=json.loads(REFINE.read_text()); targets=json.loads(TARGET_REPORT.read_text()); orient=json.loads(ORIENT.read_text())
 ids=np.asarray(refine['edge_midpoint_rows'],np.int64); P=np.asarray(z['points_world_m'],np.float64); F=np.asarray(z['source_O_faces'],np.int64); R=np.asarray(z['area_reference_037_points_world_m_f64'],np.float64); RF=np.asarray(z['area_reference_037_source_O_faces'],np.int64)
 if not np.array_equal(F,RF): raise ValueError('current/reference faces differ')
 chosen=orient['component_outward_sign_from_layer_order']['0']; parity=orient['component_fine_to_coarse_winding_anchors']['0']; sign=int(chosen['chosen_outward_sign'])*int(parity['chosen_outward_parity'])
 if sign not in (-1,1): raise ValueError('O-to-I source orientation sign unresolved')
 # The current O face table inherits the source coarse O winding. The pinned orientation
 # audit resolves that winding against I-O layer vectors; apply its derived sign here.
 byrow={int(x['query_row']):x for x in targets['targets']}; recs=[]
 for row in ids:
  ix=np.flatnonzero(np.any(F==row,axis=1)); c=np.cross(P[F[ix,1]]-P[F[ix,0]],P[F[ix,2]]-P[F[ix,0]]); r=np.cross(R[RF[ix,1]]-R[RF[ix,0]],R[RF[ix,2]]-R[RF[ix,0]])
  cs=c.sum(0); rs=r.sum(0); cu,cn=unit(cs); ru,rn=unit(rs); source=np.asarray(tz['source_surface_normal_world_unit'][np.flatnonzero(tz['query_rows']==row)[0]],np.float64); source,_=unit(source)
  direct=np.asarray(tz['proposal_direction_world_unit'][np.flatnonzero(tz['query_rows']==row)[0]],np.float64); direct,_=unit(direct)
  start=np.asarray(byrow[int(row)]['start_world_f32_m'],np.float64); target=np.asarray(byrow[int(row)]['target_world_f32_m'],np.float64); delta=target-start; disp,dist=unit(delta)
  cu_out=sign*cu; ru_out=sign*ru
  pe=np.asarray(z['refinement_node_parent_edge_rows'][row],np.int64); a,b=map(int,pe)
  parent_uses=edge_uses(np.asarray(par['source_O_faces'],np.int64),a,b); child_a=edge_uses(F,a,int(row)); child_b=edge_uses(F,int(row),b)
  fan=[]
  for fi,tri in zip(ix,F[ix]):
   for u,v in zip(tri,np.roll(tri,-1)):
    if int(u)==int(row) or int(v)==int(row): fan.append((int(u),int(v)))
  fan_counts={}
  for u,v in fan:
   key=(min(u,v),max(u,v)); fan_counts.setdefault(key,[]).append((u,v))
  fan_bad=[{'edge':list(k),'uses':v} for k,v in sorted(fan_counts.items()) if len(v)!=2 or sum(1 if x<y else -1 for x,y in v)!=0]
  rec=byrow[int(row)]; inside=[list(map(int,x)) for x in rec.get('initial_inside_banks',[])]
  tidx=int(np.flatnonzero(tz['query_rows']==row)[0])
  def directions(uses): return {'count':len(uses),'direction_balance':sum(1 if u['from']<u['to'] else -1 for u in uses),'uses':uses}
  recs.append({'row':int(row),'incident_faces':len(ix),'face_kind_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(z['face_kind'][ix],return_counts=True))},
   'current_area_sum_m2':float(np.linalg.norm(c,axis=1).sum()/2),'current_raw_normal_resultant_over_area':float(np.linalg.norm(cs)/np.linalg.norm(c,axis=1).sum()),
   'reference037_area_sum_m2':float(np.linalg.norm(r,axis=1).sum()/2),'reference037_raw_normal_resultant_over_area':float(np.linalg.norm(rs)/np.linalg.norm(r,axis=1).sum()),
   'o_to_i_orientation_sign_for_coarse_face_winding':sign,'current_raw_winding_normal_world_unit':cu.tolist(),'current_outward_normal_world_unit':cu_out.tolist(),
   'reference037_raw_winding_normal_world_unit':ru.tolist(),'reference037_outward_normal_world_unit':ru_out.tolist(),'registered_source_outward_normal_world_unit':source.tolist(),
   'current_outward_dot_registered_source_outward':float(cu_out@source),'reference037_outward_dot_registered_source_outward':float(ru_out@source),
   'current_outward_dot_direct_depth_direction':float(cu_out@direct),'reference037_outward_dot_direct_depth_direction':float(ru_out@direct),
   'registered_source_outward_dot_direct_depth_direction':float(source@direct),'registered_source_outward_dot_applied_target_displacement':float(source@disp),
   'proposal_applied_cosine':float(direct@disp),'target_displacement_mm':float(dist*1000),
   'parent_edge_rows':[a,b],'parent_edge_incidence':directions(parent_uses),'child_first_half_edge_incidence':directions(child_a),'child_second_half_edge_incidence':directions(child_b),
   'midpoint_fan_winding_bad_edges':fan_bad,'boundary_edge':len(parent_uses)==1,'initial_inside_bank_body_stable_ids':inside})
 # CSV manifest intentionally contains typed body/stable-ID inventory, not anatomical names.
 with BANKS.open(newline='') as f: bank_header=next(csv.reader(f))
 summary={'schema':'numi.surface-clearance-546.midpoint-source-normal-audit.588.v1','classification':'read_only_cached_arithmetic_diagnostic_no_target_or_geometry_run',
  'inputs':{str(p):sha(p) for p in (CAND,REFINE,PARENT,TARGET_REPORT,TARGET_NPZ,ORIENT,BANKS)},
  'orientation_receipt':{'path':str(ORIENT),'fine_outward_sign':int(chosen['chosen_outward_sign']),'fine_outward_anchor_count':int(chosen['anchor_count']),'fine_outward_agreement_fraction':float(chosen['agreement_fraction']),'fine_to_coarse_winding_parity':int(parity['chosen_outward_parity']),'parity_anchor_count':int(parity['anchor_count']),'parity_agreement_fraction':float(parity['agreement_fraction']),'derived_coarse_O_outward_sign':sign,'interpretation':'current/037 source_O_faces use coarse O winding; raw cross normals require this pinned O-to-I sign to be read as outward'},
  'bank_identity_contract':{'manifest_path':str(BANKS),'manifest_sha256':sha(BANKS),'available_columns':bank_header,'anatomical_names_present':False,'reported_identity':'body index plus stable bone ID'},
  'midpoint_rows':recs,'counts':{'rows':len(recs),'source_dot_direct_depth_nonpositive':sum(r['registered_source_outward_dot_direct_depth_direction']<=0 for r in recs),'current_source_outward_dot_negative':sum(r['current_outward_dot_registered_source_outward']<0 for r in recs),'reference_source_outward_dot_negative':sum(r['reference037_outward_dot_registered_source_outward']<0 for r in recs),'parent_edges_interior_opposite_winding':sum(r['parent_edge_incidence']['count']==2 and r['parent_edge_incidence']['direction_balance']==0 for r in recs),'child_half_edges_both_interior_opposite_winding':sum(r['child_first_half_edge_incidence']['count']==2 and r['child_first_half_edge_incidence']['direction_balance']==0 and r['child_second_half_edge_incidence']['count']==2 and r['child_second_half_edge_incidence']['direction_balance']==0 for r in recs)},
  'limitations':['No point relocation or target generation performed.','Certified source-outward direction is read from the O-to-I orientation owner; no anatomy-name mapping is carried by the bank manifest.','The direct-depth endpoint targets remain diagnostic-only and all185-certified as recorded in their original report; this audit does not claim physical admission.']}
 out=Path(out) if out else Path(__file__).with_name('report.json'); out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n'); print(out,sha(out)); print(json.dumps(summary['counts'],indent=2))
if __name__=='__main__': main()
