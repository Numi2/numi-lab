#!/usr/bin/env python3
"""Bounded 005 mesh replay for retained typed bank_0004 face 2562."""
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=ROOT/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
OUT=E/'global-inner-p1-field-owner-integration-007/replay-005-bank0004-face2562-002'
P5=E/'global-inner-p1-edge-support-403/quality-coordinate-005/candidate.npz'
P398=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz'
P395=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/p1-interior-face-support-395/refined-background-unsolved-initializer.npz'
P3=E/'global-inner-p1-edge-support-403/quality-bistellar-003/candidate-geometry-only.npz'
P2=E/'global-inner-p1-edge-support-403/build-002/candidate.npz'
P6=E/'global-inner-p1-edge-support-403/quality-transfer-006/local-overlap-005-to-002.npz'
R6=E/'global-inner-p1-edge-support-403/quality-transfer-006/transfer-report.json'
R5=E/'global-inner-p1-edge-support-403/quality-coordinate-005/final-manifest-001.json'
Q5=E/'global-inner-p1-edge-support-403/quality-coordinate-005/optimization-report-002.json'
BANK=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'
W398=E/'global-inner-p1-field-402-absolute-variable-integration/retained-399-fixed-O-witness-398.json'
ABS=E/'global-inner-p1-field-402-absolute-variable-integration/absolute_field_402.py'
CLIP=E/'robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py'
MAT14=E/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/material-parent-014-conservative-map.npz'
TARGET=.341
EXPECTED={
 P5:'5b54ffd34d74d717579fcb14677cbf2bfba01f4d4562652605b9207e65f8f698',
 P398:'8abe15f941dbc01659aee693a07e3edf66f32617c784cf0334410ca11a357890',
 P3:'176900d57eedb40255298ad77860317ab97e31a6792bc4dc76c4d48ec7d060e8',
 P2:'9c8c23c8229c46494d19f6fc12d6131667b901864dfab21cd1ef7de6a2fc6744',
 P6:'875bb77bfa65486c6856b04d3b3ad4c4ffd8d471a2fc4297d6d2809a70f5ae41',
 BANK:'39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792',
 ABS:'3a3ccbb93efa463f15575c4e9d96aef50576be9784ca1928ec741e3fefd8f446',
 CLIP:'e1b5a246e531a16d8d2438b8569d31a27c9a10712f0de063084bed87db58c436'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def req(ok,msg):
 if not bool(ok): raise RuntimeError(msg)
def loadmod(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None: raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 if OUT.exists(): raise SystemExit(f'refuse overwrite: {OUT}')
 started=time.monotonic()
 inputs=[P5,P398,P395,P3,P2,P6,R6,R5,Q5,BANK,W398,ABS,CLIP,MAT14]
 for p in inputs: req(p.is_file(),f'missing {p}')
 for p,h in EXPECTED.items(): req(sha(p)==h,f'SHA mismatch: {p}')
 r5=json.loads(R5.read_text());q5=json.loads(Q5.read_text());r6=json.loads(R6.read_text())
 req(r5.get('files',{}).get('candidate.npz')==sha(P5) and r5.get('native_admission') is False and r5.get('physical_admission') is False,'005 manifest mismatch/admission')
 req(q5.get('geometry_quality_gate_pass') is True,'005 quality gate not passing')
 req(r6.get('gates',{}).get('overall_transfer_ready') is True,'006 transfer not ready')
 for p in (P5,P2,MAT14): req(r6.get('inputs',{}).get(str(p))==sha(p),f'006 report does not pin {p}')
 owner=loadmod(E/'global-inner-p1-field-owner-integration-007/field_owner_integration_007.py','field_owner_integration_007_runner')
 z5=np.load(P5,allow_pickle=False);z398=np.load(P398,allow_pickle=False);z395=np.load(P395,allow_pickle=False);z3=np.load(P3,allow_pickle=False);z2=np.load(P2,allow_pickle=False);z6=np.load(P6,allow_pickle=False);bank=np.load(BANK,allow_pickle=False)
 X=np.asarray(z5['points_world_m']);T=np.asarray(z5['tetrahedra'],np.int64)
 Xb=np.asarray(z398['points_world_m']);Tb=np.asarray(z398['tetrahedra'],np.int64);phi0=np.asarray(z395['phi_initializer_unsolved'],np.float64)
 T3=np.asarray(z3['tetrahedra'],np.int64);T2=np.asarray(z2['tetrahedra'],np.int64)
 req(X.shape==(55553,3) and X.dtype==np.float32 and T.shape==(248662,4),'unexpected 005 domain')
 req(Xb.shape==(49951,3) and Xb.dtype==np.float32 and Tb.shape==(211439,4),'unexpected 398 domain')
 req(np.array_equal(X[:len(Xb)],Xb) and np.array_equal(z395['points_world_m'],Xb),'005/395 base point prefix mismatch')
 req(len(X)-len(Xb)==5602,'unexpected appended point count')
 for key in ('physical_outer_node_ids','physical_outer_faces','physical_shell_faces'):
  req(np.array_equal(z5[key],z398[key]),f'005 changed physical O/shell key {key}')
 o5check=np.asarray(z5['parent_input_candidate_offsets'],np.int64);r5check=np.asarray(z5['parent_input_candidate_rows'],np.int64)
 parent_rows=np.repeat(np.arange(len(T),dtype=np.int64),np.diff(o5check))
 req(np.all(z5['region_attributes'][parent_rows]==z3['region_attributes'][r5check]),'005 tet material differs from its 003 ancestry')
 # Rebuild ancestry: output 005 -> 003 -> 002 -> base 398.
 o5=np.asarray(z5['parent_input_candidate_offsets'],np.int64);r5rows=np.asarray(z5['parent_input_candidate_rows'],np.int64)
 o3=np.asarray(z3['parent_403_candidate_offsets'],np.int64);r3rows=np.asarray(z3['parent_403_candidate_rows'],np.int64)
 p2base=np.asarray(z2['parent_398_tet_row'],np.int64)
 req(T3.shape[0]==len(o3)-1 and len(p2base)==len(T2),'intermediate ancestry shape mismatch')
 o5to2,r5to2=owner.compose_parent_csrs(o5,r5rows,o3,r3rows,child_count=len(T),intermediate_count=len(T3),base_count=len(T2))
 o2=np.arange(len(T2)+1,dtype=np.int64)
 o5to398,r5to398=owner.compose_parent_csrs(o5to2,r5to2,o2,p2base,child_count=len(T),intermediate_count=len(T2),base_count=len(Tb))
 # Validate and consume 006 current material map, never 399 parts.
 m2o=np.asarray(z6['transfer_new_to_002_offsets'],np.int64);m2r=np.asarray(z6['transfer_new_to_002_parent_rows'],np.int64);m2f=np.asarray(z6['transfer_new_to_002_parent_fractions'],np.float64)
 m14o=np.asarray(z6['transfer_new_to_014_offsets'],np.int64);m14r=np.asarray(z6['transfer_new_to_014_parent_rows'],np.int64);m14f=np.asarray(z6['transfer_new_to_014_parent_fractions'],np.float64)
 req(m2o.shape==(len(T)+1,) and m2o[0]==0 and m2o[-1]==len(m2r)==len(m2f) and np.all(np.diff(m2o)>0),'006 005->002 CSR domain mismatch')
 req(np.all((m2r>=0)&(m2r<len(T2))) and np.isfinite(m2f).all() and np.all(m2f>0),'006 005->002 map invalid')
 req(m14o.shape==(len(T)+1,) and m14o[0]==0 and m14o[-1]==len(m14r)==len(m14f) and np.all(np.diff(m14o)>0),'006 005->014 CSR domain mismatch')
 req(np.isfinite(m14f).all() and np.all(m14f>0),'006 005->014 map invalid')
  # Keep the 395 interpolation as an optional diagnostic. The 402 owner
 # treats p0 as diagnostic-only; use an all-zero bounded p0 for the actual
 # constrained solve, with exact physical-O zero values.
 initializer_diagnostic={'source':'395 unsolved initializer','status':'pending'}
 try:
  diagnostic_phi,diagnostic_chosen,diagnostic_bary=owner.initialize_field_on_mesh(
   X,T,Xb,Tb,phi0,z5['physical_outer_node_ids'],o5to398,r5to398)
  initializer_diagnostic.update({'status':'pass','mapped_min':float(diagnostic_phi.min()),'mapped_max':float(diagnostic_phi.max())})
 except ValueError as exc:
  message=str(exc)
  if not message.startswith('mapped current-mesh initializer out of bounds'):
   raise
  initializer_diagnostic.update({'status':'failed_optional_bounds_diagnostic','error':message})
 # The all-zero p0 also maps appended points through the same strict
 # containing-parent ancestry. It is bounded and exactly zero on physical O.
 phi0_zero=np.zeros_like(phi0,dtype=np.float64)
 phi,chosen,chosen_bary=owner.initialize_field_on_mesh(
  X,T,Xb,Tb,phi0_zero,z5['physical_outer_node_ids'],o5to398,r5to398)
 new=np.arange(len(Xb),len(X),dtype=np.int64)
 req(np.all(chosen[new]>=0) and np.isfinite(chosen_bary[new]).all(),'some new 005 nodes lack containing base ancestry')
 O=np.asarray(z5['physical_outer_node_ids'],np.int64);req(np.array_equal(phi[O],np.zeros(len(O))),'P1 initializer changed exact O zeros')
 # Explicit coordinate-fit parent rows must agree with geometric containment.
 for node,parent,bary in zip(z5['coordinate_fit_node_rows'],z5['coordinate_fit_parent_398_tet_rows'],z5['coordinate_fit_parent_398_barycentric']):
  req(int(chosen[int(node)])==int(parent),'coordinate-fit node mapped to a different base parent tet')
  req(np.max(np.abs(chosen_bary[int(node)]-bary))<2e-5,'coordinate-fit parent barycentrics mismatch')
 # Old witness is used only to prove the retained 398 failure, not as a 005 row.
 w=json.loads(W398.read_text())
 wid=w['bank_identity']
 req(int(wid['stable_id'])==5 and wid['key']=='bank_0004' and int(wid['local_face_row'])==2562,'retained witness identity mismatch')
 triids=np.asarray(bank['bank_0004_faces'],np.int64)[2562];tri=np.asarray(bank['bank_0004_vertices'],np.float32)[triids]
 old_tet=int(w['original_tet_row']);old_bary=np.asarray(w['barycentric_constraint'],np.float64);oldids=Tb[old_tet]
 oldcap=float(old_bary[~np.isin(oldids,z398['physical_outer_node_ids'])].sum())
 req(oldcap==0.0,'retained 398 witness no longer has zero free capacity')
 clipmod=loadmod(CLIP,'human_skin_p1_obstacles_302_007');clipfn=clipmod.triangle_tet_barycentric_polygon
 req(len(clipfn(Xb[oldids].astype(np.float64),tri.astype(np.float64))[0])>0,'old witness triangle does not replay in old tet')
 # Rebuild this one typed triangle over the full current 005 tet domain.
 constraints,polys,clipmeta=owner.clip_typed_triangle_to_mesh(X,T,tri,clipfn)
 req(len(constraints)>0,'typed face 2562 did not clip on current 005 mesh')
 capacities=owner.fixed_zero_capacity_report(T,constraints,O,len(X),maximum_value=1.0)
 req(len(capacities)>0,'no polygon capacity rows')
 mincap=min(x['max_free_capacity'] for x in capacities);bad=[x for x in capacities if x['max_free_capacity']<TARGET-1e-12]
 field=None;solver_report={'status':'not_run_fixed_O_capacity_gate','pass':False,'target':TARGET}
 # Never alter fixed O. If exact zero O support leaves insufficient capacity,
 # stop immediately without invoking the LP.
 if not bad:
  solver=loadmod(ABS,'absolute_field_402_007')
  tids=[];baryrows=[]
  for ti,B in constraints:
   for b in B:tids.append(int(ti));baryrows.append(b)
  field,solver_report=solver.solve_absolute_obstacle_field(phi,T,np.asarray(tids,np.int64),np.asarray(baryrows,np.float64),O,np.zeros(len(O)),np.ones(len(X)),TARGET)
  if field is not None:
   values=np.einsum('ij,ij->i',np.asarray(baryrows),field[T[np.asarray(tids,np.int64)]])
   req(np.all(values>=TARGET-1e-8),'402 result failed rebuilt polygon verification')
   req(np.array_equal(field[O],np.zeros(len(O))),'402 changed exact O values')
 status='blocked_by_fixed_physical_O_capacity' if bad else ('bounded_one_face_005_solve_pass' if field is not None else 'bounded_one_face_005_solve_failed')
 OUT.mkdir(parents=True,exist_ok=False)
 artifact=OUT/'field-and-material-map-005.npz'
 arrays={
  'initializer_phi_005':phi,'physical_outer_node_ids':O,
  'clip_constraint_tet_ids':np.asarray([ti for ti,B in constraints for _ in range(len(B))],np.int64),
  'clip_constraint_offsets':np.asarray(np.cumsum([0]+[len(B) for _,B in constraints]),np.int64),
  'clip_constraint_tet_barycentric':np.concatenate([np.asarray(B,np.float64) for _,B in constraints]),
  'initializer_newpoint_parent_398_tet_rows':chosen[new],
  'initializer_newpoint_parent_398_barycentric':chosen_bary[new],
  'transfer_005_to_002_offsets':m2o,'transfer_005_to_002_parent_rows':m2r,'transfer_005_to_002_parent_fractions':m2f,
  'transfer_005_to_014_offsets':m14o,'transfer_005_to_014_parent_rows':m14r,'transfer_005_to_014_parent_fractions':m14f}
 if field is not None: arrays['partial_single_face_phi_005']=field
 np.savez_compressed(artifact,**arrays)
 report={
  'schema':'numi.human.global-p1-field-owner-integration-007.single-typed-face.v1','status':status,
  'native_admission':False,'physical_admission':False,'full_scene_solve_run':False,
  'coverage':{'typed_faces_rebuilt':1,'all_689_bank_clip_run':False,'old_399_parts_reused':False,'constraint_domain':'all current 005 tets after AABB pruning'},
  'current_mesh':{'path':str(P5),'sha256':sha(P5),'nodes':len(X),'tets':len(T),'base_398_prefix_exact':True,'physical_O_nodes':len(O),'physical_O_faces':len(z5['physical_outer_faces'])},
  'ancestry':{'composed_path':'005->003->002->398','entry_count':len(r5to398),'max_base_parents_per_005_tet':int(np.diff(o5to398).max()),'appended_points':len(new),'all_new_points_contained':bool(np.all(chosen[new]>=0)),'coordinate_fit_nodes':np.asarray(z5['coordinate_fit_node_rows']).tolist(),'coordinate_fit_parent_rows':np.asarray(z5['coordinate_fit_parent_398_tet_rows']).tolist(),'explicit_parent_matches':True},
  'initializer':{'path':str(P395),'sha256':sha(P395),'optional_395_pullback_diagnostic':initializer_diagnostic,'solver_p0':'all-zero bounded diagnostic initializer; 402 does not use p0 as bound, objective, or warm start','fixed_O_exact_zero':True,'appended_rows_mapped_by':'strict containing-parent 398 tetrahedron ancestry with barycentric coordinates in [0,1]','nearest_or_extrapolated_mapping_used':False},
  'material_transfer_006':{'path':str(P6),'sha256':sha(P6),'report_sha256':sha(R6),'overall_transfer_ready':True,'005_to_002_entries':len(m2r),'005_to_014_entries':len(m14r),'meaning':'conservative parent-material attribution; not an exact geometric overlap certificate or absolute mass ledger'},
  'retained_398_witness':{'path':str(W398),'sha256':sha(W398),'stable_bank_id':int(wid['stable_id']),'bank_key':'bank_0004','face_row':2562,'old_tet_row':old_tet,'old_free_capacity':oldcap,'used_as_005_constraint':False},
  'rebuilt_005_constraint':{'bank_key':'bank_0004','stable_bank_id':5,'face_row':2562,'triangle_source_vertex_ids':triids.tolist(),'aabb_candidate_tets':clipmeta['aabb_candidate_tet_count'],'intersected_tets':clipmeta['intersected_tet_count'],'polygon_vertex_rows':len(capacities),'minimum_free_capacity_at_phi_max_1':mincap,'target':TARGET,'fixed_O_conflict_count':len(bad),'fixed_O_conflicts_first32':bad[:32],'fixed_O_relaxed':False,'solver':solver_report},
  'artifact':{'path':str(artifact),'sha256':sha(artifact),'partial_one_face_solution':field is not None},
  'inputs':{str(p):sha(p) for p in inputs},
  'limitations':['One retained typed bank face only; not complete tissue coverage.','No 689-bank clipping, skin/fat extraction, native switch, or physical field admission.','006 transfer is conservative attribution with numerical overlap provenance; no absolute kg ledger.','One-face solution does not certify a valid closed-solid material field.'],
  'elapsed_seconds':time.monotonic()-started}
 (OUT/'integration-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':status,'report':str(OUT/'integration-report.json'),'artifact':str(artifact),'old_398_capacity':oldcap,'new_005_min_free_capacity':mincap,'fixed_O_conflicts':len(bad),'intersected_tets':clipmeta['intersected_tet_count'],'polygon_rows':len(capacities),'solver':solver_report.get('status'),'elapsed_s':report['elapsed_seconds']},sort_keys=True),flush=True)
 return 0 if field is not None else 2
if __name__=='__main__':raise SystemExit(main())
