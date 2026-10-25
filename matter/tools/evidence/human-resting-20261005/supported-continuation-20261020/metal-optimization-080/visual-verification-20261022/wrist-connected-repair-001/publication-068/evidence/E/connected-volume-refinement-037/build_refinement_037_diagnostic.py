from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=E/'E/connected-volume-refinement-037/run-003'
BASE=E/'route-audit-counter-fix-030/baseline-035.npz'
REF=E/'connected-volume-repair-032/repair-input.npz'
WIT=E/'route-audit-counter-fix-030/baseline035/tet-edge-report.json'
HELPER=E/'E/connected-refinement-033/canonical/matter/tools/human_skin_tetra_refinement.py'
PINS={'baseline_035_sha256':'452602aea036c15faaafe6c154e6c4676585e1b2cb57c9a1c9128de82050b570','refinement_input_sha256':'3df28e930e5b27229633bf2e3a91f0d40fd6e57d94489c3ba7de6ae2828a0e1d','helper_sha256':'11bfc32686d3e7faa29bd5f9afc00bf0f2eec154a0cb8275333eff98804dd70c'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def volq(x,t):
 p=np.asarray(x,dtype=np.float64)[np.asarray(t,dtype=np.int64)]
 a,b,c=p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]
 d=np.einsum('ij,ij->i',a,np.cross(b,c)); me=np.zeros(len(p))
 for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)): me=np.maximum(me,np.linalg.norm(p[:,i]-p[:,j],axis=1))
 return d/6,d/np.maximum(me,np.finfo(float).tiny)**3
def main():
 if OUT.exists(): raise FileExistsError(f'refuse overwrite: {OUT}')
 for p,k in ((BASE,'baseline_035_sha256'),(REF,'refinement_input_sha256'),(HELPER,'helper_sha256')):
  if sha(p)!=PINS[k]: raise RuntimeError(f'pin mismatch {p}: {sha(p)} != {PINS[k]}')
 b=np.load(BASE,allow_pickle=False); r=np.load(REF,allow_pickle=False); x=b['points_world_f32']; ids=b['global_node_ids']; t=b['tetrahedra']; mats=b['material_ids']; ref64=r['reference_points_m']
 if not (np.array_equal(x,r['points_world_f32']) and np.array_equal(t,r['tetrahedra']) and np.array_equal(mats,r['material_ids']) and np.array_equal(ids,r['global_nodes'])): raise RuntimeError('baseline-035/repair-input domain mismatch')
 if x.dtype!=np.float32 or ref64.dtype!=np.float64 or len(np.unique(ids))!=len(ids): raise RuntimeError('coordinate/global ID invariant failed')
 wr=json.loads(WIT.read_text())
 if wr.get('candidate_sha256')!=PINS['baseline_035_sha256'] or wr.get('edges_with_hits')!=2087 or wr.get('edge_bank_groups_with_hits')!=2363 or wr.get('uncertain_witnesses'): raise RuntimeError('wrong or uncertain baseline witness set')
 edges=sorted({tuple(sorted(map(int,w['endpoint_local_nodes']))) for w in wr['witnesses'] if w.get('classification')=='proper_interior_hit'})
 if len(edges)!=2087: raise RuntimeError(f'expected 2087 unique proper-hit edges, got {len(edges)}')
 for w in wr['witnesses']:
  if not np.array_equal(ids[np.asarray(w['endpoint_local_nodes'],dtype=np.int64)],np.asarray(w['endpoint_global_nodes'],dtype=np.int64)): raise RuntimeError('witness global/local endpoint mismatch')
 spec=importlib.util.spec_from_file_location('refine_tets_037',HELPER); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
 n=len(x); ptr=np.arange(n+1,dtype=np.int64)
 z=mod.refine_tetra_edges(ref64.astype(np.float32),t,edges,point_fields={'current_points_f32':x,'canonical_reference_points_f64':ref64},material_ids=mats,parent_cell_ids=np.arange(len(t),dtype=np.int64),source_vertex_indptr=ptr,source_vertex_indices=ids,source_vertex_weights=np.ones(n),minimum_normalized_quality=None,maximum_parent_reference_volume_relative_drift=None)
 xr=z['reference_points_f32']; xc=z['point_fields']['current_points_f32']; xf=z['point_fields']['canonical_reference_points_f64']; tt=z['tetrahedra']; pr=z['child_parent_rows']; n2=len(x)
 if not (np.array_equal(xr[:n2],ref64.astype(np.float32)) and np.array_equal(xc[:n2],x) and np.array_equal(xf[:n2],ref64) and np.array_equal(pr[:len(t)],np.arange(len(t)))): raise RuntimeError('original node or parent prefix changed')
 if not np.array_equal(z['material_ids'][:len(t)],mats): raise RuntimeError('material propagation prefix failed')
 ptr2,ids2,w2=z['source_vertex_ancestry_indptr'],z['source_vertex_ancestry_indices'],z['source_vertex_ancestry_weights']
 if not (np.array_equal(ptr2[:n2+1],ptr) and np.array_equal(ids2[:n2],ids) and np.array_equal(w2[:n2],np.ones(n2))): raise RuntimeError('source identity ancestry prefix changed')
 oldv,oldq=volq(ref64,t); newv,newq=volq(xf,tt); parentv=np.bincount(pr,weights=newv,minlength=len(t)); drift=np.abs(parentv-oldv)/oldv; rawf=newv/oldv[pr]
 cv,cq=volq(xc,tt); rv,rq=volq(xr,tt)
 if min(newv.min(),cv.min(),rv.min())<=0: raise RuntimeError('nonpositive F64 reference/current F32/refined F32 cell')
 if drift.max()>1e-12: raise RuntimeError(f'F64 per-parent reference volume closure failed: {drift.max()}')
 face_counts={}
 for tet in tt:
  for f in ((tet[1],tet[2],tet[3]),(tet[0],tet[3],tet[2]),(tet[0],tet[1],tet[3]),(tet[0],tet[2],tet[1])):
   k=tuple(sorted(map(int,f))); face_counts[k]=face_counts.get(k,0)+1
 bad=sum(v not in (1,2) for v in face_counts.values())
 if bad: raise RuntimeError(f'nonmanifold face incidence count={bad}')
 arrays={'reference_points_f32':xr,'current_points_f32':xc,'canonical_reference_points_f64':xf,'tetrahedra':tt,'material_ids':z['material_ids'],'child_parent_rows':pr,'child_parent_cell_ids':z['child_parent_cell_ids'],'child_reference_volume_fraction_of_input_parent_f32':z['child_reference_volume_fraction_of_input_parent'],'child_reference_volume_fraction_of_input_parent_f64':rawf,'parent_reference_volume_before_m3_f64':oldv,'parent_reference_volume_after_m3_f64':parentv,'parent_reference_volume_relative_drift_f64':drift,'new_node_parent_edges_local_rows':z['new_node_parent_edges'],'new_node_rows':z['new_node_rows'],'global_node_ids_original_prefix':ids,'source_vertex_ancestry_indptr':ptr2,'source_vertex_ancestry_global_ids':ids2,'source_vertex_ancestry_weights':w2}
 report={'status':'CONFORMING_REFINEMENT_CONSTRUCTED','input_sha256':{'baseline_035':sha(BASE),'refinement_input':sha(REF),'edge_witness':sha(WIT),'helper':sha(HELPER)},'input_equivalence':{'current_points_tetrahedra_materials_global_ids_match_repair_input':True,'reference':'repair-input.reference_points_m F64 retained; cast to F32 for orientation/refinement quality, with canonical F64 midpoint field retained','global_node_id_prefix_exact':True},'marked_edges':{'unique_exact_proper_hit_edges':len(edges),'edge_bank_witness_groups':2363,'uncertain_edges_used':0,'edge_witness_rows':len(wr['witnesses'])},'mesh':{'input_nodes':n,'output_nodes':len(xr),'new_nodes':len(z['new_node_rows']),'input_tetrahedra':len(t),'output_tetrahedra':len(tt),'touched_input_parent_rows':z['report']['touched_input_tetrahedron_count'],'materials_inherited_from_parent':True,'child_parent_row_is_dense_baseline_source_row':True,'source_ancestry_ids_are_baseline_global_node_ids':True,'bad_face_incidence':bad},'quality':{'min_reference_f32_normalized_quality':float(rq.min()),'min_current_f32_normalized_quality':float(cq.min()),'min_reference_f64_normalized_quality':float(newq.min()),'all_emitted_cells_positive_f64ref_f32ref_f32current':True,'quality_floor_required':False},'reference_volume_transfer':{'primary_frame':'canonical reference_points_m F64','raw_child_fraction_field':'child_reference_volume_fraction_of_input_parent_f64','max_parent_raw_f64_relative_drift':float(drift.max()),'max_total_f64_relative_drift':float(abs(newv.sum()-oldv.sum())/oldv.sum()),'helper_f32_parent_relative_drift':z['report']['maximum_parent_reference_volume_relative_drift'],'helper_f32_drift_limit':None,'normalized_fraction_is_bookkeeping_not_exact_overlap':True,'midpoint_off_edge_m_in_refined_f32':z['report']['maximum_midpoint_off_edge_distance_m']},'admission':{'bone_clearance':False,'native_emission':False,'physical_mass_transfer':False,'native_physics':False,'purpose':'Conforming P1 support refinement only; downstream reclip, mass/ancestry and physical gates remain required'},'helper_report':{**z['report'],'parent_reference_volume_closure_pass':'NOT_EVALUATED_DIAGNOSTIC_ONLY'}}
 OUT.mkdir(parents=True)
 np.savez_compressed(OUT/'candidate.npz',**arrays)
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 (OUT/'run-manifest.json').write_text(json.dumps({'builder_sha256':sha(Path(__file__)),'candidate_sha256':sha(OUT/'candidate.npz'),'report_sha256':sha(OUT/'report.json'),'input_sha256':report['input_sha256']},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'output':str(OUT),'candidate_sha256':sha(OUT/'candidate.npz'),'report':report},indent=2))
if __name__=='__main__': main()
