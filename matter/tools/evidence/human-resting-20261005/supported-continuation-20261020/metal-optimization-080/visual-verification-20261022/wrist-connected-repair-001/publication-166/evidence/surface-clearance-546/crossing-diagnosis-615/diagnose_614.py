from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
D=ROOT/'D/surface-clearance-546'
STATE=D/'prefix-field-transfer-614/from-612-001/final/state.json'
STATE_OBJ=json.loads(STATE.read_text())
RET=Path(STATE_OBJ['candidate'])
SURF=Path(STATE_OBJ['surface_report'])
MERGE=D/'prefix-field-transfer-614/from-612-001/merge/candidate.npz'
MERGE_REPORT=D/'prefix-field-transfer-614/from-612-001/merge/report.json'
BASE=D/'component-topology-599/run-001/lineage/candidate.npz'
SOURCE=ROOT/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
OWNER=ROOT/'publication-126/sources/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
OUT=D/'crossing-diagnosis-615/diagnosis.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
    x=np.asarray(a,dtype=float).ravel()
    if not len(x):return {'count':0}
    return {'count':int(len(x)),'min':float(np.min(x)),'p10':float(np.quantile(x,.1)),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'max':float(np.max(x))}
def area_values(points_mm,base_mm,ref_mm,faces,rows):
    f=np.asarray(faces,dtype=np.int64)[rows]
    c=np.asarray(points_mm,dtype=np.float64);b=np.asarray(base_mm,dtype=np.float64);r=np.asarray(ref_mm,dtype=np.float64)
    cb=np.cross(b[f[:,1]]-b[f[:,0]],b[f[:,2]]-b[f[:,0]])
    ct=np.cross(c[f[:,1]]-c[f[:,0]],c[f[:,2]]-c[f[:,0]])
    cr=np.cross(r[f[:,1]]-r[f[:,0]],r[f[:,2]]-r[f[:,0]])
    ab=np.linalg.norm(cb,axis=1);ar=np.linalg.norm(cr,axis=1)
    at=np.linalg.norm(ct,axis=1)
    return {'current_oriented_ratio':np.einsum('ij,ij->i',ct,cb)/(ab*ab),
            'fixed037_magnitude_ratio':at/ar,
            'orientation_vs037_ratio':np.einsum('ij,ij->i',ct,cr)/(ar*ar)}
def area_summary(values):
    r={'current_oriented_ratio':stats(values['current_oriented_ratio']),
       'fixed037_magnitude_ratio':stats(values['fixed037_magnitude_ratio']),
       'orientation_vs037_ratio':stats(values['orientation_vs037_ratio'])}
    for k,v in values.items():r[k+'_below_0.1_count']=int(np.count_nonzero(v<.1))
    return r
sp=importlib.util.spec_from_file_location('w408diag',OWNER);w=importlib.util.module_from_spec(sp);sys.modules[sp.name]=w;sp.loader.exec_module(w)
report=json.loads(SURF.read_text())
pairs=[]
bank_counts=[]
for b in report['banks']:
    rr=b.get('result') or {};ps=rr.get('triangle_pairs') or []
    if ps:
        bank_counts.append({'body':int(b['body']),'stable_bone_id':int(b['stable_bone_id']),'pair_count':len(ps)})
        pairs.extend((int(b['body']),int(b['stable_bone_id']),int(x[0]),int(x[1])) for x in ps)
pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,4)
with np.load(RET,allow_pickle=False) as z:
    cur={k:z[k].copy() for k in z.files}
    source_ref,source_receipt=w.compose_source_reference(z,SOURCE)
with np.load(MERGE,allow_pickle=False) as z:
    merge={k:z[k].copy() for k in z.files}
    merge_ref,_=w.compose_source_reference(z,SOURCE)
with np.load(BASE,allow_pickle=False) as z:
    base={k:z[k].copy() for k in z.files}
    base_ref,_=w.compose_source_reference(z,SOURCE)
faces=np.asarray(cur['source_O_faces'],dtype=np.int64);pcur=np.asarray(cur['points_world_m'],dtype=np.float64)*1000
pmerge=np.asarray(merge['points_world_m'],dtype=np.float64)*1000;pbase=np.asarray(base['points_world_m'],dtype=np.float64)*1000
refs=np.asarray(cur['area_reference_037_points_world_m_f64'],dtype=np.float64)*1000
ref_faces=np.asarray(cur['area_reference_037_source_O_faces'],dtype=np.int64)
if not np.array_equal(faces,ref_faces):raise ValueError('fixed037 face rows differ')
if len(pairs)!=1419 or len(faces)!=len(ref_faces):raise ValueError('unexpected 614 crossing/report shape')
face_rows=np.unique(pairs[:,2]);cross_nodes=np.unique(faces[face_rows].ravel())
anchors=np.asarray(cur['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
source_d_cur=np.linalg.norm(pcur-source_ref*1000,axis=1)
source_d_merge=np.linalg.norm(pmerge-merge_ref*1000,axis=1)
source_d_base=np.linalg.norm(pbase-base_ref*1000,axis=1)
area_cur=area_values(pcur,pbase,refs,faces,face_rows)
area_merge=area_values(pmerge,pbase,refs,faces,face_rows)
# Endpoint normal and remaining raw-target direction on the exact current crossing faces.
f=faces[face_rows]
cc=np.cross(pcur[f[:,1]]-pcur[f[:,0]],pcur[f[:,2]]-pcur[f[:,0]])
cn=cc/np.linalg.norm(cc,axis=1)[:,None]
rc=np.cross(refs[f[:,1]]-refs[f[:,0]],refs[f[:,2]]-refs[f[:,0]])
rn=rc/np.linalg.norm(rc,axis=1)[:,None]
rem=np.mean((pmerge-pcur)[f],axis=1)
# Match per-face bank identities and report incidence by bank plus node degree.
remaining_face_dir=np.einsum('ij,ij->i',rem,cn)
remaining_source_dir=np.einsum('ij,ij->i',rem,rn)
node_degree=np.bincount(faces[pairs[:,2]].ravel(),minlength=len(pcur))
all_rows_in_cross=np.unique(faces[pairs[:,2]].ravel())
anchor_faces=np.flatnonzero(np.isin(faces[face_rows],anchors).any(axis=1))
# Verify the candidate's embedded R0 source field agrees with 408's composed reference.
stored=np.asarray(cur.get('local_witness_cluster_222_source_reference_R0_m',source_ref),dtype=np.float64)
source_ref_error=np.linalg.norm(source_ref-stored,axis=1)
# all-face exact construction limits, arithmetic only
all_cur=area_values(pcur,pbase,refs,faces,np.arange(len(faces)))
all_merge=area_values(pmerge,pbase,refs,faces,np.arange(len(faces)))
all_base=area_values(pbase,pbase,refs,faces,np.arange(len(faces)))
all_rows=np.arange(len(faces))
def subset_extrema(v, rows):
    a=np.asarray(v);return {'min_row':int(rows[int(np.argmin(a))]),'min_value':float(np.min(a)),'max_row':int(rows[int(np.argmax(a))]),'max_value':float(np.max(a))}
area_detail={'retained_crossing_fixed037':subset_extrema(area_cur['fixed037_magnitude_ratio'],face_rows),'retained_crossing_current_orientation':subset_extrema(area_cur['current_oriented_ratio'],face_rows),'merge_crossing_fixed037':subset_extrema(area_merge['fixed037_magnitude_ratio'],face_rows),'merge_crossing_current_orientation':subset_extrema(area_merge['current_oriented_ratio'],face_rows)}
min_global_current=int(np.argmin(all_cur['current_oriented_ratio']));min_global_fixed=int(np.argmin(all_cur['fixed037_magnitude_ratio']));min_merge_fixed=int(np.argmin(all_merge['fixed037_magnitude_ratio']))
area_detail.update({'global_retained_current_orientation_row':min_global_current,'global_retained_current_orientation_is_crossing':bool(min_global_current in set(map(int,face_rows))),'global_retained_fixed037_row':min_global_fixed,'global_retained_fixed037_is_crossing':bool(min_global_fixed in set(map(int,face_rows))),'global_merge_fixed037_row':min_merge_fixed,'global_merge_fixed037_is_crossing':bool(min_merge_fixed in set(map(int,face_rows))),'retained_crossing_faces_fixed037_margin_lt_0_01_rows':face_rows[area_cur['fixed037_magnitude_ratio']<.11].tolist(),'merge_crossing_faces_fixed037_below_floor_rows':face_rows[area_merge['fixed037_magnitude_ratio']<.1].tolist()})
def extrema_rows(v, rows=all_rows):
    a=np.asarray(v);return {"min_row":int(rows[int(np.argmin(a))]),"max_row":int(rows[int(np.argmax(a))])}
current_resid=np.linalg.norm(pmerge-pcur,axis=1)
face_resid_max=np.linalg.norm((pmerge-pcur)[faces[face_rows]],axis=2).max(axis=1)
face_fully_at_merge=np.array([np.ascontiguousarray(pmerge[faces[r]]).tobytes()==np.ascontiguousarray(pcur[faces[r]]).tobytes() for r in face_rows])
pair_fully_at_merge=np.array([face_fully_at_merge[np.searchsorted(face_rows,int(r))] for r in pairs[:,2]])
summary={
 'schema':'numi.human.614-crossing-construction-limit-diagnosis.v1',
 'classification':'read-only arithmetic/set diagnosis from existing exact 159 witness rows; no geometry, fit, or exact audit run',
 'candidate':str(RET),'candidate_sha256':sha(RET),'state':str(STATE),'state_sha256':sha(STATE),
 'surface_report':str(SURF),'surface_report_sha256':sha(SURF),'merge_target':str(MERGE),'merge_target_sha256':sha(MERGE),
 'merge_report':str(MERGE_REPORT),'merge_report_sha256':sha(MERGE_REPORT),'baseline':str(BASE),'baseline_sha256':sha(BASE),
 'source_base':str(SOURCE),'source_base_sha256':sha(SOURCE),'source_reference_owner':str(OWNER),'source_reference_owner_sha256':sha(OWNER),
 'source_reference_composition':source_receipt,
 'source_reference_matches_embedded_R0_field':{'max_error_mm':float(source_ref_error.max(initial=0)*1000),'p99_error_mm':float(np.quantile(source_ref_error,.99)*1000)},
 'exact_159':{'pair_count':len(pairs),'unique_skin_face_count':len(face_rows),'unique_skin_node_count':len(all_rows_in_cross),
   'bank_count_nonzero':len(bank_counts),'nonzero_banks':sorted(bank_counts,key=lambda x:(-x['pair_count'],x['body'],x['stable_bone_id'])),
   'pair_rows_encoding':'[body, stable_bone_id, skin_face_row, bank_triangle_row] from the already complete report'},
 'source_cap_12mm':{
   'all_candidate_max_source_distance_mm':float(source_d_cur.max()),'all_candidate_min_slack_mm':float((12-source_d_cur).min()),
   'crossing_nodes':len(cross_nodes),'crossing_node_distance_mm':stats(source_d_cur[cross_nodes]),
   'crossing_node_slack_mm':stats(12-source_d_cur[cross_nodes]),
   'crossing_nodes_slack_lt_0_1mm':int(np.count_nonzero((12-source_d_cur[cross_nodes])<.1)),
   'crossing_nodes_slack_lt_0_5mm':int(np.count_nonzero((12-source_d_cur[cross_nodes])<.5)),
   'all_nodes_slack_lt_0_1mm':int(np.count_nonzero((12-source_d_cur)<.1)),
   'all_nodes_slack_lt_0_5mm':int(np.count_nonzero((12-source_d_cur)<.5)),
   'all_nodes_with_minimum_slack':int(np.argmin(12-source_d_cur)),
   'crossing_nodes_with_slack_lt_0_1mm':cross_nodes[(12-source_d_cur[cross_nodes])<.1].astype(int).tolist(),
   'all_nodes_with_slack_lt_0_1mm':np.flatnonzero((12-source_d_cur)<.1).astype(int).tolist(),
   'minimum_slack_node_is_crossing':bool(int(np.argmin(12-source_d_cur)) in set(map(int,cross_nodes))),
   'all_candidate_merge_max_source_distance_mm':float(source_d_merge.max()),
   'merge_target_nodes_over_12mm':int(np.count_nonzero(source_d_merge>12+1e-6)),
   'merge_target_crossing_node_distance_mm':stats(source_d_merge[cross_nodes]),
   'merge_target_crossing_nodes_over_12mm':int(np.count_nonzero(source_d_merge[cross_nodes]>12+1e-6))},
 'fixed037_and_orientation':{
   'all_candidate_fixed037_min_ratio':float(all_cur['fixed037_magnitude_ratio'].min()),
   'all_candidate_current_orientation_min_ratio_vs599':float(all_cur['current_oriented_ratio'].min()),
   'all_candidate_orientation_vs037_negative_count':int(np.count_nonzero(all_cur['orientation_vs037_ratio']<0)),
   'base599_orientation_vs037_negative_count':int(np.count_nonzero(all_base['orientation_vs037_ratio']<0)),
   'candidate_vs037_negative_faces_new_vs_base':int(np.count_nonzero((all_cur['orientation_vs037_ratio']<0)&(all_base['orientation_vs037_ratio']>=0))),
   'candidate_vs037_nonnegative_faces_lost_vs_base':int(np.count_nonzero((all_cur['orientation_vs037_ratio']>=0)&(all_base['orientation_vs037_ratio']<0))),
   'all_candidate_min_current_orientation_row':extrema_rows(all_cur['current_oriented_ratio']),
   'all_candidate_min_fixed037_magnitude_row':extrema_rows(all_cur['fixed037_magnitude_ratio']),
   'row_detail':area_detail,
   'merge_min_current_orientation_row':extrema_rows(all_merge['current_oriented_ratio']),
   'merge_min_fixed037_magnitude_row':extrema_rows(all_merge['fixed037_magnitude_ratio']),
   'crossing_face_rows':face_rows.tolist(),
   'crossing_faces_retained':area_summary(area_cur),'crossing_faces_unretained_merge_target':area_summary(area_merge),
   'crossing_faces_current_orient_margin_lt_0_05':int(np.count_nonzero(area_cur['current_oriented_ratio']<.15)),
   'crossing_faces_fixed037_margin_lt_0_05':int(np.count_nonzero(area_cur['fixed037_magnitude_ratio']<.15)),
   'crossing_faces_vs037_negative_retained':int(np.count_nonzero(area_cur['orientation_vs037_ratio']<0)),
   'crossing_faces_vs037_negative_base599':int(np.count_nonzero(area_values(pbase,pbase,refs,faces,face_rows)['orientation_vs037_ratio']<0)),
   'merge_all_faces_bad_current_orientation_floor_0_1':int(np.count_nonzero(all_merge['current_oriented_ratio']<.1)),
   'merge_all_faces_bad_fixed037_magnitude_floor_0_1':int(np.count_nonzero(all_merge['fixed037_magnitude_ratio']<.1))},
 'anchors':{'count':len(anchors),'crossing_faces_touching_anchor_count':len(anchor_faces),
   'crossing_node_anchor_count':int(np.intersect1d(cross_nodes,anchors).size),
   'anchor_coords_bitwise_equal_to_599_base':bool(cur['points_world_m'][anchors].dtype==base['points_world_m'][anchors].dtype and np.ascontiguousarray(cur['points_world_m'][anchors]).tobytes()==np.ascontiguousarray(base['points_world_m'][anchors]).tobytes()),
   'merge_anchor_coords_bitwise_equal_to_599_base':bool(merge['points_world_m'][anchors].dtype==base['points_world_m'][anchors].dtype and np.ascontiguousarray(merge['points_world_m'][anchors]).tobytes()==np.ascontiguousarray(base['points_world_m'][anchors]).tobytes())},
 'remaining_unretained_target_direction_on_same_crossing_faces':{
   'unique_face_mean_step_mm':stats(np.linalg.norm(rem,axis=1)),
   'mean_step_dot_retained_current_oriented_normal_mm':stats(remaining_face_dir),
   'mean_step_dot_fixed037_oriented_normal_mm':stats(remaining_source_dir),
   'faces_with_positive_current_normal_step':int(np.count_nonzero(remaining_face_dir>0)),
   'faces_with_negative_current_normal_step':int(np.count_nonzero(remaining_face_dir<0)),
   'faces_with_positive_037_normal_step':int(np.count_nonzero(remaining_source_dir>0)),
   'faces_with_negative_037_normal_step':int(np.count_nonzero(remaining_source_dir<0)),
   'retained_remaining_step_full_mesh_mm':stats(current_resid),
   'crossing_faces_already_bitwise_at_saved_merge_target':int(np.count_nonzero(face_fully_at_merge)),
   'crossing_faces_with_any_remaining_target_step':int(np.count_nonzero(~face_fully_at_merge)),
   'exact_159_pair_rows_already_bitwise_at_saved_merge_target':int(np.count_nonzero(pair_fully_at_merge)),
   'per_bank_remaining_face_rows':[{**b,'unique_crossing_faces_total':int(len(np.unique(pairs[(pairs[:,0]==b['body'])&(pairs[:,1]==b['stable_bone_id']),2]))),'unique_crossing_faces_with_residual':int(len(set(map(int,pairs[(pairs[:,0]==b['body'])&(pairs[:,1]==b['stable_bone_id']),2]))-set(map(int,face_rows[face_fully_at_merge]))))} for b in sorted(bank_counts,key=lambda x:(-x['pair_count'],x['body'],x['stable_bone_id']))],
   'note':'This is the residual displacement from the passing 614 retention endpoint toward the saved 019485 merge target; it is a direction diagnostic, not an endpoint certificate.'},
 'native_admission':False,'accepted_physical_steps':0}
OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'out':str(OUT),'sha256':sha(OUT),'crossing_faces':len(face_rows),'crossing_nodes':len(cross_nodes),'banks':bank_counts,'source':summary['source_cap_12mm'],'area':summary['fixed037_and_orientation'],'anchors':summary['anchors'],'remaining':summary['remaining_unretained_target_direction_on_same_crossing_faces']},indent=2))
