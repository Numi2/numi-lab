from pathlib import Path
import importlib.util, json, os, sys, hashlib
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069')
RUN=D/'D/finite-triangle-fit-511/all-pairs-001-launch/argv.json'
OUT=D/'D/D/absolute-deficit-surface-weighting-520/matched-all-pairs-row-preflight.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def flagval(argv,flag):return Path(argv[argv.index(flag)+1])
def stats(a):
    x=np.asarray(a,dtype=np.float64)
    q=np.quantile(x,[0,.01,.05,.25,.5,.75,.95,.99,1])
    return {'count':int(len(x)),'min':float(q[0]),'p01':float(q[1]),'p05':float(q[2]),'p25':float(q[3]),'median':float(q[4]),'p75':float(q[5]),'p95':float(q[6]),'p99':float(q[7]),'max':float(q[8]),'sum':float(x.sum())}
argv=json.loads(RUN.read_text())
base=flagval(argv,'--base-candidate'); audit=flagval(argv,'--audit159-490')
point_npz=flagval(argv,'--point-target-npz'); point_report=flagval(argv,'--point-target-report')
face_npz=flagval(argv,'--face-target-npz'); face_report=flagval(argv,'--face-target-report')
runner=load('fit458_preflight520',D/'D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py')
os.environ['NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458']=sha(base)
os.environ['NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458']=sha(audit)
cross=runner.prepare_crossing_inputs(base,audit)
owner=load('fitter408_preflight520',runner.OWNER)
objective=load('objective458_preflight520',runner.HELPER)
sat=load('triangle_sat_rows_466_preflight520',D/'D/D/triangle-sat-rows-466/triangle_sat_rows_466.py')
helper=load('anchor_finite_triangle_513_preflight520',D/'D/D/anchor-compatible-finite-triangle-513/anchor_finite_triangle_513.py')
with np.load(base,allow_pickle=False) as z:
    points_f32=np.asarray(z['points_world_m'],dtype=np.float32)
    anchors=np.asarray(z['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
point_targets,face_targets=owner.load_clearance_target_inputs(point_npz,point_report,face_npz,face_report,base,points_f32,cross['faces'])
support=np.unique(np.concatenate((point_targets['point_rows'],face_targets['face_nodes'].ravel()))).astype(np.int64)
points_mm=points_f32.astype(np.float64)*1000.0
bary=owner.load_module('barycentric_owner_preflight520',owner.DEFAULT_BARY_OWNER)
selected={(int(p['face_row']),int(p['body']),int(p['bone_id']),int(p['triangle_id'])) for p in cross['pairs']}
result=helper.build_anchor_compatible_rows(points_mm,cross['faces'],support,anchors,cross['pairs'],cross['bone_triangles_mm'],sat=sat,selected_pair_ids=selected,total_weight=(.5 if len(point_targets['point_rows']) else 1.0))
proxy=type('Objective520',(),{'build_fixed_bone_outward_plane_rows':staticmethod(lambda *a,**kw:result)})
state=runner.install_objective(owner,proxy,cross)
C0,h0,w0,meta0=owner.build_clearance_rows(points_mm,cross['faces'],support,anchors,point_targets,face_targets,bary)
pn=len(point_targets['point_rows']);fh=np.asarray(h0[pn:],dtype=np.float64);fw=np.asarray(w0[pn:],dtype=np.float64);ph=np.asarray(h0[:pn],dtype=np.float64);pw=np.asarray(w0[:pn],dtype=np.float64)
weighted=load('absolute_deficit_weighting_preflight520',D/'D/D/absolute-deficit-surface-weighting-520/absolute_deficit_weighting_520.py')
face={"C":C0[pn:].copy(),"h":fh.copy(),"weights":fw.copy(),"metadata":[dict(x) for x in meta0[pn:]]}
face2=weighted.reweight_surface_rows(face,total_weight=float(fw.sum()))
w1=np.concatenate((pw,face2['weights']))
# The new strategy is exactly weight-only after rows have been assembled.
assert np.array_equal(h0,np.concatenate((ph,fh))) and len(meta0)==len(h0)
assert np.array_equal(w1[:pn],pw)
assert np.array_equal(face2['h'],fh) and np.array_equal(face2['C'].data,C0[pn:].data)
assert np.allclose(face2['weights']/fh,np.full(len(fh),float(fw.sum())/fh.sum()),rtol=2e-14,atol=0)
byrow={int(m['query_row']):{'h_mm':float(ph[i]),'weight':float(pw[i]),'weight_over_h_per_mm':float(pw[i]/ph[i])} for i,m in enumerate(meta0[:pn])}
receipt={
 'schema':'numi.absolute_deficit_surface_weights_520.matched_row_preflight.v1',
 'candidate_path':str(base),'candidate_sha256':sha(base),'audit159_path':str(audit),'audit159_sha256':sha(audit),
 'point_target_npz_sha256':sha(point_npz),'face_target_npz_sha256':sha(face_npz),
 'pair_count':len(cross['pairs']),'selected_sat_pair_count':len(selected),'support_node_count':len(support),
 'point_row_count':pn,'surface_row_count':len(fh),'point_family_weight_mass':float(pw.sum()),'surface_family_weight_mass':float(fw.sum()),
 'point_h_mm':stats(ph),'point_weight':stats(pw),'point_weight_over_h_per_mm':stats(pw/ph),
 'surface_h_mm':stats(fh),'surface_original_weight':stats(fw),'surface_original_weight_over_h_per_mm':stats(fw/fh),
 'surface_reweighted_weight':stats(face2['weights']),'surface_reweighted_weight_over_h_per_mm':stats(face2['weights']/fh),
 'selected_point_rows':{str(k):v for k,v in byrow.items() if k in (19226,19415)},
 'same_point_prefix_weights':True,'same_surface_C_and_h':True,'same_hard_rows_guards_bounds_and_certification':True,
 'objective_changed_from_uniform_normalized_slack':True,'global_surface_clearance_certified':False,
 '458_objective_state':state['objective'],'row_metadata_sha256':hashlib.sha256(json.dumps(meta0,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
 'helper520_sha256':sha(D/'D/D/absolute-deficit-surface-weighting-520/absolute_deficit_weighting_520.py'),
 'wrapper520_sha256':sha(D/'D/D/absolute-deficit-surface-weighting-520/run_absolute_deficit_520.py')}
OUT.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:receipt[k] for k in ('pair_count','point_row_count','surface_row_count','point_family_weight_mass','surface_family_weight_mass','surface_h_mm','surface_original_weight_over_h_per_mm','surface_reweighted_weight_over_h_per_mm','selected_point_rows','global_surface_clearance_certified')},indent=2))
print('receipt',OUT,'sha256',sha(OUT))
