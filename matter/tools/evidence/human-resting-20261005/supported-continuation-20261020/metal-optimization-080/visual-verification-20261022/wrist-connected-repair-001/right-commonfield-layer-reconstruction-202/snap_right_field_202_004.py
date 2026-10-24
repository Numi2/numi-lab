from pathlib import Path
import importlib.util,json,numpy as np,hashlib
from scipy.optimize import least_squares
B=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/right-commonfield-layer-reconstruction-202');S=B/'attempt-003';O=B/'attempt-004';O.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('geom',B/'tools/human_skin_patch_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
z=np.load(S/'right-commonfield-calibrated-202.npz');X=z['points_world_f32_m'].astype(float);T=z['tetrahedra'];vol=z['cell_volume_m3'];base=z['phi'];fixed=set(np.unique(z['outer_faces']).tolist()+z['I_cut_ids'].tolist()+z['D_cut_ids'].tolist());groups=[];selection={}
for role in ['I','D']:
 s=np.load(S/f'right-{role}-interface-202.npz');k=s['source_volume_edge_nodes'];w=s['source_edge_endpoint_one_weight'];dist=np.linalg.norm(X[k[:,1]]-X[k[:,0]],axis=1);near=(k[:,0]!=k[:,1])&(np.minimum(w,1-w)*dist<5e-6);candidate=np.unique(np.where(w[near]<.5,k[near,0],k[near,1]));chosen=[];diag=[]
 for i in candidate:
  if int(i) in fixed:continue
  all_dist=np.r_[w[k[:,0]==i]*dist[k[:,0]==i],(1-w[k[:,1]==i])*dist[k[:,1]==i]];maximum=float(all_dist.max());accept=maximum<=50e-6
  diag.append({'node':int(i),'maximum_affected_edge_cut_distance_m':maximum,'selected':accept})
  if accept:chosen.append(int(i))
 groups.append(np.array(chosen,np.int64));selection[role]=diag
if np.intersect1d(*groups).size:raise ValueError('overlapping I/D snap sets')
targets=np.array([.0003491,.0015436])
def field(l):
 p=base.copy();p[np.r_[z['I_cut_ids'],groups[0]]]=l[0];p[np.r_[z['D_cut_ids'],groups[1]]]=l[1];return p
def residual(l):
 if l[0]>=l[1]:return np.array([10.,10.])
 p=field(l);return np.array([np.dot(vol,g.p1_tetrahedral_sublevel_fractions(p,T,t)) for t in l])/targets-1
r=least_squares(residual,z['levels'],bounds=([1e-6,2e-6],[1-2e-6,1-1e-6]),diff_step=1e-5,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100);p=field(r.x);plateaus=[int(np.all(p[T]==t,axis=1).sum()) for t in r.x];passed=bool(r.success and max(abs(residual(r.x)))<1e-6 and not any(plateaus))
kwargs={k:z[k] for k in z.files};kwargs.update(phi=p,levels=r.x,sublevel_fractions=np.stack([g.p1_tetrahedral_sublevel_fractions(p,T,t) for t in r.x]));np.savez_compressed(O/'right-commonfield-calibrated-202.npz',**kwargs)
report={'physical_admission':False,'method':'collapse near-endpoint contour features by equal field knots, retaining exact volume/source nodes; only nodes whose entire incident old contour cuts lie within 50um are eligible','endpoint_trigger_m':5e-6,'maximum_star_cut_displacement_before_recalibration_m':50e-6,'selection':selection,'source_sha256':hashlib.sha256((S/'right-commonfield-calibrated-202.npz').read_bytes()).hexdigest(),'levels_before':z['levels'].tolist(),'levels_after':r.x.tolist(),'relative_target_error':residual(r.x).tolist(),'constant_threshold_cells':plateaus,'volume_calibration_pass':passed,'remaining':['measure actual emitted interface displacement','exact interface and bone tests','Float32 material volume quality','conservative physical ancestry and native mechanics']};(O/'field-refinement-report-202.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if not passed:raise SystemExit('failed calibration')
