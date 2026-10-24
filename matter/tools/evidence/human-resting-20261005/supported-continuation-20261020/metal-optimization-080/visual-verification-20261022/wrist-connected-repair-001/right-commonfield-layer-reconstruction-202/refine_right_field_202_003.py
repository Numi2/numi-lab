from pathlib import Path
import importlib.util,json,numpy as np,hashlib
from scipy.optimize import least_squares
B=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/right-commonfield-layer-reconstruction-202');S=B/'attempt-002';O=B/'attempt-003';O.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('geom',B/'tools/human_skin_patch_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
z=np.load(S/'right-commonfield-calibrated-202.npz'); surf=np.load(S/'right-D-interface-202.npz');audit=json.loads((S/'interface-bone-clearance-202.json').read_text());hits=audit['roles']['D']['witnesses'];faces=sorted(set(h['surface_face'] for h in hits));nodes=np.unique(surf['source_volume_edge_nodes'][surf['faces'][faces]])
fixed=np.unique(np.r_[np.unique(z['outer_faces']),z['I_cut_ids'],z['D_cut_ids']]);free=np.setdiff1d(nodes,fixed)
if not len(free): raise ValueError('all witness-support vertices are fixed; requires local volume refinement')
base=z['phi'].copy();old=base[free].copy();base[free]=1.
T=z['tetrahedra'];vol=z['cell_volume_m3'];targets=np.array([.0003491,.0015436]);I=z['I_cut_ids'];D=z['D_cut_ids']
def field(l):
 phi=base.copy();phi[I]=l[0];phi[D]=l[1];return phi
def residual(l):
 if l[0]>=l[1]:return np.array([10.,10.])
 phi=field(l); return np.array([np.dot(vol,g.p1_tetrahedral_sublevel_fractions(phi,T,t)) for t in l])/targets-1
r=least_squares(residual,z['levels'],bounds=([1e-6,2e-6],[1-2e-6,1-1e-6]),diff_step=1e-5,xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100);phi=field(r.x)
plateaus=[int(np.count_nonzero(np.all(phi[T]==t,axis=1))) for t in r.x]
passed=bool(r.success and max(abs(residual(r.x)))<1e-6 and not any(plateaus))
kwargs={k:z[k] for k in z.files};kwargs.update(phi=phi,levels=r.x,sublevel_fractions=np.stack([g.p1_tetrahedral_sublevel_fractions(phi,T,t) for t in r.x]));np.savez_compressed(O/'right-commonfield-calibrated-202.npz',**kwargs)
report={'physical_admission':False,'source_calibrated_sha256':hashlib.sha256((S/'right-commonfield-calibrated-202.npz').read_bytes()).hexdigest(),'exact_bone_witnesses':len(hits),'witness_interface_faces':faces,'adjusted_free_volume_nodes':free.tolist(),'before_nodal_field':old.tolist(),'after_nodal_field':base[free].tolist(),'method':'raise only free support nodes of exact D/bone witnesses to the existing bone-exclusion field endpoint; preserve every O and inherited cut node; recalibrate original construction material-volume targets','levels_before':z['levels'].tolist(),'levels_after':r.x.tolist(),'target_relative_error':residual(r.x).tolist(),'constant_threshold_cells':plateaus,'volume_calibration_pass':passed}
(O/'field-refinement-report-202.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
if not passed:raise SystemExit('failed calibration')
