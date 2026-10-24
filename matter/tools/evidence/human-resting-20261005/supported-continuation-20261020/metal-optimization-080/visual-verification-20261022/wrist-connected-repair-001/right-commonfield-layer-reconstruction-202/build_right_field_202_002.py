#!/usr/bin/env python3
"""Prepare one common P1 field on the already qualified right PLC mesh.
Construction only: independently reconstructed boundaries and native admission
remain mandatory. Physical skin distance excludes every construction cap.
"""
from pathlib import Path
import ast, hashlib, importlib.util, json, time, sys
import numpy as np
from scipy.optimize import least_squares
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
A=E/'ankle-parent-tet-refinement-026'; BASE=E/'right-commonfield-layer-reconstruction-202'; OUT=BASE/'attempt-002'; OUT.mkdir(exist_ok=False)
SOURCE=A/'generated-225-actual007-cut-stitch-domain-preflight/local-quality-remesh-right-225-o-face-barycenter-subdivision/right-tetgen-local-quality-repaired.npz'
DOMAIN=A/'generated-225-actual007-cut-stitch-domain-preflight/right-closed-domain.npz'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_functions(p, ns):
 tree=ast.parse(Path(p).read_text()); keep=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
 exec(compile(ast.Module(body=keep,type_ignores=[]),str(p),'exec'),ns)
ns={'__name__':'frozen_common_field_geometry','MAX_TET_VOL':1.5e-7}
load_functions(A/'build_common_volume_126.py',ns)
ns.update(BANK=E/'registered-surrounding-tissue-native007-190/registered-tissue-banks.npz',INV=E/'registered-surrounding-tissue-native007-190/inventory.json',POSE=E/'native-app-main-header-integration-151/source-preflight-007/registered-skin-poses.json')
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
load_functions(A/'generated-224-commonfield-tetgen-volume-actual007/build_commonfield_224.py',ns)
ns.update(sha=sha)
spec=importlib.util.spec_from_file_location('geom',BASE/'tools/human_skin_patch_geometry.py');geom=importlib.util.module_from_spec(spec);spec.loader.exec_module(geom)
start=time.monotonic();z=np.load(SOURCE,allow_pickle=False); d=np.load(DOMAIN,allow_pickle=False)
X=z['points_world_f64_m'].astype(np.float32).astype(np.float64);T=z['tetrahedra'].astype(np.int64)
vol,q=ns['tet_metrics'](X,T)
if np.any(vol<=0) or np.any(q<1e-5): raise ValueError('source volume no longer passes Float32 quality')
B=z['original_boundary_faces'];M=z['original_boundary_markers'];O=B[M==1]
banks=ns['classify_banks']('right',X[O].reshape(-1,3))
phi,detail=ns['build_bone_field'](X,X,O,banks)
# Override exact inherited cut rings only; never move their coordinates.
Oids=np.unique(O); Iids=d['stitch_I_local_ids']; Dids=d['stitch_D_local_ids']; center=d['D_cap_center_local_id']
if np.intersect1d(Oids,np.r_[Iids,Dids]).size or np.intersect1d(Iids,Dids).size: raise ValueError('aliased cut boundaries')
phi[Oids]=0; phi[center]=1.0  # construction-only center is beyond D, never a D interface vertex
np.savez_compressed(OUT/'right-commonfield-raw-202.npz',points_world_f32_m=X.astype(np.float32),tetrahedra=T,cell_volume_m3=vol,normalized_determinant=q,phi_before_ring_calibration=phi,outer_faces=O,construction_faces=B[M!=1],I_cut_ids=Iids,D_cut_ids=Dids)
target=np.array([.0003491,.0003491+.0011945])
def field(levels):
 f=phi.copy(); f[Iids]=levels[0]; f[Dids]=levels[1]; return f
calls=0
def residual(levels):
 global calls
 calls+=1
 if levels[0]>=levels[1]: return np.array([10+levels[0]-levels[1],10+levels[0]-levels[1]])
 f=field(levels)
 actual=np.array([np.dot(vol,geom.p1_tetrahedral_sublevel_fractions(f,T,t)) for t in levels])
 return (actual-target)/target
result=least_squares(residual,[.08,.45],bounds=([1e-6,2e-6],[1-2e-6,1-1e-6]),diff_step=1e-5,xtol=1e-12,gtol=1e-12,ftol=1e-12,max_nfev=100)
levels=result.x; finalphi=field(levels); fractions=np.stack([geom.p1_tetrahedral_sublevel_fractions(finalphi,T,t) for t in levels]); actual=fractions@vol
plateaus=[int(np.count_nonzero(np.all(finalphi[T]==t,axis=1))) for t in levels]
passed=bool(result.success and np.max(np.abs((actual-target)/target))<1e-6 and not any(plateaus) and levels[0]<levels[1])
np.savez_compressed(OUT/'right-commonfield-calibrated-202.npz',phi=finalphi,levels=levels,sublevel_fractions=fractions,cell_volume_m3=vol,tetrahedra=T,points_world_f32_m=X.astype(np.float32),outer_faces=O,construction_faces=B[M!=1],I_cut_ids=Iids,D_cut_ids=Dids)
clean={k:v for k,v in detail.items() if not isinstance(v,np.ndarray)}
report={'source_mesh':str(SOURCE),'source_mesh_sha256':sha(SOURCE),'closed_domain_sha256':sha(DOMAIN),'script_sha256':sha(__file__),'geometry_helper_sha256':sha(BASE/'tools/human_skin_patch_geometry.py'),'frame':'source-preflight-007 actual registered Float32 world','physical_O_faces':len(O),'construction_faces_excluded_from_outer_distance':int((M!=1).sum()),'nodes':len(X),'tetrahedra':len(T),'minimum_normalized_determinant':float(q.min()),'total_domain_volume_m3':float(vol.sum()),'targets_m3':target.tolist(),'targets_provenance':'retained 126/224 reference construction targets; not measured regional anatomy or validated thickness','levels':levels.tolist(),'actual_sublevel_volume_m3':actual.tolist(),'relative_target_error':((actual-target)/target).tolist(),'positive_volume_constant_threshold_cells':plateaus,'volume_calibration_pass':passed,'optimizer_success':bool(result.success),'residual_evaluations':calls,'field_diagnostics':clean,'wall_seconds':time.monotonic()-start,'physical_admission':False,'remaining':['extract independent interfaces including exact inherited cut rings','exact intersection/orientation/clearance checks','ancestry/material mass/transfer mapping','native contact ownership and physical acceptance']}
(OUT/'right-commonfield-report-202.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,sort_keys=True),flush=True)
if not passed:raise SystemExit('calibration failed')
