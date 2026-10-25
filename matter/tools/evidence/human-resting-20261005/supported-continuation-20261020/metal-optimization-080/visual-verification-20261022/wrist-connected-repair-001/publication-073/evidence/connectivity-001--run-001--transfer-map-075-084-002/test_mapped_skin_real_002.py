from __future__ import annotations
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
ROOT=Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded')
MAP=ROOT/'connectivity-001/run-001/transfer-map-075-084-002'
CAND=ROOT/'connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
EXPORT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/accepted-transfer-row-export-075/run-001/source-preflight')
DETAIL=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bilateral-hand-functional-rebind-020/run-001/candidate.nhskin')
ADAPTER=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-2025-10-05/none')
ADAPTER=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/E/accepted-transfer-row-adapter-075')
CANON=Path('/Users/n/Work/numi-human-connected-surface-069/canonical/matter/tools')
PINS={
 'transfer_rows':'71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14',
 'coarse_reference':'9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2',
 'coarse_faces':'7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2',
 'detail_payload':'7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 sys.path.insert(0,str(ADAPTER)); sys.path.insert(0,str(CANON))
 from load_transfer_075 import load_transfer075
 from human_skin_fat_embedded_surface import MappedSkinSurface
 W=load_npz(MAP/'Wnew-detail-to-084-nodes.npz').tocsr()
 side=np.load(MAP/'transfer-sidecar.npz',allow_pickle=False)
 detail=np.asarray(side['detail_origin_f32'],dtype=np.float32)
 X=np.asarray(side['mechanical_origin_f32'],dtype=np.float32)
 W0,d0,c0,F=load_transfer075(EXPORT/'accepted-transfer-rows.csv',EXPORT/'accepted-coarse-reference.csv',EXPORT/'accepted-coarse-faces.csv',DETAIL,expected_sha256=PINS)
 active=np.asarray(side['active_mask'],dtype=bool)
 if W.shape!=(len(detail),len(X)) or detail.shape[1]!=3 or not np.array_equal(active,np.diff(W.indptr)>0):
  raise RuntimeError('real Wnew shape/active row mismatch')
 surf=MappedSkinSurface(X,detail,F,W)
 baseline=surf.positions(X)
 if not np.array_equal(baseline,detail): raise RuntimeError('baseline detailed origin is not bitwise preserved')
 # Translation check through the actual canonical owner.
 trans=np.array([1.25e-3,-.75e-3,.5e-3],dtype=np.float32)
 Xt=(X+trans).astype(np.float32)
 Yt=surf.positions(Xt)
 dx=Xt.astype(np.float64)-X.astype(np.float64)
 expected=(detail.astype(np.float64)+W@dx).astype(np.float32)
 if not np.array_equal(Yt,expected): raise RuntimeError('translation output differs from canonical additive W formula')
 disp_t=Yt.astype(np.float64)-detail.astype(np.float64)
 active_dot=disp_t[active]@trans.astype(np.float64)
 if np.any(active_dot<=0): raise RuntimeError('active rows did not respond in translation direction')
 if np.any(Yt[~active]!=detail[~active]): raise RuntimeError('inactive detailed rows moved under translation')
 # Spatially varying affine displacement, to ensure the map uses all 7360 columns.
 A=np.array([[.012,-.004,.002],[.003,-.009,.001],[-.002,.005,.007]],dtype=np.float64)
 b=np.array([.35e-3,-.20e-3,.15e-3],dtype=np.float64)
 d_aff=(X.astype(np.float64)@A.T+b).astype(np.float32)
 Xa=(X+d_aff).astype(np.float32)
 da=Xa.astype(np.float64)-X.astype(np.float64)
 Ya=surf.positions(Xa)
 expected_a=(detail.astype(np.float64)+W@da).astype(np.float32)
 if not np.array_equal(Ya,expected_a): raise RuntimeError('affine output differs from canonical additive W formula')
 if np.any(Ya[~active]!=detail[~active]): raise RuntimeError('inactive detailed rows moved under affine displacement')
 # Work-adjoint identity for deterministic arbitrary detail forces and FEM displacement.
 rng=np.random.default_rng(86002)
 f=rng.standard_normal((len(detail),3)); u=rng.standard_normal((len(X),3))
 lhs=float(np.sum(f*(W@u))); rhs=float(np.sum((W.T@f)*u))
 adj=abs(lhs-rhs)
 if adj>1e-9*max(1.,abs(lhs),abs(rhs)): raise RuntimeError(f'adjoint identity failed: {adj}')
 report={
  'schema':'real_075_084_mapped_skin_owner_smoke_v1','status':'pass_static_additive_owner_smoke',
  'inputs_sha256':{'Wnew':sha(MAP/'Wnew-detail-to-084-nodes.npz'),'sidecar':sha(MAP/'transfer-sidecar.npz'),'transfer_rows':PINS['transfer_rows'],'detail_payload':PINS['detail_payload'],'embedded_surface_source':sha(CANON/'human_skin_fat_embedded_surface.py')},
  'dimensions':{'detail_rows':int(len(detail)),'mechanical_nodes':int(len(X)),'W_nnz':int(W.nnz),'active_rows':int(active.sum()),'inactive_rows':int((~active).sum()),'detail_faces':int(len(F))},
  'baseline':{'bitwise_detail_origin_preserved':True,'max_abs_delta_m':float(np.max(np.abs(baseline-detail)))},
  'translation':{'vector_m':trans.tolist(),'active_rows_positive_dot_count':int(np.count_nonzero(active_dot>0)),'active_row_count':int(active.sum()),'min_active_dot_m2':float(active_dot.min()),'inactive_rows_bitwise_fixed':bool(np.array_equal(Yt[~active],detail[~active])),'matches_canonical_formula_bitwise':True},
  'affine_displacement':{'matrix':A.tolist(),'offset_m':b.tolist(),'all_rows_match_canonical_formula_bitwise':True,'inactive_rows_bitwise_fixed':bool(np.array_equal(Ya[~active],detail[~active])),'max_abs_output_displacement_m':float(np.max(np.abs(Ya.astype(np.float64)-detail.astype(np.float64))))},
  'adjoint':{'lhs':lhs,'rhs':rhs,'absolute_error':adj,'relative_error':adj/max(1.,abs(lhs),abs(rhs)),'seed':86002},
  'limits':['CPU canonical MappedSkinSurface owner only; native Metal shader parity is unverified.','This verifies map application semantics, not mapped-surface self intersection, connected clearance, cell gates, or physical/native admission.']}
 out=MAP/'canonical-owner-smoke-001';out.mkdir(exist_ok=True)
 report['test_script_sha256']=sha(Path(__file__))
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__': main()
