from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy import sparse
D=Path('/Users/n/Work/numi-human-connected-surface-069/direct-authoring-080')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
TOOLS=D/'canonical/matter/tools';sys.path.insert(0,str(TOOLS))
import human_skin_fat_connected_repair as repair
from human_skin_fat_embedded_surface import MappedSkinSurface
AD=E/'E/accepted-transfer-row-adapter-075';sys.path.insert(0,str(AD))
from load_transfer_075 import load_transfer075
I=D/'repair-run-embedded-080/iteration-001'
IN=I/'input.npz';CAND=I/'candidate.npz'
z=np.load(IN);X=z['points_world_f32'];R=z['reference_points_m'];T=z['tetrahedra'];F=z['free_nodes'];O=z['construction_origin_f32']
yc=np.load(CAND)['points_world_f32']
assert X.dtype==np.float32 and yc.dtype==np.float32 and X.shape==yc.shape and len(F)==6243
direction_m=(yc[F].astype(np.float64)-X[F].astype(np.float64))*32.
direction_mm=direction_m*1000.
TRANSFER=E/'accepted-transfer-row-export-075/run-001/source-preflight'
TR=TRANSFER/'accepted-transfer-rows.csv';CR=TRANSFER/'accepted-coarse-reference.csv';CF=TRANSFER/'accepted-coarse-faces.csv'
DP=E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'
W,dorg,corig,dfaces=load_transfer075(TR,CR,CF,DP,expected_sha256={
'transfer_rows':'71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14',
'coarse_reference':'9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2',
'coarse_faces':'7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2',
'detail_payload':'7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'})
W=sparse.hstack([W,sparse.csr_matrix((W.shape[0],len(X)-W.shape[1]))],format='csr')
embedded=MappedSkinSurface(O,dorg,dfaces,W)
BF=np.load(D/'repair-run-embedded-080/boundary-faces.npy')
L=sparse.load_npz(D/'repair-run-embedded-080/layer-rows.npz')
lt=np.load(D/'repair-run-embedded-080/layer-targets.npy')
C=sparse.load_npz(I/'contact-rows.npz')
ct=z['contact_targets']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
dGC=repair.sparse_displacement_gradient(X,T,F)
dGR=repair.sparse_displacement_gradient(R,T,F)
gcur=np.asarray(dGC@direction_mm.ravel()).ravel()
gref=np.asarray(dGR@direction_mm.ravel()).ravel()
gradient={'current_max_abs':float(np.max(np.abs(gcur))),'reference_max_abs':float(np.max(np.abs(gref))),
          'current_rows_over_0_12':int(np.count_nonzero(np.abs(gcur)>.12)),
          'reference_rows_over_0_12':int(np.count_nonzero(np.abs(gref)>.12)),
          'unit':'dimensionless gradient from reconstructed displacement in mm'}
fractions=[1.,.5,.25,.125,.0625,.03125]
rows=[]
for alpha in fractions:
 Y=X.copy()
 Y[F]=(X[F].astype(np.float64)+alpha*direction_m).astype(np.float32)
 J,_,q,_=repair.cell_values_gradients(Y,R,T)
 layer=np.asarray(L@Y.astype(float).ravel()-lt).ravel()
 _,_,ba,bc=repair.boundary_face_values_gradients(Y,O,BF)
 _,_,da,dc=embedded.metrics(Y)
 deficit=float(np.maximum(ct-C@Y.astype(float).ravel(),0).sum()*1000)
 cell_qbad=np.flatnonzero(q<1e-5); cell_low=np.flatnonzero(J<.35);cell_high=np.flatnonzero(J>2.5)
 layerbad=np.flatnonzero(layer < -1e-9)
 ba_lo=np.flatnonzero(ba<.5);ba_hi=np.flatnonzero(ba>2.);bc_bad=np.flatnonzero(bc<.5)
 da_lo=np.flatnonzero(da<.5);da_hi=np.flatnonzero(da>2.);dc_bad=np.flatnonzero(dc<.5)
 gates=[
  ('q',len(cell_qbad)==0,{'count':len(cell_qbad),'min':float(q.min()),'row':int(np.argmin(q))}),
  ('J',len(cell_low)==0 and len(cell_high)==0,{'low_count':len(cell_low),'high_count':len(cell_high),'min':float(J.min()),'min_row':int(np.argmin(J)),'max':float(J.max()),'max_row':int(np.argmax(J))}),
  ('layer',len(layerbad)==0,{'count':len(layerbad),'min_margin':float(layer.min()),'row':int(np.argmin(layer))}),
  ('coarse_area',len(ba_lo)==0 and len(ba_hi)==0,{'below_count':len(ba_lo),'above_count':len(ba_hi),'min':float(ba.min()),'min_row':int(np.argmin(ba)),'max':float(ba.max()),'max_row':int(np.argmax(ba))}),
  ('coarse_normal',len(bc_bad)==0,{'count':len(bc_bad),'min_cosine':float(bc.min()),'row':int(np.argmin(bc))}),
  ('detail_area',len(da_lo)==0 and len(da_hi)==0,{'below_count':len(da_lo),'above_count':len(da_hi),'min':float(da.min()),'min_row':int(np.argmin(da)),'max':float(da.max()),'max_row':int(np.argmax(da))}),
  ('detail_normal',len(dc_bad)==0,{'count':len(dc_bad),'min_cosine':float(dc.min()),'row':int(np.argmin(dc))}),
 ]
 first=next((name for name,ok,_ in gates if not ok),None)
 rows.append({'fraction':alpha,'q_min':float(q.min()),'J_min':float(J.min()),'J_max':float(J.max()),
              'layer_min_margin':float(layer.min()),'coarse_area_min':float(ba.min()),'coarse_area_max':float(ba.max()),
              'coarse_normal_min':float(bc.min()),'detail_area_min':float(da.min()),'detail_area_max':float(da.max()),
              'detail_normal_min':float(dc.min()),'contact_deficit_l1_mm':deficit,
              'first_failed_gate':first,'gates':{name:{'pass':bool(ok),**values} for name,ok,values in gates}})
report={'kind':'approximate line-direction gate replay from accepted Float32 endpoint; NOT exact LP direction',
'method':'direction_m = 32 * (accepted_candidate_F32 - original037_F32) at the saved free nodes; reconstruct each tested point by Float32 emission from original baseline plus alpha*direction',
'no_lp_solve':True,'candidate_sha256':sha(CAND),'input_sha256':sha(IN),
'solver_sha256':sha(TOOLS/'human_skin_fat_connected_repair.py'),'free_nodes':len(F),
'accepted_fraction':.03125,'direction_reconstruction_warning':'F32 rounding at the accepted endpoint is amplified by 32; larger-fraction results are approximate diagnostics and cannot identify the exact LP backtracking predicate.',
'gradient_diagnostic':gradient,'fractions':rows}
out=D/'diagnosis-rounded-direction-001.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'output':str(out),'sha256':sha(out),'gradient':gradient,'fractions':[{'fraction':r['fraction'],'first_failed_gate':r['first_failed_gate'],'qmin':r['q_min'],'J':[r['J_min'],r['J_max']],'layer':r['layer_min_margin'],'boundary':[r['coarse_area_min'],r['coarse_area_max'],r['coarse_normal_min']],'detail':[r['detail_area_min'],r['detail_area_max'],r['detail_normal_min']]} for r in rows]},indent=2))

