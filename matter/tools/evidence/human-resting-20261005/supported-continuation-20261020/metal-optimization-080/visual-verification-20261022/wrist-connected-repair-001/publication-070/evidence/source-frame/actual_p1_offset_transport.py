#!/usr/bin/env python3
"""Bounded AP->039 P1 offset-transport diagnostic on exact D36->294 anchors.
No solve, mesh rewrite, NHSKIN edit, or physical qualification is performed.
"""
from __future__ import annotations
import csv, hashlib, json, socket, sys, time
from pathlib import Path
import numpy as np

E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=E/"wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001"
DETAIL=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001")
AP=E/"bone-clearance-feasibility-033/world-bone-clearance-constraints.npz"
REPAIR=E/"connected-volume-continuation-039/iteration-005/repair-input.npz"
CAND=E/"connected-volume-continuation-039/iteration-005/candidate.npz"
EMBED=D/"generated/embedding.csv"; NODES=D/"generated/nodes.csv"; FACES=D/"generated/faces.csv"
CURRENT=DETAIL/"native-step-001/registered-skin.csv"
FRAME=E/"elbow-refinement-current-main-024-final080-candidate-source-frames-001/native-source-only-024/common-rigid-transfer-row-diagnostic-005.json"
PROLONG=E/"current-294-skin-displacement-prolongation-001/run-002/prolongation.npz"
SOURCE=DETAIL/"bodyparts3d-myosim-skinned-shell.nhskin"

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def xyz_csv(rs,prefix='w'):
 return np.array([[float(r[prefix+c]) for c in 'xyz'] for r in rs],float)
def q(x):
 x=np.asarray(x,float)
 return {"count":int(x.size),"min":float(np.min(x)),"median":float(np.median(x)),"p95":float(np.quantile(x,.95)),"p99":float(np.quantile(x,.99)),"max":float(np.max(x))}
def run(out:Path):
 start=time.time();out.mkdir(parents=True,exist_ok=False)
 inputs=[AP,REPAIR,CAND,EMBED,NODES,FACES,CURRENT,FRAME,PROLONG,SOURCE]
 for p in inputs:
  if not p.is_file():raise FileNotFoundError(p)
 ap=np.load(AP);ri=np.load(REPAIR);cand=np.load(CAND);old=np.load(PROLONG)
 x0=np.asarray(ap['points'],float); xref=np.asarray(ri['reference_points_m'],float);x1=np.asarray(cand['points_world_f32'],np.float64)
 tet=np.asarray(ap['tetrahedra'],np.int64);mat=np.asarray(ap['material_ids'],np.int8)
 if x0.shape!=(7233,3) or xref.shape!=x0.shape or x1.shape!=x0.shape or tet.shape!=(28632,4):raise ValueError('pinned physical grid shape mismatch')
 if np.max(np.linalg.norm(xref-x0,axis=1))>1e-6:raise ValueError('repair baseline differs from AP basis')
 emb=rows(EMBED);node_rows=rows(NODES);face_rows=rows(FACES);cur_rows=rows(CURRENT)
 if len(emb)!=2411 or len(node_rows)!=2411 or len(face_rows)!=4777 or len(cur_rows)!=61950:raise ValueError('pinned ancestry/surface row count mismatch')
 faces=np.array([[int(r[k]) for k in ('a','b','c')] for r in face_rows],np.int64)
 if np.any(faces>=2411):raise ValueError('D36 coarse O face contains non-O vertex')
 # Build exact face->unique tetrahedron incidence from indexed tetrahedra.
 incidence={}
 for ti,t in enumerate(tet):
  for raw in ((t[1],t[2],t[3]),(t[0],t[3],t[2]),(t[0],t[1],t[3]),(t[0],t[2],t[1])):
   key=tuple(sorted(map(int,raw)));incidence.setdefault(key,[]).append(ti)
 face_tet=np.empty(len(faces),np.int64)
 for i,f in enumerate(faces):
  hit=incidence.get(tuple(sorted(map(int,f))),[])
  if len(hit)!=1:raise ValueError(f'outer source face {i} has {len(hit)} adjacent tets')
  face_tet[i]=hit[0]
 if not np.all(mat[face_tet]==1):raise ValueError('outer O surface is not attached to skin material 1')
 # Exact constant P1 gradient for each outer-face-adjacent tet: F=Ds Dm^{-1}.
 t0=xref[tet[face_tet]];t1=x1[tet[face_tet]]
 Dm=np.transpose(t0[:,1:,:]-t0[:,:1,:],(0,2,1))
 Ds=np.transpose(t1[:,1:,:]-t1[:,:1,:],(0,2,1))
 Fface=Ds@np.linalg.inv(Dm)
 detface=np.linalg.det(Fface)
 if not np.isfinite(Fface).all():raise ValueError('nonfinite boundary P1 gradients')
 # At coarse vertices P1 gradients are elementwise discontinuous. The owner-level
 # diagnostic uses reference-area-weighted incident boundary-cell averaging.
 tri0=x0[faces]; area=np.linalg.norm(np.cross(tri0[:,1]-tri0[:,0],tri0[:,2]-tri0[:,0]),axis=1)*0.5
 if np.any(area<=0):raise ValueError('degenerate reference O face')
 Fnode=np.zeros((2411,3,3),float);den=np.zeros(2411,float)
 for k in range(3):
  np.add.at(Fnode,faces[:,k],Fface*area[:,None,None]);np.add.at(den,faces[:,k],area)
 Fnode/=den[:,None,None]
 if np.any(den<=0):raise ValueError('uncovered O vertex in P1 gradient map')
 R=np.asarray(json.loads(FRAME.read_text())['placement_rotation'],float); trans=np.asarray(json.loads(FRAME.read_text())['placement_translation_m'],float)
 # D36 registered source coordinates -> AP coordinate frame.
 source_current=xyz_csv(cur_rows)
 source_ap=source_current@R.T+trans
 source_ids=np.array([int(r['source_vertex']) for r in emb],np.int64)
 direct=source_ids!=np.iinfo(np.uint32).max
 source_faces=np.asarray([[int(r[f'b{k}']) for k in range(3)] for r in []]) if False else None
 source_face_ids=np.array([int(r['face']) for r in emb],np.int64)
 bary=np.asarray([[float(r[f'b{k}']) for k in range(3)] for r in emb],float)
 skin_faces=np.asarray(old['source_faces'],np.int64)
 # Exact D36 source points at O nodes are read using direct IDs or source-face barycentrics.
 source_anchor=np.empty((2411,3),float)
 for i in range(2411):
  if direct[i]:source_anchor[i]=source_ap[source_ids[i]]
  else:
   fi=source_face_ids[i]
   if fi<0 or fi>=len(skin_faces):raise ValueError(f'invalid detailed source face at coarse row {i}')
   source_anchor[i]=bary[i]@source_ap[skin_faces[fi]]
 delta=source_anchor-x0[:2411]
 # D36-to-AP alignment was already independently pinned; use AP rigid frame for both
 # the source offset and the P1 gradient to avoid mixing the <0.13um fit residual.
 candidate_o=x1[:2411]; coarse_delta=candidate_o-x0[:2411]
 old_direct=np.asarray(old['direct_anchor_displacements_m'],float)
 old_bary=np.asarray(old['bary_anchor_displacements_m'],float)
 old_ids=np.asarray(old['direct_anchor_ids'],np.int64);old_bary_faces=np.asarray(old['bary_anchor_face_ids'],np.int64)
 if not np.array_equal(old_ids,source_ids[direct]) or not np.array_equal(old_bary_faces,source_face_ids[~direct]):raise ValueError('saved prolongation anchor IDs differ from exact D36 ancestry')
 delta_direct=coarse_delta[direct];delta_bary=coarse_delta[~direct]
 direct_parity=np.linalg.norm(old_direct@R.T-delta_direct,axis=1)
 bary_parity=np.linalg.norm(old_bary@R.T-delta_bary,axis=1)
 # Legacy additive result: source + Delta. Co-deformed offset result: O_candidate + F*offset.
 naive=source_anchor+coarse_delta
 transported=candidate_o+np.einsum('nij,nj->ni',Fnode,delta)
 correction=transported-naive
 # Save exact anchor-level experimental arrays only; no detailed mesh or NHSKIN is emitted.
 npz=out/'anchor-offset-transport.npz'
 np.savez_compressed(npz,outer_o_row=np.arange(2411,dtype=np.uint32),source_anchor_ap_m=source_anchor.astype(np.float64),baseline_o_ap_m=x0[:2411],candidate_o_ap_m=candidate_o.astype(np.float32),source_offset_ap_m=delta.astype(np.float64),area_weighted_p1_gradient=Fnode,legacy_additive_anchor_prediction_m=naive,transported_anchor_prediction_m=transported,offset_transport_correction_m=correction,direct_mask=direct,bary_source_face=source_face_ids,bary_weights=bary,direct_saved_delta_d36_m=old_direct,direct_saved_delta_ap_m=old_direct@R.T,bary_saved_delta_d36_m=old_bary,bary_saved_delta_ap_m=old_bary@R.T,coarse_candidate_delta_ap_m=coarse_delta)
 sv=np.linalg.svd(Fnode,compute_uv=False)
 report={"schema":"numi.human.current294-anchor-offset-p1-transport.v1","classification":"bounded_anchor_level_transport_experiment_no_harmonic_rerun_no_NHSKIN_mutation_no_admission","method":{"owner_observation":"AcceptedFEMSkinTransfer::makeCurrentCoarseBaseline rotates coarse-source rest offsets with donor rigid pose; applyReference adds interpolated coarse world Delta only.","proposed_equation":"u_detail = u_coarse + (F_P1 - I) * (x_detail_current - x_coarse_baseline_current), so x_detail_new = x_coarse_accepted + F_P1 * offset.","gradient":"Exact constant P1 gradient from the unique material-1 tetrahedron incident to each outer O face; for each O vertex use reference-area-weighted average of incident outer-face gradients because vertex gradients are discontinuous.","scope":"2411 exact D36 O-to-294 direct/bary source anchors only; no extension to other detailed vertices and no re-solve of the harmonic prolongation."},"inputs":{str(p):sha(p) for p in inputs},"frame":{"common_rigid_rotation":R.tolist(),"common_rigid_translation_m":trans.tolist(),"AP_vs_repair_reference_max_m":float(np.max(np.linalg.norm(x0-xref,axis=1))),"saved_delta_axis_conversion":"D36 row vectors rotate by R.T into AP before comparison"},"checks":{"outer_faces_exact_boundary_tets":int(len(face_tet)),"outer_faces_unique_skin_material_tets":int(np.count_nonzero(mat[face_tet]==1)),"P1_face_gradient_det":q(detface),"P1_face_nonpositive_det_count":int(np.count_nonzero(detface<=0)),"P1_nodal_area_weighted_min_singular":q(sv[:,-1]),"P1_nodal_area_weighted_max_singular":q(sv[:,0]),"P1_nodal_det":q(np.linalg.det(Fnode)),"P1_nodal_nonpositive_det_count":int(np.count_nonzero(np.linalg.det(Fnode)<=0)),"D36_direct_anchor_delta_parity_max_m":float(direct_parity.max(initial=0.0)),"D36_bary_anchor_delta_parity_max_m":float(bary_parity.max(initial=0.0)),"source_offset_m":q(np.linalg.norm(delta,axis=1)),"legacy_vs_transported_correction_m":q(np.linalg.norm(correction,axis=1)),"direct_offset_correction_m":q(np.linalg.norm(correction[direct],axis=1)),"bary_offset_correction_m":q(np.linalg.norm(correction[~direct],axis=1)),"legacy_vs_P1_transported_position_m":q(np.linalg.norm(transported-naive,axis=1))},"outputs":{"npz":{"path":str(npz),"sha256":sha(npz)}},"limitations":["This tests co-deforming the existing coarse-to-detail source offset; it does not repair the 294 source correspondence residuals or change the coarse P1 candidate.","The P1 gradient at a mesh vertex is not unique. The explicitly reported area-weighted incident boundary-cell average is a nodal reconstruction, not a cellwise exact gradient for every detailed transfer row.","No full 61,950-vertex detail prolongation, harmonic solve, geometry/collision/mass/native/physical gate, or NHSKIN mutation was performed."],"runtime":{"seconds":time.time()-start,"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__}}
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\\n')
 print(json.dumps({"report":str(out/'report.json'),"checks":report['checks'],"npz_sha256":sha(npz)},indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
