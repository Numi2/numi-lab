#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time,subprocess
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';B=W/'combined-d071-b109-b61-001';D=W/'b74-104/attempt-001-contact-feature-local-fit-806-001'
OUT=D/'screen-alpha0125-absolute-001'
BASE=B/'candidate.npz';AUD=B/'audit-159/report.json';SUM=B/'audit-130/summary.json'
DIR=D/'fit-direction-slsqp-line-search-stop.npz'
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
SCREEN=W/'b43-16/attempt-030-radius-aware-position-001/rebase-024-on-806/deep-backtrack-001/screen_rebased_direction.py'
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
TRANSFER=W/'b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight';EXPORT=E/'E/generic-static-query-042/prepare_generic_query_points.py';QUERYBIN=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query')
PIN='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756';ALPHA=.125;BODY=74;BONE=104;FACES=np.asarray([22587,22588,22589,22590],dtype=np.int64)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def pairset(r):return {(int(f),int(t)) for f,t in (r.get('result') or {}).get('triangle_pairs',[])}
def point_query(points,outdir):
 outdir.mkdir(parents=True,exist_ok=True);inp=outdir/'points.npz';np.savez_compressed(inp,points_world_f32=np.asarray(points,dtype=np.float32))
 pts=outdir/'points.csv';cmd=[sys.executable,str(EXPORT),'--candidate-npz',str(inp),'--input-sha256',sha(inp),'--field','points_world_f32','--output',str(pts)]
 run=subprocess.run(cmd,capture_output=True,text=True);(outdir/'prepare.log').write_text(run.stdout+run.stderr)
 if run.returncode:raise RuntimeError('centroid exporter failed')
 pairs=outdir/'pairs.csv';nodes=outdir/'nodes.csv';run=subprocess.run([str(QUERYBIN),str(BANKDIR/'exact-emitted.triangles.csv'),str(BANKDIR/'exact-emitted.bodies.csv'),str(pts),str(pairs),str(nodes),'185'],capture_output=True,text=True);(outdir/'query.log').write_text(run.stdout+run.stderr)
 if run.returncode:raise RuntimeError('centroid allbank query failed')
 return list(csv.DictReader(pairs.open())),list(csv.DictReader(nodes.open())),inp
def main():
 t=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
 if sha(BASE)!=PIN:raise ValueError('806 baseline pin mismatch')
 if sha(DIR)!='83a5ae671e3471deb650542729ad2f26d06cdfac2047d1cd636e5c314a370f13':raise ValueError('direction artifact pin mismatch')
 audit=mod('b74v4_audit',AUD159);exact=audit.exact_module();helper=mod('b74v4_helper',HELPER);screen=mod('b74v4_screen',SCREEN);q=mod('b74v4_q',QPATH);tf=mod('b74v4_tf',TRANSFER)
 with np.load(BASE,allow_pickle=False) as z:P={k:z[k].copy() for k in z.files}
 with np.load(DIR,allow_pickle=False) as z:nodes=np.asarray(z['variable_nodes'],dtype=np.int64);d=np.asarray(z['delta_mm'],dtype=np.float64)
 X=np.asarray(P['points_world_m'],dtype=np.float32);F=np.asarray(P['source_O_faces'],dtype=np.int64);Y=X.copy()
 for i,n in enumerate(nodes):Y[n]=(X[n].astype(np.float64)+(ALPHA*d[i])/1000.).astype(np.float32)
 changednodes=np.flatnonzero(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64);chf=np.flatnonzero(np.any(np.isin(F,changednodes),axis=1)).astype(np.int64)
 metrics=screen.shape_metrics(X,Y,F,chf)
 Xtri=X[F[chf]].astype(np.float64)*1000.;Ytri=Y[F[chf]].astype(np.float64)*1000.
 n0=np.cross(Xtri[:,1]-Xtri[:,0],Xtri[:,2]-Xtri[:,0]);n1=np.cross(Ytri[:,1]-Ytri[:,0],Ytri[:,2]-Ytri[:,0])
 signed_ratio=np.sum(n0*n1,axis=1)/np.maximum(np.sum(n0*n0,axis=1),1e-300)
 degenerate=np.flatnonzero(np.linalg.norm(n1,axis=1)<=1e-12)
 inverted=np.flatnonzero(signed_ratio<=0)
 anchor=np.asarray(P['typed_coarse_anchor_output_rows_018'],dtype=np.int64);anchor_same=bool(np.array_equal(Y[anchor].view(np.uint32),X[anchor].view(np.uint32)))
 allfaces=np.arange(len(F),dtype=np.int64);qarea=q.full_f32_area_gate(Y,X.astype(np.float64)*1000.,P['area_reference_037_points_world_m_f64'].astype(np.float64)*1000.,F,.1)
 with np.load(B/'skin-fat-layers-541.npz',allow_pickle=False) as z:L={k:z[k].copy() for k in z.files}
 layer=screen.layer_check(X,Y,F,chf,L,tf)
 ad=json.loads(AUD.read_text());doc=json.loads(SUM.read_text());bankrow=next(r for r in ad['banks'] if (int(r['body']),int(r['stable_bone_id']))==(BODY,BONE));raw0=pairset(bankrow)
 with np.load(BASE,allow_pickle=False) as z:ip=z['refinement_face_parent_ancestry_indptr'].astype(np.int64);pa=z['refinement_face_parent_ancestry_face_rows'].astype(np.int64)
 map0={(int(p),int(bt)) for f,bt in raw0 for p in pa[ip[f]:ip[f+1]]}
 BT,bids,_,_,M,off,pos,_,_,_=helper.read_bank_inputs(BANKDIR,E)
 stored,box0=mod('b74v4_fithelpers',D/'run_fit.py').audit_one_bank(X.astype(np.float64),F,chf,BT,bids,M,off,pos,exact,audit,BODY,BONE)
 expect={(f,b) for f,b in raw0 if f in set(map(int,chf))}
 if stored!=expect:raise ValueError('baseline exact local control mismatch')
 afterlocal,box1=mod('b74v4_fithelpers2',D/'run_fit.py').audit_one_bank(Y.astype(np.float64),F,chf,BT,bids,M,off,pos,exact,audit,BODY,BONE)
 raw1=(raw0-expect)|afterlocal;map1={(int(p),int(bt)) for f,bt in raw1 for p in pa[ip[f]:ip[f+1]]}
 newraw=raw1-raw0;removedraw=raw0-raw1;newmap=map1-map0;removedmap=map0-map1
 rec,_=exact.float32_surface_records(Y,F,source_seam_groups=());sr=exact._audit_self_pairs_touching_faces(rec,set(map(int,chf)));selfpairs={(int(a),int(b)) for a,b in (sr.get('triangle_pairs') or [])}
 # Exact complete-185 location at the four crossing-face centroids.
 bc=np.mean(X[F[FACES]].astype(np.float64),axis=1);yc=np.mean(Y[F[FACES]].astype(np.float64),axis=1)
 bp,bn,bnpz=point_query(bc,OUT/'centroids-baseline');cp,cn,cnpz=point_query(yc,OUT/'centroids-alpha0125')
 for rs in [bn,cn]:
  if any(int(r['total_bone_banks'])!=185 or int(r['ambiguous_banks']) or int(r['invalid_banks']) for r in rs):raise ValueError('incomplete centroid point query')
 inside0={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])) for r in bp if r['classification']=='inside'}
 inside1={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])) for r in cp if r['classification']=='inside'}
 pdoc={'faces':FACES.tolist(),'baseline_inside_count':len(inside0),'candidate_inside_count':len(inside1),'target_baseline':[r['classification'] for r in bp if (int(r['bank_body']),int(r['bank_stable_id']))==(BODY,BONE)],'target_candidate':[r['classification'] for r in cp if (int(r['bank_body']),int(r['bank_stable_id']))==(BODY,BONE)],'new_inside':sorted([list(z) for z in inside1-inside0]),'removed_inside':sorted([list(z) for z in inside0-inside1]),'baseline_points_sha256':sha(bnpz),'candidate_points_sha256':sha(cnpz)}
 report={'schema':'numi.human.b74-104.contact-feature-alpha0125-absolute-screen.v1','status':'DIAGNOSTIC_ABSOLUTE_SHAPE_AND_LAYER_SCREEN','geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'baseline_candidate_sha256':PIN,'direction_sha256':sha(DIR),'alpha':ALPHA,'emission':'(base_world_float32 promoted to float64 + alpha*delta_mm/1000) cast once to float32','solver_status_from_direction_receipt':'SLSQP positive directional derivative; feasible iterate preserved, optimizer did not report success','changed_nodes':int(len(changednodes)),'changed_faces':int(len(chf)),'anchors_bitwise_same':anchor_same,'absolute_geometry':{'face_count':len(chf),'min_angle_before_deg':metrics['min_angle_before_deg'],'min_angle_after_deg':metrics['min_angle_after_deg'],'max_normal_turn_deg':metrics['maximum_normal_turn_deg'],'minimum_area_ratio':metrics['minimum_area_ratio'],'minimum_per_face_min_angle_delta_deg':metrics['minimum_min_angle_delta_deg'],'maximum_aspect_ratio_ratio':metrics['maximum_aspect_ratio_ratio'],'zero_area_before':metrics['zero_area_before'],'zero_area_after':metrics['zero_area_after'],'degenerate_face_rows':chf[degenerate].tolist(),'inverted_face_rows':chf[inverted].tolist(),'min_signed_area_ratio':float(np.min(signed_ratio)),'full_area_gate':qarea},'layer_gate':layer,'exact_target_74_104':{'raw_before':len(raw0),'raw_after':len(raw1),'mapped_before':len(map0),'mapped_after':len(map1),'new_raw':sorted([list(z) for z in newraw]),'removed_raw':sorted([list(z) for z in removedraw]),'new_mapped':sorted([list(z) for z in newmap]),'removed_mapped':sorted([list(z) for z in removedmap]),'changed_face_baseline_pairs':len(stored),'changed_face_candidate_pairs':len(afterlocal),'aabb_before':box0,'aabb_after':box1},'exact_self_touching_changed_faces':{'count':len(selfpairs),'pairs':sorted([list(z) for z in selfpairs])},'face_centroid_query':pdoc,'not_run':{'all_185_neighbor_bank_face_audit':'not run because this is a non-improving diagnostic','complete_body_self':'only exact pairs touching changed faces run'},'candidate_emitted':False,'elapsed_s':time.monotonic()-t}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'min_angle':[metrics['min_angle_before_deg'],metrics['min_angle_after_deg']],'normal_turn':metrics['maximum_normal_turn_deg'],'min_area_ratio':metrics['minimum_area_ratio'],'angle_delta':metrics['minimum_min_angle_delta_deg'],'degenerate':report['absolute_geometry']['degenerate_face_rows'],'inverted':report['absolute_geometry']['inverted_face_rows'],'layer':layer['pass'],'raw':[len(raw0),len(raw1)],'mapped':[len(map0),len(map1)],'new_mapped':report['exact_target_74_104']['new_mapped'],'removed_mapped':report['exact_target_74_104']['removed_mapped'],'self':len(selfpairs),'centroids':[len(inside0),len(inside1)],'target_centroids':[pdoc['target_baseline'],pdoc['target_candidate']]},indent=2))
if __name__=='__main__':main()
