#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time,subprocess
import numpy as np
from scipy import sparse
from scipy.optimize import minimize,LinearConstraint,Bounds
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';B=W/'combined-d071-b109-b61-001'
OUT=W/'b74-104/attempt-001-contact-feature-local-fit-806-001'
BASE=B/'candidate.npz';AUD=B/'audit-159/report.json';SUM=B/'audit-130/summary.json'
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
FIT= W/'b43-16/attempt-031-contact-feature-fit-001/fit_contact_features.py'
SCREEN=W/'b43-16/attempt-030-radius-aware-position-001/rebase-024-on-806/deep-backtrack-001/screen_rebased_direction.py'
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
TRANSFER=W/'b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
EXPORT=E/'E/generic-static-query-042/prepare_generic_query_points.py'
QUERYBIN=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query')
LAYER=B/'skin-fat-layers-541.npz'
PIN='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
BODY=74;BONE=104;MARGIN_MM=.03;TRUST_MM=3.0;LAMBDA_SMOOTH=5.0;MAXITER=300
SEED_FACES=np.asarray([22587,22588,22589,22590],dtype=np.int64)
CORNERS={19094,19092,12694}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p)
 if sp is None or sp.loader is None:raise RuntimeError(f'module not found {p}')
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def pairset(r):
 return {(int(f),int(t)) for f,t in (r.get('result') or {}).get('triangle_pairs',[])}
def mapped(raw,ip,pa):
 out=set()
 for f,t in raw:
  for q in pa[int(ip[f]):int(ip[f+1])]:out.add((int(q),int(t)))
 return out
def audit_one_bank(world,F,changed,bankT,bankids,M,off,pos,exact,audit,body,bone):
 local=(world-pos[body])@M[body].T-off[body];skin=local[F[changed]]
 mask=(bankids[:,0]==body)&(bankids[:,1]==bone)
 bank=bankT[mask];bid=bankids[mask,3]
 if not len(bank):raise ValueError(f'empty typed bank {body}/{bone}')
 blo,bhi=bank.min((0,1)),bank.max((0,1));slo,shi=skin.min(1),skin.max(1)
 active=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
 if not len(active):return set(),{'skin_aabb_active':0,'bank_aabb_active':0}
 rlo,rhi=slo[active].min(0),shi[active].max(0)
 ba=np.flatnonzero(np.all(bank.max(1)>=rlo,axis=1)&np.all(bank.min(1)<=rhi,axis=1))
 sr=audit.records(skin[active],changed[active],exact);br=audit.records(bank[ba],bid[ba],exact)
 out=exact._audit_pair(sr,br,same_surface=False)
 return {(int(a),int(b)) for a,b in (out.get('triangle_pairs') or [])},{'skin_aabb_active':int(len(active)),'bank_aabb_active':int(len(ba))}
def point_query(points,outdir):
 outdir.mkdir(parents=True,exist_ok=True)
 inp=outdir/'points.npz';np.savez_compressed(inp,points_world_f32=np.asarray(points,dtype=np.float32))
 pts=outdir/'points.csv';args=[sys.executable,str(EXPORT),'--candidate-npz',str(inp),'--input-sha256',sha(inp),'--field','points_world_f32','--output',str(pts)]
 run=subprocess.run(args,capture_output=True,text=True);(outdir/'prepare.log').write_text(run.stdout+run.stderr)
 if run.returncode:raise RuntimeError('point exporter failed')
 pairs=outdir/'pairs.csv';nodes=outdir/'nodes.csv';cmd=[str(QUERYBIN),str(BANKDIR/'exact-emitted.triangles.csv'),str(BANKDIR/'exact-emitted.bodies.csv'),str(pts),str(pairs),str(nodes),'185']
 run=subprocess.run(cmd,capture_output=True,text=True);(outdir/'query.log').write_text(run.stdout+run.stderr)
 if run.returncode:raise RuntimeError('all-185 centroid query failed')
 return list(csv.DictReader(pts.open())),list(csv.DictReader(pairs.open())),list(csv.DictReader(nodes.open())),inp
def main():
 t=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
 if sha(BASE)!=PIN:raise ValueError('exact 806 baseline SHA mismatch')
 audit=mod('b74_audit159',AUD159);exact=audit.exact_module();helper=mod('b74_bankhelper',HELPER);fit=mod('b74_fit',FIT);screen=mod('b74_screen',SCREEN);q=mod('b74_q',QPATH);tf=mod('b74_transfer',TRANSFER)
 with np.load(BASE,allow_pickle=False) as z:P={k:z[k].copy() for k in z.files}
 X=np.asarray(P['points_world_m'],dtype=np.float32);F=np.asarray(P['source_O_faces'],dtype=np.int64)
 ad=json.loads(AUD.read_text());doc=json.loads(SUM.read_text())
 if ad.get('candidate_sha256')!=PIN or doc.get('candidate_sha256')!=PIN or not ad.get('complete'):raise ValueError('stale/incomplete 806 audit')
 bankrow=next(r for r in ad['banks'] if (int(r['body']),int(r['stable_bone_id']))==(BODY,BONE))
 rawbase=pairset(bankrow)
 if len(rawbase)!=25:raise ValueError(f'baseline raw pair count {len(rawbase)} not pinned 25')
 ip=np.asarray(P['refinement_face_parent_ancestry_indptr'],dtype=np.int64);pa=np.asarray(P['refinement_face_parent_ancestry_face_rows'],dtype=np.int64)
 mapbase=mapped(rawbase,ip,pa)
 if len(mapbase)!=19:raise ValueError(f'baseline mapped identities {len(mapbase)} not pinned 19')
 # No bank-74/104 sampled inside nodes in the complete 806 point audit.
 inside=[r for r in doc.get('inside_witnesses',[]) if (int(r['bank_body']),int(r['bank_stable_id']))==(BODY,BONE) and r.get('classification')=='inside' and int(r.get('ambiguous',0))==0]
 if inside:raise ValueError(f'expected zero sampled inside nodes, got {len(inside)}')
 bankT,bankids,_,_,M,off,pos,_,_,_=helper.read_bank_inputs(BANKDIR,E)
 bodytri={int(bankids[i,3]):int(i) for i in np.flatnonzero((bankids[:,0]==BODY)&(bankids[:,1]==BONE))}
 Xloc=(X.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]
 # Exact segment endpoints of each stored crossing pair on exact 806 geometry.
 feature_rows=[];feature_meta=[]
 for sf,bt in sorted(rawbase):
  if sf not in set(map(int,SEED_FACES)):raise ValueError(f'unexpected target skin face {sf}')
  if bt not in bodytri:raise ValueError(f'missing exact bank triangle {bt}')
  tri=bankT[bodytri[bt]]
  feat=fit.pair_contact_features(Xloc[F[sf]],tri)
  if len(feat)!=2:raise ValueError(f'exact pair {sf}/{bt} yielded {len(feat)} segment endpoints')
  nl=np.cross(tri[1]-tri[0],tri[2]-tri[0]);nl/=np.linalg.norm(nl);nw=nl@M[BODY];nw/=np.linalg.norm(nw)
  for j,(pt,bary) in enumerate(feat):
   feature_rows.append((int(sf),int(bt),np.asarray(bary,dtype=np.float64),nw))
   feature_meta.append({'skin_face':int(sf),'bone_triangle':int(bt),'endpoint':j,'skin_barycentric':np.asarray(bary).tolist(),'bone_normal_world':nw.tolist()})
 if len(feature_rows)!=50:raise ValueError(f'expected 50 exact endpoints, got {len(feature_rows)}')
 # Two-ring connected support around the four crossing faces, retaining the three source-face corners and typed anchors.
 edges=np.unique(np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
 adj=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(X),len(X)))
 seed_nodes=set(map(int,F[SEED_FACES].reshape(-1)));support=set(seed_nodes);front=set(seed_nodes);rings=[]
 for rr in range(2):
  nxt=set(map(int,adj[list(front)].indices))-support;support|=nxt;front=nxt;rings.append(sorted(nxt))
 anchors=set(map(int,P['typed_coarse_anchor_output_rows_018']))
 varnodes=sorted(support-CORNERS-anchors);loc={n:i for i,n in enumerate(varnodes)}
 if not varnodes:raise ValueError('empty movable support')
 # Endpoints must be represented in movable nodes; any fixed-weight contribution is zero displacement.
 rows=[];cols=[];vals=[]
 for j,(sf,bt,bary,nw) in enumerate(feature_rows):
  for k,node in enumerate(F[sf]):
   ii=loc.get(int(node))
   if ii is None:continue
   for c in range(3):
    v=float(bary[k]*nw[c])
    if v:rows.append(j);cols.append(3*ii+c);vals.append(v)
 C=sparse.coo_matrix((vals,(rows,cols)),shape=(len(feature_rows),3*len(varnodes))).tocsr()
 rowmax=np.asarray(abs(C).sum(axis=1)).reshape(-1)
 if np.any(rowmax<1e-10):raise ValueError(f'{int(np.count_nonzero(rowmax<1e-10))} contact endpoints have no movable support')
 # Smooth sparse quadratic objective over support graph; collar nodes are more expensive.
 weights=[]
 for n in varnodes:
  ring=0 if n in seed_nodes else (1 if n in set(rings[0]) else 2)
  weights.append([1.,8.,30.][ring])
 H=sparse.diags(np.repeat(np.asarray(weights,dtype=np.float64),3),format='csr')
 ep_rows=[];ep_cols=[];ep_vals=[];rnum=0;edge_rows=[]
 for u,v in edges:
  iu=loc.get(int(u));iv=loc.get(int(v))
  if iu is None and iv is None:continue
  length=float(np.linalg.norm(X[u].astype(np.float64)-X[v].astype(np.float64))*1000.)
  w=1.0/np.sqrt(max(length,.05))
  for c in range(3):
   if iu is not None:ep_rows.append(rnum);ep_cols.append(3*iu+c);ep_vals.append(w)
   if iv is not None:ep_rows.append(rnum);ep_cols.append(3*iv+c);ep_vals.append(-w)
   rnum+=1
  edge_rows.append((int(u),int(v),length))
 E0=sparse.coo_matrix((ep_vals,(ep_rows,ep_cols)),shape=(rnum,3*len(varnodes))).tocsr()
 H=H+LAMBDA_SMOOTH*(E0.T@E0)+1e-8*sparse.eye(3*len(varnodes),format='csr')
 Hd=H.toarray();Cd=C.toarray()
 def obj(x):return .5*float(x@Hd@x)
 def jac(x):return Hd@x
 res=minimize(obj,np.zeros(C.shape[1]),jac=jac,method='SLSQP',bounds=Bounds(-TRUST_MM,TRUST_MM),constraints=[LinearConstraint(Cd,MARGIN_MM,np.inf)],options={'maxiter':MAXITER,'ftol':1e-10,'disp':False})
 x=np.asarray(res.x,dtype=np.float64)
 residual=np.asarray(C@x).reshape(-1)
 solve={'success':bool(res.success),'status':int(res.status),'message':str(res.message),'iterations':int(res.nit),'variables':int(C.shape[1]),'constraints':int(C.shape[0]),'objective':float(obj(x)),'contact_margin_mm':MARGIN_MM,'minimum_predicted_endpoint_motion_mm':float(residual.min(initial=np.inf)),'max_predicted_endpoint_motion_mm':float(residual.max(initial=-np.inf)),'max_node_move_mm':float(np.max(np.linalg.norm(x.reshape(-1,3),axis=1),initial=0)),'trust_limit_mm':TRUST_MM,'support_nodes':len(support),'variable_nodes':varnodes,'seed_patch_nodes':sorted(seed_nodes),'fixed_corners':sorted(CORNERS),'anchors_in_support':sorted(anchors&support),'ring_node_counts':[len(z) for z in rings],'graph_edges':len(edge_rows),'smoothness_weight':LAMBDA_SMOOTH}
 if np.any(residual<MARGIN_MM-1e-7) or solve['max_node_move_mm']>TRUST_MM*1.75:
  report={'schema':'numi.human.b74-104.contact-feature-fit-806.v1','status':'STOPPED_CONTACT_FEATURE_FIT_FAILED_OR_OVERTRUST','baseline_candidate_sha256':PIN,'baseline_raw_pairs':len(rawbase),'baseline_mapped_identities':len(mapbase),'baseline_raw_pairs_list':sorted([list(z) for z in rawbase]),'baseline_mapped_identities_list':sorted([list(z) for z in mapbase]),'baseline_sampled_inside_nodes':0,'source_face_rows':sorted(set(map(int,pa[ip[22587]:ip[22588]]))),'actual_intersection_endpoint_count':len(feature_rows),'actual_intersection_features':feature_meta,'solver':solve,'candidate_emitted':False,'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'elapsed_s':time.monotonic()-t}
  np.savez_compressed(OUT/'fit-direction.npz',variable_nodes=np.asarray(varnodes,dtype=np.int64),delta_mm=x.reshape(-1,3),contact_matrix=Cd,contact_lower_bound_mm=np.full(len(feature_rows),MARGIN_MM))
  (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
  print(json.dumps({'status':report['status'],'solver':solve},indent=2));return
 direction=x.reshape(-1,3)
 with np.load(LAYER,allow_pickle=False) as z:L={k:z[k].copy() for k in z.files}
 if not np.array_equal(L['outer_points_f32'].view(np.uint32),X.view(np.uint32)) or not np.array_equal(L['outer_faces'],F[:,[0,2,1]]):raise ValueError('806 layer base mismatch')
 ref=np.asarray(P['area_reference_037_points_world_m_f64'],dtype=np.float64);bm=X.astype(np.float64)*1000.;refmm=ref*1000.
 trial_rows=[];first=None;selected=None;pointq=None;allbank=None
 for alpha in [1.,.5,.25,.125]:
  Y=X.copy()
  for i,n in enumerate(varnodes):Y[n]=(X[n].astype(np.float64)+(alpha*direction[i])/1000.).astype(np.float32)
  changed_nodes=np.flatnonzero(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64)
  changed_faces=np.flatnonzero(np.any(np.isin(F,changed_nodes),axis=1)).astype(np.int64)
  sm=screen.shape_metrics(X,Y,F,changed_faces)
  ag=q.full_f32_area_gate(Y,bm,refmm,F,.1)
  spass=bool(sm['zero_area_after']==0 and sm['minimum_area_ratio']>=.95 and sm['maximum_normal_turn_deg']<=5.0 and sm['minimum_min_angle_delta_deg']>=-.05 and sm['maximum_aspect_ratio_ratio']<=1.03 and ag['pass'])
  lg=screen.layer_check(X,Y,F,changed_faces,L,tf)
  tr={'alpha':alpha,'moved_node_count':int(len(changed_nodes)),'affected_face_count':int(len(changed_faces)),'shape_metrics':sm,'area_gate':ag,'shape_pass':spass,'layer_gate':lg,'exact_checked':False}
  if not spass or not lg['pass']:
   tr['status']='REJECTED_SHAPE_OR_LAYER';trial_rows.append(tr);continue
  tr['exact_checked']=True
  storedpairs,box0=audit_one_bank(X.astype(np.float64),F,changed_faces,bankT,bankids,M,off,pos,exact,audit,BODY,BONE)
  expected={(f,b) for f,b in rawbase if f in set(map(int,changed_faces))}
  if storedpairs!=expected:raise ValueError(f'baseline exact target changed-row control mismatch: {len(storedpairs)} != {len(expected)}')
  localafter,box1=audit_one_bank(Y.astype(np.float64),F,changed_faces,bankT,bankids,M,off,pos,exact,audit,BODY,BONE)
  rawafter=(rawbase-expected)|localafter;mapafter=mapped(rawafter,ip,pa)
  rawnew=rawafter-rawbase;rawremoved=rawbase-rawafter;mapnew=mapafter-mapbase;mapremoved=mapbase-mapafter
  rec,_=exact.float32_surface_records(Y,F,source_seam_groups=())
  sr=exact._audit_self_pairs_touching_faces(rec,set(map(int,changed_faces)))
  selfpairs={(int(a),int(b)) for a,b in (sr.get('triangle_pairs') or [])}
  tr.update(status='LOCAL_TARGET_EXACT_COMPLETE',target_exact={'raw_before':len(rawbase),'raw_after':len(rawafter),'mapped_before':len(mapbase),'mapped_after':len(mapafter),'new_raw':sorted([list(z) for z in rawnew]),'removed_raw':sorted([list(z) for z in rawremoved]),'new_mapped':sorted([list(z) for z in mapnew]),'removed_mapped':sorted([list(z) for z in mapremoved]),'changed_face_baseline_pairs':len(storedpairs),'changed_face_candidate_pairs':len(localafter),'aabb_before':box0,'aabb_after':box1},self_touching_changed_faces={'count':len(selfpairs),'pairs':sorted([list(z) for z in selfpairs])})
  trial_rows.append(tr)
  # Keep the first exact local trial for diagnostics; full all-bank scope only if local target has strict identity improvement.
  if first is None:first=(Y,changed_nodes,changed_faces,tr)
  if len(mapafter)<len(mapbase) and not mapnew and not rawnew and not selfpairs:
   selected=(Y,changed_nodes,changed_faces,tr);break
 # Point-location check on centroids of the four original crossing faces, all 185 banks.
 centroid_report=None
 if first:
  Y,changed_nodes,changed_faces,tr=first
  basecent=np.mean(X[F[SEED_FACES]].astype(np.float64),axis=1)
  candcent=np.mean(Y[F[SEED_FACES]].astype(np.float64),axis=1)
  bp,bpair,bnodes,bnpz=point_query(basecent,OUT/'centroids-baseline')
  cp,cpair,cnodes,cnpz=point_query(candcent,OUT/'centroids-candidate')
  for nrows in (bnodes,cnodes):
   if any(int(r['total_bone_banks'])!=185 or int(r['ambiguous_banks']) or int(r['invalid_banks']) for r in nrows):raise ValueError('centroid point query incomplete/ambiguous')
  baseline_inside={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])) for r in bpair if r['classification']=='inside'}
  candidate_inside={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])) for r in cpair if r['classification']=='inside'}
  target_before=[r for r in bpair if (int(r['bank_body']),int(r['bank_stable_id']))==(BODY,BONE)]
  target_after=[r for r in cpair if (int(r['bank_body']),int(r['bank_stable_id']))==(BODY,BONE)]
  centroid_report={'faces':SEED_FACES.tolist(),'all_bank_baseline_inside_pair_count':len(baseline_inside),'all_bank_candidate_inside_pair_count':len(candidate_inside),'target_baseline_classifications':[r['classification'] for r in target_before],'target_candidate_classifications':[r['classification'] for r in target_after],'new_inside_pairs':sorted([list(z) for z in candidate_inside-baseline_inside]),'removed_inside_pairs':sorted([list(z) for z in baseline_inside-candidate_inside]),'baseline_points_sha256':sha(bnpz),'candidate_points_sha256':sha(cnpz)}
 # If target identities improved and all local shape/layer/self gates pass, audit all banks on changed face rows.
 if selected:
  Y,changed_nodes,changed_faces,tr=selected
  stored_by={(int(r['body']),int(r['stable_bone_id'])):pairset(r) for r in ad['banks']}
  after_by={};new_global=set();removed_global=set();new_global_mapped=set();removed_global_mapped=set();bank_counts=[]
  for body,bone in sorted({(int(a),int(b)) for a,b in bankids[:,:2]}):
   before=stored_by.get((body,bone),set())
   local0,box0=audit_one_bank(X.astype(np.float64),F,changed_faces,bankT,bankids,M,off,pos,exact,audit,body,bone)
   expected={(f,t) for f,t in before if f in set(map(int,changed_faces))}
   if local0!=expected:raise ValueError(f'zero-motion all-bank control mismatch {body}/{bone} exact={len(local0)} expected={len(expected)}')
   local1,box1=audit_one_bank(Y.astype(np.float64),F,changed_faces,bankT,bankids,M,off,pos,exact,audit,body,bone)
   after=(before-expected)|local1;after_by[(body,bone)]=after
   nr=after-before;rr=before-after
   if nr or rr:bank_counts.append({'body':body,'stable_bone_id':bone,'raw_before':len(before),'raw_after':len(after),'new_raw':sorted([list(z) for z in nr]),'removed_raw':sorted([list(z) for z in rr])})
   new_global|={(body,bone,f,t) for f,t in nr};removed_global|={(body,bone,f,t) for f,t in rr}
   new_global_mapped|={(body,bone,p,t) for p,t in mapped(nr,ip,pa)};removed_global_mapped|={(body,bone,p,t) for p,t in mapped(rr,ip,pa)}
  allbank={'bank_count':len(stored_by),'changed_banks':bank_counts,'new_raw_total':len(new_global),'removed_raw_total':len(removed_global),'new_mapped_total':len(new_global_mapped),'removed_mapped_total':len(removed_global_mapped),'new_raw_first_64':sorted([list(z) for z in new_global])[:64],'removed_raw_first_64':sorted([list(z) for z in removed_global])[:64],'new_mapped_first_64':sorted([list(z) for z in new_global_mapped])[:64],'removed_mapped_first_64':sorted([list(z) for z in removed_global_mapped])[:64]}
 report={'schema':'numi.human.b74-104.contact-feature-fit-806.v1','status':'ALL_BANK_LOCAL_AUDIT_COMPLETE' if allbank else 'REJECTED_NO_LOCAL_TARGET_IDENTITY_REDUCTION','geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'baseline_candidate_sha256':PIN,'baseline_audit159_sha256':sha(AUD),'baseline_audit130_sha256':sha(SUM),'layer_bundle_sha256':sha(LAYER),'bank_input_pins':{p.name:sha(p) for p in [BANKDIR/'exact-emitted.triangles.csv',BANKDIR/'exact-emitted.bodies.csv',BANKDIR/'exact-emitted.nodes.csv',BANKDIR/'all-bank-containment.csv.banks.csv']},'target_bank':[BODY,BONE],'baseline_raw_pairs':len(rawbase),'baseline_raw_pairs_list':sorted([list(z) for z in rawbase]),'baseline_mapped_identities':len(mapbase),'baseline_mapped_identity_list':sorted([list(z) for z in mapbase]),'baseline_sampled_inside_nodes':0,'source_chart_rows':sorted({int(x) for f in SEED_FACES for x in pa[ip[f]:ip[f+1]]}),'seed_face_rows':SEED_FACES.tolist(),'seed_patch_nodes':sorted(seed_nodes),'exact_endpoint_count':len(feature_rows),'exact_endpoint_features':feature_meta,'solver':solve,'backtracks':trial_rows,'centroid_point_query':centroid_report,'all_bank_changed_face_audit':allbank,'candidate_emitted':False,'candidate_path':None,'candidate_sha256':None,'elapsed_s':time.monotonic()-t}
 useful=bool(selected and allbank and allbank['new_raw_total']==0 and allbank['new_mapped_total']==0 and selected[3]['self_touching_changed_faces']['count']==0 and centroid_report is not None)
 if useful:
  Y,changed_nodes,changed_faces,tr=selected
  payload={k:v for k,v in P.items()};payload['points_world_m']=Y;candidate=OUT/'candidate-private.npz'
  with candidate.open('xb') as f:np.savez_compressed(f,**payload)
  report.update(status='PRIVATE_LOCAL_IMPROVEMENT_PENDING_COMPLETE_BODY_AUDITS',candidate_emitted=True,candidate_path=str(candidate),candidate_sha256=sha(candidate))
  np.save(OUT/'trial-points-world-f32.npy',Y,allow_pickle=False)
 else:
  if first:np.save(OUT/'trial-points-world-f32.npy',first[0],allow_pickle=False)
  if not selected:report['status']='REJECTED_NO_LOCAL_TARGET_IDENTITY_REDUCTION_OR_GATE'
  elif allbank and (allbank['new_raw_total'] or allbank['new_mapped_total']):report['status']='REJECTED_NEW_NEIGHBOR_BANK_CROSSINGS'
  else:report['status']='REJECTED_ALL_BANK_OR_SELF_GATE'
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'solver':solve,'trial_summary':[{'alpha':x['alpha'],'shape':x['shape_pass'],'layer':x['layer_gate']['pass'],'raw_after':(x.get('target_exact') or {}).get('raw_after'),'mapped_after':(x.get('target_exact') or {}).get('mapped_after'),'new_mapped':len((x.get('target_exact') or {}).get('new_mapped',[])),'removed_mapped':len((x.get('target_exact') or {}).get('removed_mapped',[])),'self':(x.get('self_touching_changed_faces') or {}).get('count')} for x in trial_rows],'allbank':None if allbank is None else {'banks':allbank['bank_count'],'new_raw':allbank['new_raw_total'],'new_mapped':allbank['new_mapped_total'],'removed_raw':allbank['removed_raw_total'],'removed_mapped':allbank['removed_mapped_total']},'centroids':centroid_report},indent=2))
if __name__=='__main__':main()
