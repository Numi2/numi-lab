#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy.optimize import linprog
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';B=W/'combined-d071-b109-b61-001'
D031=W/'b43-16/attempt-031-contact-feature-fit-001/projection-001'
OUT=W/'b43-16/attempt-032-targeted-contact-layer-constraint-001'
BASE=B/'candidate.npz';AUD=B/'audit-159/report.json';SUM=B/'audit-130/summary.json'
FIT=D031.parent/'fit_contact_features.py';PROJ=D031/'project_contacts.py'
SCREEN=W/'b43-16/attempt-030-radius-aware-position-001/rebase-024-on-806/deep-backtrack-001/screen_rebased_direction.py'
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
LAYER=B/'skin-fat-layers-541.npz'
TRANSFER=W/'b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py'
BASEPIN='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
DIRPIN='3a871195ee0203184b3f3b18ea02fd7b713f76214e0a29cc625c2e04dfedb37d'
ALPHA=.0625; CONTACT_MARGIN_MM=.001; FD_H_MM=.0005; MAX_CORRECTION_MM=.015
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def volume_row(tf,outer,inner,faces,field):
 return np.asarray(tf.face_flags(outer,inner,faces,field)['volumes'],dtype=np.float64).reshape(-1)
def main():
 t=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
 if sha(BASE)!=BASEPIN or sha(D031/'contact-feature-direction.npz')!=DIRPIN:raise ValueError('baseline or diagnostic direction pin mismatch')
 proj=mod('a032_proj',PROJ);fitter=mod('a032_fit',FIT);screen=mod('a032_screen',SCREEN);audit=mod('a032_audit',AUD159);helper=mod('a032_helper',HELPER);tf=mod('a032_transfer',TRANSFER);exact=audit.exact_module()
 with np.load(BASE,allow_pickle=False) as z: P={k:z[k].copy() for k in z.files}
 X=np.asarray(P['points_world_m'],dtype=np.float32);F=np.asarray(P['source_O_faces'],dtype=np.int64)
 with np.load(D031/'contact-feature-direction.npz',allow_pickle=False) as z:support=np.asarray(z['support_nodes'],dtype=np.int64);dd=np.asarray(z['direction_delta_mm'],dtype=np.float64)
 Y0=X.copy();Y0[support]=(X[support].astype(np.float64)+(ALPHA*dd)/1000.).astype(np.float32)
 if not np.array_equal((X[support].astype(np.float64)+np.zeros_like(dd)/1000.).astype(np.float32).view(np.uint32),X[support].view(np.uint32)):raise ValueError('zero-motion control failed')
 doc=json.loads(SUM.read_text());ad=json.loads(AUD.read_text())
 witnesses=[r for r in doc['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id']))==(43,16) and r.get('classification')=='inside' and int(r.get('ambiguous',0))==0]
 inside_nodes=np.asarray(sorted(int(r['query_row']) for r in witnesses),dtype=np.int64)
 with np.load(LAYER,allow_pickle=False) as z:L={k:z[k].copy() for k in z.files}
 if not np.array_equal(L['outer_points_f32'].view(np.uint32),X.view(np.uint32)) or not np.array_equal(L['outer_faces'],F[:,[0,2,1]]):raise ValueError('806 layer pin mismatch')
 O0=np.asarray(L['outer_points_f64'],dtype=np.float64);I0=np.asarray(L['inner_points_f64'],dtype=np.float64);D0=np.asarray(L['deep_points_f64'],dtype=np.float64)
 skinfield=L['skin_thickness_transferred_m'];fatfield=L['fat_thickness_transferred_m'];rev=F[:,[0,2,1]]
 # The removed pair is kept absent by freezing its three candidate vertices.
 removed_face=19239;frozen=set(map(int,F[removed_face]))
 anchors=set(map(int,P['typed_coarse_anchor_output_rows_018']))
 contact_faces=[19203,19214,19221]
 layer_faces=[19121,19873]
 contact_vertices=set(map(int,F[contact_faces].reshape(-1)))
 layer_vertices=set(map(int,F[layer_faces].reshape(-1)))
 variable_nodes=sorted((contact_vertices|layer_vertices)-frozen-anchors)
 nodepos={n:i for i,n in enumerate(variable_nodes)}
 if len(variable_nodes)==0:raise ValueError('no movable local nodes')
 # Reconstruct the six exact segment endpoints for the three newly added pairs.
 bankT,bids,_,_,M,offset,pos,_,_,_=helper.read_bank_inputs(BANKDIR,E)
 tidx={int(bids[i,3]):int(i) for i in np.flatnonzero((bids[:,0]==43)&(bids[:,1]==16))}
 fitter=mod('a032_fit_again',FIT)
 Xlocal=(Y0.astype(np.float64)-pos[43])@M[43].T-offset[43]
 contacts=[];feature_log=[]
 for sf,bt in [(19203,140),(19214,140),(19221,140)]:
  if bt not in tidx:raise ValueError(f'missing target bone triangle {bt}')
  feat=fitter.pair_contact_features(Xlocal[F[sf]],bankT[tidx[bt]])
  if len(feat)!=2:raise ValueError(f'expected two actual endpoints for {sf}/{bt}, got {len(feat)}')
  tri=bankT[tidx[bt]];nl=np.cross(tri[1]-tri[0],tri[2]-tri[0]);nl/=np.linalg.norm(nl);nw=nl@M[43];nw/=np.linalg.norm(nw)
  for endpoint,(pnt,bary) in enumerate(feat):
   row=np.zeros(3*len(variable_nodes),dtype=np.float64)
   for k,node in enumerate(F[sf]):
    j=nodepos.get(int(node))
    if j is not None:row[3*j:3*j+3]+=float(bary[k])*nw
   contacts.append(row);feature_log.append({'skin_face':sf,'bone_triangle':bt,'endpoint':endpoint,'skin_barycentric':np.asarray(bary).tolist(),'outward_normal_world':nw.tolist(),'movable_weight_sum':float(sum(bary[k] for k,n in enumerate(F[sf]) if int(n) in nodepos))})
 C=np.asarray(contacts,dtype=np.float64)
 # Current prism volumes and finite-difference derivatives for only the two witnessed cells.
 delta0=Y0.astype(np.float64)-X.astype(np.float64);Icur=I0+delta0;Dcur=D0+delta0;Ocur=Y0.astype(np.float32).astype(np.float64)
 faces1=rev[np.asarray([19121,19873],dtype=np.int64)]
 vol_skin0=volume_row(tf,O0,I0,rev[[19121]],skinfield)
 vol_fat0=volume_row(tf,I0,D0,rev[[19873]],fatfield)
 base_skin=float(vol_skin0[1]);base_fat=float(vol_fat0[1])
 def cell_vols(Y):
  delt=Y.astype(np.float64)-X.astype(np.float64);I=I0+delt;D=D0+delt;O=Y.astype(np.float32).astype(np.float64)
  return float(volume_row(tf,O,I,rev[[19121]],skinfield)[1]),float(volume_row(tf,I,D,rev[[19873]],fatfield)[1])
 vcur=np.asarray(cell_vols(Y0),dtype=np.float64)
 grad=np.zeros((2,3*len(variable_nodes)),dtype=np.float64)
 for j,node in enumerate(variable_nodes):
  for a in range(3):
   yp=Y0.copy();ym=Y0.copy()
   yp[node]=(Y0[node].astype(np.float64)+np.eye(3)[a]*FD_H_MM/1000.).astype(np.float32)
   ym[node]=(Y0[node].astype(np.float64)-np.eye(3)[a]*FD_H_MM/1000.).astype(np.float32)
   grad[:,3*j+a]=(np.asarray(cell_vols(yp))-np.asarray(cell_vols(ym)))/(2*FD_H_MM)
 thresholds=np.asarray([max(.2*base_skin,1e-18),base_fat],dtype=np.float64)
 # One non-regression row for each of the 43 target point nodes; rows with frozen nodes are identically zero.
 qrows=list(csv.DictReader((D031/'inside-query-43/pairs.csv').open()))
 pnorm={}
 for qi,node in enumerate(inside_nodes):
  r=next(x for x in qrows if int(x['query_row'])==qi and int(x['bank_body'])==43 and int(x['bank_stable_id'])==16)
  triid=int(r['nearest_triangle']);tri=bankT[tidx[triid]]
  nl=np.cross(tri[1]-tri[0],tri[2]-tri[0]);nl/=np.linalg.norm(nl);nw=nl@M[43];nw/=np.linalg.norm(nw)
  pnorm[int(node)]=nw
 inrows=[]
 for node in inside_nodes:
  row=np.zeros(3*len(variable_nodes),dtype=np.float64);j=nodepos.get(int(node))
  if j is not None:row[3*j:3*j+3]=pnorm[int(node)]
  inrows.append(row)
 IROW=np.asarray(inrows,dtype=np.float64)
 # L1 minimum correction, with hard contact, cell, point, and trust bounds.
 nv=C.shape[1];nb=nv
 Aub=[];bub=[]
 for row in C:Aub.append(np.r_[-row,np.zeros(nb)]);bub.append(-CONTACT_MARGIN_MM)
 for k in range(2):Aub.append(np.r_[-grad[k],np.zeros(nb)]);bub.append(float(vcur[k]-thresholds[k]))
 for row in IROW:Aub.append(np.r_[-row,np.zeros(nb)]);bub.append(0.0)
 for j in range(nv):
  row=np.zeros(2*nv);row[j]=1;row[nv+j]=-1;Aub.append(row);bub.append(0.0)
  row=np.zeros(2*nv);row[j]=-1;row[nv+j]=-1;Aub.append(row);bub.append(0.0)
 res=linprog(np.r_[np.zeros(nv),np.ones(nv)],A_ub=np.asarray(Aub),b_ub=np.asarray(bub),bounds=[(-MAX_CORRECTION_MM,MAX_CORRECTION_MM)]*nv+[(0,MAX_CORRECTION_MM)]*nv,method='highs',options={'maxiter':1500})
 if not res.success: 
  report={'schema':'numi.human.b43-16.targeted-contact-layer-correction.v1','status':'REJECTED_LOCAL_CONSTRAINTS_INFEASIBLE','baseline_sha256':BASEPIN,'diagnostic_direction_sha256':DIRPIN,'diagnostic_alpha':ALPHA,'new_raw_pairs_targeted':[[19203,140],[19214,140],[19221,140]],'removed_raw_pair_preservation':[19239,144],'frozen_removed_face_nodes':sorted(frozen),'variable_nodes':variable_nodes,'variable_count':nv,'contact_endpoint_rows':len(C),'contact_margin_mm':CONTACT_MARGIN_MM,'contact_feature_rows':feature_log,'inside_nodes':inside_nodes.tolist(),'inside_nonregression_rows':len(IROW),'layer_cells':{'skin_face_19121_slot_1_baseline_mm3':base_skin*1e9,'fat_face_19873_slot_1_baseline_mm3':base_fat*1e9,'current_trial_mm3':[float(x*1e9) for x in vcur],'required_trial_mm3':[float(x*1e9) for x in thresholds],'gradient_rows_per_mm':[g.tolist() for g in grad]},'solver':{'method':'bounded_sparse-input_L1_LP_highs','success':False,'status':int(res.status),'message':res.message,'iterations':int(getattr(res,'nit',-1)),'variables':int(2*nv),'constraints':len(Aub)},'candidate_emitted':False,'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0}
  (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'status':report['status'],'message':res.message,'nodes':variable_nodes,'constraints':len(Aub)},indent=2));return
 x=np.asarray(res.x[:nv],dtype=np.float64)
 # Exact float32 emitter: first candidate position, then correction, one final cast.
 Y=Y0.copy()
 for j,node in enumerate(variable_nodes):Y[node]=(Y0[node].astype(np.float64)+x[3*j:3*j+3]/1000.).astype(np.float32)
 totalchanged=np.flatnonzero(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64)
 changedfaces=np.flatnonzero(np.any(np.isin(F,totalchanged),axis=1)).astype(np.int64)
 shapep,sm,ag=fitter.shape_pass(X,Y,F,changedfaces,screen,mod('a032_area',Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')),P['area_reference_037_points_world_m_f64'])
 lg=screen.layer_check(X,Y,F,changedfaces,L,tf)
 vafter=np.asarray(cell_vols(Y),dtype=np.float64)
 # Actual target exact pair delta and mapped identities.
 rawbase={(int(f),int(bt)) for f,bt in next(r for r in ad['banks'] if (int(r['body']),int(r['stable_bone_id']))==(43,16))['result']['triangle_pairs']}
 ip=np.asarray(P['refinement_face_parent_ancestry_indptr'],dtype=np.int64);pa=np.asarray(P['refinement_face_parent_ancestry_face_rows'],dtype=np.int64)
 mapbase=screen.map_pairs(rawbase,ip,pa)
 baseline_local,box0=screen.audit_target(X.astype(np.float64),F,changedfaces,bankT,bids,M,offset,pos,exact,audit)
 expected={(f,bt) for f,bt in rawbase if f in set(map(int,changedfaces))}
 if baseline_local!=expected:raise ValueError('zero-motion exact target parity failure')
 candlocal,box1=screen.audit_target(Y.astype(np.float64),F,changedfaces,bankT,bids,M,offset,pos,exact,audit)
 rawafter=(rawbase-expected)|candlocal;mapafter=screen.map_pairs(rawafter,ip,pa)
 rawnew=rawafter-rawbase;rawremoved=rawbase-rawafter;mapnew=mapafter-mapbase;mapremoved=mapbase-mapafter
 records,_=exact.float32_surface_records(Y,F,source_seam_groups=())
 selfres=exact._audit_self_pairs_touching_faces(records,set(map(int,changedfaces)))
 selfpairs={(int(a),int(b)) for a,b in (selfres.get('triangle_pairs') or [])}
 # Candidate point classification/depth for all 43 nodes against all 185 banks.
 qpoints,qpairs,qnrows,pnpz=proj.point_query_nodes(Y,inside_nodes,OUT/'inside-query-43')
 if any(int(r['total_bone_banks'])!=185 or int(r['ambiguous_banks']) or int(r['invalid_banks']) for r in qnrows):raise ValueError('incomplete/ambiguous point query')
 insidepairs={(int(inside_nodes[int(r['query_row'])]),int(r['bank_body']),int(r['bank_stable_id'])) for r in qpairs if r['classification']=='inside'}
 basinside={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])) for r in doc['inside_witnesses'] if int(r['query_row']) in set(map(int,inside_nodes))}
 targetrows=[r for r in qpairs if (int(inside_nodes[int(r['query_row'])]),int(r['bank_body']),int(r['bank_stable_id']))[1:]==(43,16) and r['classification']=='inside' and r.get('signed_face_distance_m') not in ('','None')]
 depths=[abs(float(r['signed_face_distance_m']))*1000 for r in targetrows]
 beforedepth=[abs(float(r['signed_face_distance_m']))*1000 for r in witnesses]
 np.savez_compressed(OUT/'correction-vector.npz',variable_nodes=np.asarray(variable_nodes,dtype=np.int64),correction_delta_mm=x.reshape(-1,3),contact_rows=C,cell_gradients_per_mm=grad)
 report={'schema':'numi.human.b43-16.targeted-contact-layer-correction.v1','status':'LOCAL_EXACT_SCREEN_COMPLETE','baseline_sha256':BASEPIN,'diagnostic_direction_sha256':DIRPIN,'diagnostic_alpha':ALPHA,'candidate_sha256_private':None,'new_raw_pairs_targeted':[[19203,140],[19214,140],[19221,140]],'removed_raw_pair_preservation':[19239,144],'frozen_removed_face_nodes':sorted(frozen),'variable_nodes':variable_nodes,'variable_count':nv,'contact_endpoint_rows':len(C),'contact_margin_mm':CONTACT_MARGIN_MM,'contact_feature_rows':feature_log,'inside_nodes':inside_nodes.tolist(),'inside_nonregression_rows':len(IROW),'solver':{'method':'bounded_L1_LP_highs','success':True,'iterations':int(res.nit),'variables':int(2*nv),'constraints':len(Aub),'max_abs_correction_mm':float(np.max(np.abs(x),initial=0)),'l1_objective_mm':float(np.sum(np.abs(x)))},'cell_model':{'finite_difference_h_mm':FD_H_MM,'baseline_mm3':[base_skin*1e9,base_fat*1e9],'current_trial_mm3':[float(v*1e9) for v in vcur],'linearized_required_mm3':[float(v*1e9) for v in thresholds],'linearized_predicted_mm3':[float((vcur[k]+grad[k]@x)*1e9) for k in range(2)],'emitted_actual_mm3':[float(v*1e9) for v in vafter]},'shape':{'pass':bool(shapep and ag['pass']),'metrics':sm,'area_gate':ag},'layer_gate':lg,'exact_43_16':{'raw_before':len(rawbase),'raw_after':len(rawafter),'mapped_before':len(mapbase),'mapped_after':len(mapafter),'new_raw':sorted([list(y) for y in rawnew]),'removed_raw':sorted([list(y) for y in rawremoved]),'new_mapped':sorted([list(y) for y in mapnew]),'removed_mapped':sorted([list(y) for y in mapremoved]),'preserved_removed_identity':(19004,144) not in mapafter,'local_before':len(baseline_local),'local_after':len(candlocal),'aabb_before':box0,'aabb_after':box1},'exact_self_touching_changed_faces':{'count':len(selfpairs),'pairs':sorted([list(y) for y in selfpairs])},'inside_query':{'nodes':inside_nodes.tolist(),'target_inside_before':sum((n,43,16) in basinside for n in inside_nodes),'target_inside_after':sum((n,43,16) in insidepairs for n in inside_nodes),'allbank_inside_pair_count_before':len(basinside),'allbank_inside_pair_count_after':len(insidepairs),'new_inside_pairs':sorted([list(y) for y in insidepairs-basinside]),'removed_inside_pairs':sorted([list(y) for y in basinside-insidepairs]),'depth_before_mm':{'min':float(np.min(beforedepth)),'median':float(np.median(beforedepth)),'max':float(np.max(beforedepth))},'depth_after_mm':{'min':float(np.min(depths)) if depths else None,'median':float(np.median(depths)) if depths else None,'max':float(np.max(depths)) if depths else None},'invalid_or_ambiguous':0,'point_npz_sha256':sha(pnpz)},'full_neighbor_bank_face_audit':'NOT_RUN unless all local gates pass','candidate_emitted':False,'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'elapsed_s':time.monotonic()-t}
 useful=bool(shapep and ag['pass'] and lg['pass'] and not rawnew and not mapnew and not selfpairs and (19004,144) not in mapafter)
 if useful:
  payload={k:v for k,v in P.items()};payload['points_world_m']=Y
  candidate=OUT/'candidate-private.npz'
  with candidate.open('xb') as f:np.savez_compressed(f,**payload)
  report['candidate_emitted']=True;report['candidate_path']=str(candidate);report['candidate_sha256']=sha(candidate);report['status']='LOCAL_GATES_PASS_NEIGHBOR_BANK_AUDIT_REQUIRED';report['candidate_sha256_private']=sha(candidate)
 else:
  report['status']='REJECTED_LOCAL_GATES_OR_NO_NET_IDENTITY_IMPROVEMENT'
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'solve_iter':res.nit,'max_move_mm':report['solver']['max_abs_correction_mm'],'shape':report['shape']['pass'],'layer':lg['pass'],'raw':[len(rawbase),len(rawafter)],'mapped':[len(mapbase),len(mapafter)],'new_mapped':report['exact_43_16']['new_mapped'],'removed_mapped':report['exact_43_16']['removed_mapped'],'preserved_removed':report['exact_43_16']['preserved_removed_identity'],'self':len(selfpairs),'inside':[report['inside_query']['target_inside_before'],report['inside_query']['target_inside_after']],'depth_after':report['inside_query']['depth_after_mm']},indent=2))
if __name__=='__main__':main()
