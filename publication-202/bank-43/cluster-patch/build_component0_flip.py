#!/usr/bin/env python3
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027/b43-16/attempt-025-witness-cluster-topology-001/component-00-oneflip'
BASE=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz'
PARENT=S/'fit-from-663-001/seed-retry-663-001/fraction-small-001/candidate.npz'
AUD=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/audit-159/report.json'
SELF=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/audit-130/summary.json'
SELFP=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/audit-130/surface-self-pairs.npy'
OWNER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py')
GUARD=S/'edge-refinement-661-threebank-001/refresh_guards_multi_parent_generation002.py'
BASE_SHA='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db'
PARENT_SHA='beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35'
AUD_SHA='2e870a160a1514e7a5621ceb504845303e984871532f60278bac12b6bbf817a8'
SELF_SHA='c1aa0b43b19cfa860ee7c92400ecd396d16f5442c0918bdb2ef4c434bfff6a0d'
SELFP_SHA='55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de'
OWNER_SHA='f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764'
GUARD_SHA='90bafb6a51e9d7a2dd9b49570fcd977423616e18e992c2d4a4306feea5bd5eab'
SEED=[19114,19126,19133];DETAIL=[8265]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def area(points,tri):
 p=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64)
 return float(.5*np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0])))
def min_angle(points,tri):
 p=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64)
 e=[np.linalg.norm(p[1]-p[0]),np.linalg.norm(p[2]-p[1]),np.linalg.norm(p[0]-p[2])]
 cs=[(e[0]**2+e[2]**2-e[1]**2)/(2*e[0]*e[2]),(e[0]**2+e[1]**2-e[2]**2)/(2*e[0]*e[1]),(e[1]**2+e[2]**2-e[0]**2)/(2*e[1]*e[2])]
 return float(min(np.degrees(np.arccos(np.clip(v,-1,1))) for v in cs))
def aspect(points,tri):
 p=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64);e=[np.dot(p[1]-p[0],p[1]-p[0]),np.dot(p[2]-p[1],p[2]-p[1]),np.dot(p[0]-p[2],p[0]-p[2])];a=area(points,tri)
 return float(max(e)/(4*np.sqrt(3)*a)) if a>0 else float('inf')
def bary_map(point,tri_points):
 p=np.asarray(tri_points,dtype=np.float64);x=np.asarray(point,dtype=np.float64);M=np.column_stack((p[1]-p[0],p[2]-p[0]));uv=np.linalg.lstsq(M,x-p[0],rcond=None)[0];w=np.asarray([1.-uv.sum(),uv[0],uv[1]]);return w,float(np.linalg.norm(w@p-x))
def main():
 started=time.time()
 if W.exists() and any(W.iterdir()):raise FileExistsError(W)
 pins={'a69':sha(BASE),'immediate_parent':sha(PARENT),'audit159':sha(AUD),'audit130':sha(SELF),'self_pairs':sha(SELFP),'owner402':sha(OWNER),'generation002_guard':sha(GUARD)}
 expected={'a69':BASE_SHA,'immediate_parent':PARENT_SHA,'audit159':AUD_SHA,'audit130':SELF_SHA,'self_pairs':SELFP_SHA,'owner402':OWNER_SHA,'generation002_guard':GUARD_SHA}
 if pins!=expected:raise ValueError('pinned input mismatch '+str(pins))
 audit=json.loads(AUD.read_text());sd=json.loads(SELF.read_text());p0=np.load(SELFP,allow_pickle=False)
 if sd.get('candidate_sha256')!=BASE_SHA or not sd.get('complete_surface_audit') or sd.get('exact_PLC_surface_pairs')!=0 or p0.shape!=(0,2):raise ValueError('a69 baseline self audit invalid')
 with np.load(BASE,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
 with np.load(PARENT,allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
 owner=mod('owner402_component0',OWNER);guard=mod('generation002_component0',GUARD)
 owner.validate_candidate(c)
 details=np.asarray(c['parent_fine_detail_face_row'])
 if not all(int(details[r])==8265 for r in SEED):raise ValueError('component 0 detail identity changed')
 out,rep=owner.propose_patch_flips(c,DETAIL,SEED,max_flips=1,max_rounds=1,min_angle_gain_deg=.5,min_normal_dot=0.0)
 if int(rep.get('accepted_flip_count',0))!=1:
  outdir=W;outdir.mkdir(parents=True,exist_ok=True)
  r={'schema':'numi.human.b43-16.component0.topology-repair-build.v1','status':'NO_ELIGIBLE_OWNER_FLIP','input_pins':pins,'seed_face_rows':SEED,'source_detail_rows':DETAIL,'owner_report':rep,'geometry_admitted':False,'physical_steps':0}
  p=outdir/'report.json';p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,indent=2));return
 op=rep['operations'][0];changed=sorted(map(int,op['face_rows']))
 if not set(changed)<=set(SEED):raise ValueError('chosen operation escaped the clear 3-face component boundary')
 if not np.array_equal(out['points_world_m'],c['points_world_m']) or out['points_world_m'].dtype!=np.float32:raise ValueError('topology owner changed coordinates')
 observed=np.flatnonzero(np.any(out['source_O_faces']!=c['source_O_faces'],axis=1)).tolist()
 if observed!=changed:raise ValueError('topology diff does not equal operation rows')
 if not np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']):raise ValueError('matched 037 topology stale')
 # Refresh single-parent affine rows from immutable matched reference geometry and immediate parent.
 f=np.asarray(out['source_O_faces'],np.int64);ref=np.asarray(out['area_reference_037_points_world_m_f64'],np.float64)
 bary=np.array(out['refinement_face_parent_barycentric'],copy=True);scalar=np.asarray(out['refinement_face_parent_face_row'],np.int64)
 parent_faces=np.asarray(parent['area_reference_037_source_O_faces'],np.int64);parent_ref=np.asarray(parent['area_reference_037_points_world_m_f64'],np.float64)
 parip=out['refinement_face_parent_ancestry_indptr'];parrows=out['refinement_face_parent_ancestry_face_rows'];maps=[]
 for row in changed:
  ances=list(map(int,parrows[int(parip[row]):int(parip[row+1])]))
  if not ances:raise ValueError('empty immediate parent CSR after flip')
  if len(ances)==1:
   pr=ances[0]
   if int(scalar[row])!=pr:raise ValueError('single-parent scalar disagrees with CSR')
   ptri=parent_ref[parent_faces[pr]];weights=[];err=[]
   for node in f[row]:
    w,e=bary_map(ref[int(node)],ptri);weights.append(w);err.append(e)
   Wg=np.asarray(weights)
   if max(err)>1e-12 or np.min(Wg)<-2e-8 or np.max(Wg)>1+2e-8 or np.max(np.abs(Wg.sum(1)-1))>2e-8:raise ValueError('refreshed barycentric row violates map/reconstruction bounds')
   bary[row]=Wg;maps.append({'face_row':row,'mode':'single_parent_recomputed','parent':pr,'max_reconstruction_error_m':max(err),'min_weight':float(Wg.min()),'max_weight':float(Wg.max())})
  else:
   if int(scalar[row])!=-1 or not np.isnan(bary[row]).all():raise ValueError('multi-parent row must carry -1/NaN sentinel')
   maps.append({'face_row':row,'mode':'multi_parent_union_csr_nan_scalar_map','parents':ances})
 out['refinement_face_parent_barycentric']=bary
 lineage=guard.validate_ancestry(out,parent,audit)
 oldf=np.asarray(c['source_O_faces'],np.int64);oldp=np.asarray(c['points_world_m'],np.float32);newf=np.asarray(out['source_O_faces'],np.int64);refp=np.asarray(out['area_reference_037_points_world_m_f64'],np.float64)
 rr=[]
 old_area=sum(area(oldp,oldf[r]) for r in changed);old_ref_area=sum(area(refp,oldf[r]) for r in changed)
 shape_ok=True
 for row in changed:
  v={'face_row':row,'triangle':newf[row].tolist()}
  for label,points,den in [('current_f32',oldp,old_area),('immutable_037',refp,old_ref_area)]:
   a=area(points,newf[row]);ang=min_angle(points,newf[row]);asp=aspect(points,newf[row]);ratio=a/den if den else 0.
   v[label]={'area_m2':a,'area_fraction_old_patch':ratio,'min_angle_deg':ang,'aspect':asp}
   shape_ok &= bool(a>0 and ratio>=.1 and ang>=.5 and asp<=50)
  rr.append(v)
 if not shape_ok:raise ValueError('changed-face shape gate failed '+json.dumps(rr))
 outdir=W;outdir.mkdir(parents=True,exist_ok=True)
 cand=outdir/'candidate-private.npz';np.savez_compressed(cand,**out)
 r={'schema':'numi.human.b43-16.component0.topology-repair-build.v1','status':'PRIVATE_COMPONENT0_ONE_FLIP_PENDING_EXACT_GATES',
  'input_pins':pins,'input_a69_sha256':BASE_SHA,'immediate_parent_sha256':PARENT_SHA,
  'component_seed_faces':SEED,'clear_fixed_boundary_node_count':5,'clear_fixed_boundary_edges_pass_43_16':True,
  'operation':op,'changed_face_rows':changed,'changed_face_count':len(changed),
  'coordinates_bitwise_identical_to_a69':True,'faces_unchanged_outside_operation_rows':True,'matched_037_connectivity_synchronized':True,
  'ancestry_refresh':maps,'generation002_ancestry_guard':lineage,
  'shape_gate':{'pass':True,'minimum_angle_deg':.5,'maximum_aspect':50,'minimum_area_fraction_old_patch':.1,'changed_rows':rr},
  'source_detail_ancestry':'all input component faces are source/detail 8265; owner regenerated face-source/detail unions',
  'candidate_path':str(cand),'candidate_sha256':sha(cand),'elapsed_seconds':time.time()-started,
  'geometry_admitted':False,'physical_steps':0,
  'limitations':['fixed-coordinate topology operation only; full exact self, target bank, layer transfer and all-bank checks remain gating','multi-parent categorical ancestry is preserved by CSR union; continuous material transfer still requires owner reconciliation']}
 rp=outdir/'report.json';rp.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');r['report_path']=str(rp);r['report_sha256']=sha(rp)
 print(json.dumps({k:r[k] for k in ['status','operation','changed_face_rows','ancestry_refresh','generation002_ancestry_guard','shape_gate','candidate_sha256','report_sha256']},indent=2,default=lambda v:v.item() if hasattr(v,'item') else v.tolist() if hasattr(v,'tolist') else str(v)))
if __name__=='__main__':main()
