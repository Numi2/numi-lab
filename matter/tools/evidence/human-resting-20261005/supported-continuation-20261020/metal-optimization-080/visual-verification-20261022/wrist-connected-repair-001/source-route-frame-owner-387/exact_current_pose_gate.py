from pathlib import Path
from fractions import Fraction
import hashlib, json, os, runpy, sys
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=E/'source-route-frame-owner-387'; CACHE=E/'registered-surrounding-tissue-native013-191'
D=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003'
BONE_SCRIPT=E/'fine-source-wholehand-repair-003/static-gate-003/audit_wholehand_candidate_003.py'
BONE_MAN=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
EXACT_SRC=E/'exact-plane-filter-owner-264/src'; sys.path.insert(0,str(EXACT_SRC))
from numilab_human import cardiac_cavity_intersections as ci
WANTED={131,132,133,134,137,138,141,142}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)
def exact_records(V,F):
 F=np.asarray(F,np.int64); pts=[]
 for row in np.asarray(V,np.float32):
  key=ci.float32_point_lattice_key(row);pts.append(tuple(Fraction(int(k),1<<149) for k in key))
 return ci._records(pts,F.tolist())
def bounds(V):
 V=np.asarray(V,np.float64);return V.min(0),V.max(0)
def overlaps(a,b):return bool(np.all(a[1]>=b[0]) and np.all(b[1]>=a[0]))
def topology(F):
 uses={}
 for tri in np.asarray(F,np.int64):
  for a,b in ((int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))):
   uses.setdefault((min(a,b),max(a,b)),[]).append((a,b))
 return {'unique_edges':len(uses),'boundary_edges':sum(len(v)==1 for v in uses.values()),'nonmanifold_edges':sum(len(v)>2 for v in uses.values()),'same_direction_two_face_edges':sum(len(v)==2 and v[0]==v[1] for v in uses.values()),'max_incidence':max(map(len,uses.values()))}
def pair_report(label,br,cr,extra=None):
 bp={tuple(map(int,x)) for x in br['triangle_pairs']};cp={tuple(map(int,x)) for x in cr['triangle_pairs']}
 return dict(label=label,baseline_pair_count=len(bp),candidate_pair_count=len(cp),new_pair_count=len(cp-bp),removed_pair_count=len(bp-cp),baseline_aabb_candidates=br['aabb_candidate_pairs'],candidate_aabb_candidates=cr['aabb_candidate_pairs'],baseline_allowed_shared=br['allowed_shared_vertex_or_edge_pairs'],candidate_allowed_shared=cr['allowed_shared_vertex_or_edge_pairs'],new_pairs=[list(x) for x in sorted(cp-bp)[:200]],removed_pairs=[list(x) for x in sorted(bp-cp)[:200]],witness_limit=200,**(extra or {}))
def audit(a,b,same=False):return ci._audit_pair(a,b,same_surface=same)
print('Load frozen exact-191 current-pose emission preflight',flush=True)
import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):env=runpy.run_path(str(OUT/'current_pose_preflight.py'),run_name='preflight')
assert env['report']['baseline_match']['per_bank_submicron'], 'invalid baseline-to-191 match'
base,cand=env['base'],env['cand']; bank_by_stable=env['bank_by_stable']
base_rec={s:exact_records(base[s][0],base[s][1]) for s in sorted(WANTED)}
cand_rec={s:exact_records(cand[s][0],cand[s][1]) for s in sorted(WANTED)}
stage={'schema':'numi.human.nhtiss.route-weight-counterfactual.exact-current-pose.v1','candidate386_receipt_sha256':sha(E/'source-route-frame-owner-386/candidate-receipt.json'),'candidate_payload_sha256':sha(E/'source-route-frame-owner-386/corrected-route-weights-candidate.nhtissue'),'emission_preflight_sha256':sha(OUT/'current-pose-emission-preflight-001.json'),'identity003_pose_sha256':sha(D/'registered-skin-poses.json'),'registered191_inventory_sha256':sha(CACHE/'inventory.json'),'registered191_cache_sha256':sha(CACHE/'registered-tissue-banks.npz'),'exact_predicate_sha256':sha(EXACT_SRC/'numilab_human/cardiac_cavity_intersections.py'),'pose':'identity-003 current; exact 191 emission order','baseline_match':env['report']['baseline_match'],'topology':{},'self_pairs':[],'selected_bank_pairs':[],'registered_surrounding_pairs':[],'bone_pairs':[],'limitations':['Exact intersections are evaluated on Float32 current-pose outputs. Pair counts are compared against the matched 191 baseline. No physical/dynamic or tissue-attachment qualification is implied.']}
def save():write(OUT/'exact-current-pose-gate-001.partial.json',stage)
for sid in sorted(WANTED):
 bv,bf,_=base[sid];cv,cf,_=cand[sid]
 assert np.array_equal(bf,cf)
 def degenerates(V,F):
  t=np.asarray(V,np.float64)[np.asarray(F,np.int64)];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);a=np.linalg.norm(n,axis=1)
  return int(np.count_nonzero(a==0)),float(a.min())
 bz,bmin=degenerates(bv,bf);cz,cmin=degenerates(cv,cf)
 tb,tc=topology(bf),topology(cf)
 assert tb==tc
 stage['topology'][str(sid)]={'baseline':tb,'candidate':tc,'same_face_indices':True,'baseline_zero_area_faces':bz,'candidate_zero_area_faces':cz,'baseline_min_cross_norm':bmin,'candidate_min_cross_norm':cmin}
save();print('Topology and exact Float32 degeneracy checks done',flush=True)
for sid in sorted(WANTED):
 print('Exact self',sid,flush=True)
 br=audit(base_rec[sid],base_rec[sid],True);cr=audit(cand_rec[sid],cand_rec[sid],True)
 stage['self_pairs'].append(pair_report(f'self:{sid}',br,cr));save()
sids=sorted(WANTED)
for i,a in enumerate(sids):
 for b in sids[i+1:]:
  if not overlaps(bounds(base[a][0]),bounds(base[b][0])) and not overlaps(bounds(cand[a][0]),bounds(cand[b][0])):continue
  print('Exact selected pair',a,b,flush=True)
  br=audit(base_rec[a],base_rec[b]);cr=audit(cand_rec[a],cand_rec[b])
  stage['selected_bank_pairs'].append(pair_report(f'{a}:{b}',br,cr));save()
cache=np.load(CACHE/'registered-tissue-banks.npz');inv=json.loads((CACHE/'inventory.json').read_text()); banks=inv['banks']
bank_by_key={b['key']:b for b in banks}; selected_keys={bank_by_stable[s]['key'] for s in WANTED};rec_cache={}
def getbank(key):
 if key not in rec_cache:
  meta=bank_by_key[key];V=cache[key+'_vertices'];F=cache[key+'_faces'].astype(np.int64)
  rec_cache[key]=(V,F,exact_records(V,F))
 return rec_cache[key]
for sid in sorted(WANTED):
 key=bank_by_stable[sid]['key'];bV,bF,_=base[sid];cV,cF,_=cand[sid]
 for meta in banks:
  tkey=meta['key']
  if tkey==key or tkey in selected_keys:continue
  tbounds=(np.asarray(meta['aabb_min_m'],float),np.asarray(meta['aabb_max_m'],float))
  if not overlaps(bounds(bV),tbounds) and not overlaps(bounds(cV),tbounds):continue
  print('Exact registered surrounding',sid,tkey,meta['kind'],meta['stable_id'],flush=True)
  tV,tF,tR=getbank(tkey)
  br=audit(base_rec[sid],tR);cr=audit(cand_rec[sid],tR)
  stage['registered_surrounding_pairs'].append(pair_report(f'{sid}->{tkey}',br,cr,{'target_kind':meta['kind'],'target_stable_id':int(meta['stable_id']),'target_layer':int(meta['layer']),'target_face_count':len(tF)}));save()
print('Load exact identity-current bone bank meshes',flush=True)
bone_env=runpy.run_path(str(BONE_SCRIPT),run_name='bone_gate_import')
_,man,bpath,recs,bv,bf,anchors=bone_env['bone_record_arrays'](BONE_MAN)
pose_data=json.loads((D/'registered-skin-poses.json').read_text());pose_by={int(x['body_index']):x for x in pose_data['all_bodies']}
for rec,anchor in zip(recs,anchors):
 BV,BF=bone_env['bone_world_mesh'](rec,anchor,bv,bf,pose_by);bb=bounds(BV);brec=exact_records(BV,BF)
 name=str(anchor['myosim_body']);body=int(rec[0]);boneid=int(rec[5])
 for sid in sorted(WANTED):
  bV,bF,_=base[sid];cV,cF,_=cand[sid]
  if not overlaps(bounds(bV),bb) and not overlaps(bounds(cV),bb):continue
  print('Exact bone',sid,name,body,boneid,flush=True)
  br=audit(base_rec[sid],brec);cr=audit(cand_rec[sid],brec)
  stage['bone_pairs'].append(pair_report(f'{sid}->{name}',br,cr,{'bone_body_index':body,'bone_stable_id':boneid,'bone_face_count':len(BF)}));save()
def total(rows,key):return sum(x[key] for x in rows)
stage['summary']={'candidate_zero_area_faces':sum(x['candidate_zero_area_faces'] for x in stage['topology'].values()),'self_baseline_pairs':total(stage['self_pairs'],'baseline_pair_count'),'self_candidate_pairs':total(stage['self_pairs'],'candidate_pair_count'),'self_new_pairs':total(stage['self_pairs'],'new_pair_count'),'selected_baseline_pairs':total(stage['selected_bank_pairs'],'baseline_pair_count'),'selected_candidate_pairs':total(stage['selected_bank_pairs'],'candidate_pair_count'),'selected_new_pairs':total(stage['selected_bank_pairs'],'new_pair_count'),'surrounding_baseline_pairs':total(stage['registered_surrounding_pairs'],'baseline_pair_count'),'surrounding_candidate_pairs':total(stage['registered_surrounding_pairs'],'candidate_pair_count'),'surrounding_new_pairs':total(stage['registered_surrounding_pairs'],'new_pair_count'),'bone_baseline_pairs':total(stage['bone_pairs'],'baseline_pair_count'),'bone_candidate_pairs':total(stage['bone_pairs'],'candidate_pair_count'),'bone_new_pairs':total(stage['bone_pairs'],'new_pair_count')}
stage['gate_status']='PASS_STATIC_PAIR_NONREGRESSION' if all(stage['summary'][k]==0 for k in ('candidate_zero_area_faces','self_new_pairs','selected_new_pairs','surrounding_new_pairs','bone_new_pairs')) else 'FAIL_OR_LIMITED'
write(OUT/'exact-current-pose-gate-001.json',stage)
print(json.dumps(stage['summary'],indent=2),flush=True);print('RESULT',OUT/'exact-current-pose-gate-001.json',sha(OUT/'exact-current-pose-gate-001.json'),flush=True)
