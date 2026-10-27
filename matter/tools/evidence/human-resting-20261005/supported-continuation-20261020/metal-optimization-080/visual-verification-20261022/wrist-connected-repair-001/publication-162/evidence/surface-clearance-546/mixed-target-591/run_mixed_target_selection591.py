#!/usr/bin/env python3
"""Emit a saved 8-row mixed endpoint target; no geometry admission."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069');D=ROOT/'D/surface-clearance-546'
BASE=D/'edge-only-refinement-585/pilot-008/candidate.npz';REF=D/'edge-only-refinement-585/pilot-008/report.json'
SRC=D/'source-outward-targets-589/pilot-008';DEP=D/'source-outward-targets-589/depth-maximizing-pilot-008'
NORMAL=D/'source-outward-targets-589/pilot-008.current-source-normal-audit.json';PREFLIGHT=D/'source-outward-targets-589/mixed-endpoint-preflight-003/report.json'
V2=D/'midpoint-exit-relocation-586/v2-current-q-validation/relocate_midpoint_targets586_v2.py'
V1=D/'source-outward-targets-589/run_source_outward_targets589.py';VDEP=D/'source-outward-targets-589/run_depthmax_targets589_v2.py'
EXPECTED={
 BASE:'6b912d5dc08b044d1dd08a55527435bf6584a0e3a2c3f8c36f43ac4b30d3ded1',REF:'db1f2cccaa12f6def91b3d360eb947afb0b5a7ed21e9b3edda22271c22d52cac',
 SRC/'nearest-union-exits.json':'6ef6a923671c102cf16d571b9af918dd4c782cfe62e967435893a0b40718a663',SRC/'nearest-union-exits.npz':'6575b16f0b5eeabd3fa18ac1a6ea848b0984269f77b3e6e742cb56397876d28a',
 DEP/'nearest-union-exits.json':'59b8b74edeeb079839f0b7dd39e011bcf44f7a17168d520de2275bdfad8ba01a',DEP/'nearest-union-exits.npz':'7d29acfe753d7aae75acc6eb0111d021c4d7beb88238dcebf80bbcb47d87f711',
 NORMAL:'2929d5cd7455a9d0d232f92036a4506658dca987ef98de2905805d91b1855396',PREFLIGHT:'123f1a738f1aaf7609eec59e7d4f442e920c63135606bc296e8a03e66b7ec7cb',
 V2:'b7f3aa8431b7051e0ed8e0c97bc760833c9297c71259da03e74803ec727e210f',V1:'11f8a4a4b906294e9832173b63601f438c9c4bf1f57e488e8fd95f398f0038ec',
 VDEP:'cd64f5280dec578ef8441aa9e44b9c01d44ba703edbd3fde39167ed471c71200'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadmod(p,n):
 s=importlib.util.spec_from_file_location(n,p)
 if s is None or s.loader is None:raise ImportError(str(p))
 m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def loadnpz(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def same(a,b):
 a,b=np.asarray(a),np.asarray(b)
 return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def select_targets(rows,proof,source_targets,depth_targets):
 rows=np.asarray(rows,dtype=np.int64);items=proof.get('selected_rows')
 if rows.ndim!=1 or len(np.unique(rows))!=len(rows) or not isinstance(items,list):raise ValueError('invalid target row domain')
 byrow={int(x['row']):x for x in items}
 if len(byrow)!=len(items) or set(byrow)!=set(map(int,rows)):raise ValueError('preflight rows differ from refinement midpoint rows')
 out=[];origins=[]
 for i,r in enumerate(rows):
  q=byrow[int(r)];mode=q.get('chosen_mode')
  if mode not in ('source-outward','depth-maximizing'):raise ValueError('unknown endpoint origin')
  if not q.get('all185_endpoint_certified') or q['actual_displacement_dot_source_outward']<=0 or q['actual_displacement_dot_signed_current_outward']<=0:raise ValueError('selected endpoint lacks all185 or positive actual outward dots')
  if q['chosen_move_mm']>4.000001 or q['source_distance_mm']>12.000001:raise ValueError('selected endpoint exceeds trust/source cap')
  if mode=='depth-maximizing' and q['actual_displacement_dot_depth_covector']<.05-1e-10:raise ValueError('depth endpoint below cosine floor')
  out.append((source_targets if mode=='source-outward' else depth_targets)[i]);origins.append(mode)
 return np.asarray(out,dtype=np.float32),origins
def replace_points(payload,rows,targets,anchors=()):
 p=np.asarray(payload['points_world_m']);r=np.asarray(rows);t=np.asarray(targets)
 if p.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or r.ndim!=1 or r.dtype.kind not in 'iu' or len(np.unique(r))!=len(r) or t.dtype!=np.float32 or t.shape!=(len(r),3):raise ValueError('invalid candidate/target arrays')
 if np.any(r<0) or np.any(r>=len(p)) or np.intersect1d(r,np.asarray(anchors,dtype=np.int64)).size:raise ValueError('target row out of domain or is anchored')
 y=p.copy();y[r]=t;out=dict(payload);out['points_world_m']=y
 keep=np.ones(len(p),bool);keep[r]=False
 if not same(y[keep],p[keep]):raise AssertionError('non-target point changed')
 return out
def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args(argv)
 if a.out.exists():raise FileExistsError(a.out)
 for p,h in EXPECTED.items():
  if sha(p)!=h:raise ValueError('frozen input/source hash mismatch: '+str(p))
 v2=loadmod(V2,'v586_591');v1=loadmod(V1,'v589_591');vd=loadmod(VDEP,'vd589_591');v2.validate_owner_pins()
 rr=json.loads(REF.read_text());parent_path=Path(rr['input_candidate_path']);parent=loadnpz(parent_path);base=loadnpz(BASE)
 rows=v2.validate_refinement(parent_path,BASE,REF,parent,base,rr)
 sr=json.loads((SRC/'nearest-union-exits.json').read_text());dr=json.loads((DEP/'nearest-union-exits.json').read_text())
 if sr.get('inputs',{}).get('candidate_sha256')!=sha(BASE) or dr.get('inputs',{}).get('candidate_sha256')!=sha(BASE):raise ValueError('target baseline mismatch')
 sv=v1.validate_545_source_outward_report(sr);sa=loadnpz(SRC/'nearest-union-exits.npz');dv=vd.validate_545_depth_maximizing_report(dr,loadnpz(DEP/'nearest-union-exits.npz'));da=loadnpz(DEP/'nearest-union-exits.npz')
 normal=json.loads(NORMAL.read_text())
 if normal.get('candidate_sha256')!=sha(BASE) or normal.get('status')!='PASS' or normal.get('agreement_count')!=len(rows):raise ValueError('current/source normal guard stale')
 for report,arr in ((sr,sa),(dr,da)):
  if not np.array_equal(arr['query_rows'],rows) or not arr['pass_mask'].all() or not arr['target_exactly_outside_all185'].all():raise ValueError('targets do not certify exact midpoint rows')
  rec={int(x['query_row']):x for x in report['targets']}
  if set(rec)!=set(map(int,rows)):raise ValueError('target report row domain mismatch')
  for i,r in enumerate(rows):
   x=rec[int(r)]
   if x.get('exact_endpoint_bank_count')!=185 or not np.array_equal(arr['target_world_f32_m'][i],np.asarray(x['target_world_f32_m'],np.float32)):raise ValueError('target report/NPZ mismatch')
 proof=json.loads(PREFLIGHT.read_text());targets,origins=select_targets(rows,proof,sa['target_world_f32_m'],da['target_world_f32_m'])
 payload=replace_points(base,rows,targets,base['typed_coarse_anchor_output_rows_018'])
 if not same(payload['points_world_m'][base['typed_coarse_anchor_output_rows_018']],base['points_world_m'][base['typed_coarse_anchor_output_rows_018']]):raise AssertionError('anchor changed')
 a.out.mkdir(parents=True);cp=a.out/'candidate.npz';np.savez_compressed(cp,**payload)
 report={'schema':'numi.human.surface-clearance-546.mixed-certified-exit-coordinate-target.591.v1','classification':'unadmitted coordinate target only; no full-set geometry or 130/159 check','base_candidate':str(BASE),'base_candidate_sha256':sha(BASE),'candidate':str(cp),'candidate_sha256':sha(cp),'refinement_report':str(REF),'refinement_report_sha256':sha(REF),'mixed_preflight_report':str(PREFLIGHT),'mixed_preflight_report_sha256':sha(PREFLIGHT),'point_domain_count':len(base['points_world_m']),'changed_point_rows':rows.tolist(),'origins':origins,'rows':proof['selected_rows'],'nonpoint_payload_keys_preserved':list(base),'all_nonpoint_arrays_preserved':True,'anchors_unchanged':True,'586_v2_Q_and_037_refinement_validation':'PASS','source_545_validation':sv,'depth_545_validation':dv,'individual_targets_all185_certified':True,'mixed_set_arithmetic_note':'Cached full-set preflight retains 4 bad area faces and 8 bad edge rows; downstream 582 damping and exact checks are required.','physical_steps':0,'native_admission':False}
 (a.out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 for p,h in EXPECTED.items():
  if sha(p)!=h:raise ValueError('source/input changed during selection: '+str(p))
 print(json.dumps({'status':'UNADMITTED_COORDINATE_TARGET_WRITTEN','candidate':str(cp),'candidate_sha256':sha(cp),'report':str(a.out/'report.json'),'report_sha256':sha(a.out/'report.json')},indent=2))
if __name__=='__main__':main()
