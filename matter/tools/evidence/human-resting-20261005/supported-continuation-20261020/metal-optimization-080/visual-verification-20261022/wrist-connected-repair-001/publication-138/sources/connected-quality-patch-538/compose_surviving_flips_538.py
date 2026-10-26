#!/usr/bin/env python3
"""Compose exact-self-safe 531 topology replay with the pinned 536 point field.

Uses 534's unchanged batched replay owner; no geometry audit or admission.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/n/Work/numi-human-connected-surface-069/D')
BASE=ROOT/'finite-triangle-backtrack-512/all-pairs-001/backtrack-373/pass-008/candidate.npz'
C531=ROOT/'connected-quality-patch-531/run-002/candidate.npz'
R531=ROOT/'connected-quality-patch-531/run-002/report.json'
A531=ROOT/'connected-quality-patch-531/root-exact-run002/audit-130/summary.json'
P531=ROOT/'connected-quality-patch-531/root-exact-run002/audit-130/surface-self-pairs.npy'
A536=ROOT/'connected-composition-536/root-exact-run001/audit-130/summary.json'
P536=ROOT/'connected-composition-536/root-exact-run001/audit-130/surface-self-pairs.npy'
COORD=ROOT/'connected-composition-536/run-001/candidate.npz'
COORD_REPORT=ROOT/'connected-composition-536/run-001/report.json'
HELPER=ROOT/'connected-quality-patch-534/replay_surviving_flips_534.py'
REPLAY=ROOT/'local-two-parent-retriangulation-405/cross_seam_search_405.py'
PINS={
 'baseline':('c1c0b63ee448703ef3240dbcc251e650022d7f76f475bc75db9cae040d1e3a6c',BASE),
 '531_candidate':('32ec169a84ce4ed22915c0ffb9ea110cac6d7962c682bad9e6d025b0b03f5e4e',C531),
 '531_report':('b02c863f5409f5f14a72d65931c8e7e2f7427279f13adc35f3badeb2f8bd86b7',R531),
 '531_audit':('b2a8713ffc9ca1a893f141e341b771afad98394b36662e38e399937f6064278d',A531),
 '531_pairs':('129927157f3a00a39d521048cb94d9ff24f4cb5baa87822450f6330f22f9416d',P531),
 '536_coordinate_candidate':('e880f1c96e8bb540f1918673fecd07e9dccfea54ebe5cefc91fc14a77307edc5',COORD),
 '536_audit':('b6141b11fb4043aff9151566ae56a9670540af183b9d1eda88522b1ab22b6380',A536),
 '536_pairs':('e9ed0f4b155ce3ec034478a693ac659829445e6c7eb57012bd4f4096b6c17ffc',P536),
 '536_report':('5cee823ba93dbd105d4d8c2ca85821169564869a16518f9df43fa6102ab8d9bf',COORD_REPORT),
 '534_replay_helper':('4559fd425a3300528f49e5211afbd633674910ecaa0d6671408ee0e9d145e504',HELPER),
 '405_replay_owner':('e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967',REPLAY),
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def loadmod(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None:raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def require_inputs():
 out={}
 for name,(want,path) in PINS.items():
  if not path.is_file():raise ValueError(f'missing {name}: {path}')
  got=sha(path)
  if got!=want:raise ValueError(f'{name} SHA mismatch: {got} != {want}')
  out[name]={'path':str(path),'sha256':got}
 return out

def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def compose(base,full531,report531,pairs531,pairs536,coordinate_candidate,helper,replay):
 if not helper.identical(base['points_world_m'],full531['points_world_m']):
  raise ValueError('531 topology candidate unexpectedly changed points')
 if coordinate_candidate['points_world_m'].shape!=base['points_world_m'].shape or coordinate_candidate['points_world_m'].dtype!=base['points_world_m'].dtype:
  raise ValueError('536 point field shape/dtype differs from baseline')
 p1=np.asarray(pairs531,dtype=np.int64);p2=np.asarray(pairs536,dtype=np.int64)
 if p1.ndim!=2 or p1.shape[1]!=2 or p2.ndim!=2 or p2.shape[1]!=2:raise ValueError('self-pair input must be [N,2]')
 union=np.unique(np.sort(np.vstack((p1,p2)),axis=1),axis=0)
 ops=report531.get('selected_flips',[])
 keep,removed=helper.select_remaining(ops,union.ravel())
 remaining=[op for _,op in keep]
 out=helper.apply_ops_batch(base,remaining,replay)
 out['points_world_m']=np.array(coordinate_candidate['points_world_m'],copy=True)
 if not helper.identical(out['points_world_m'],coordinate_candidate['points_world_m']):raise ValueError('536 point coordinates were not copied bitwise')
 if not np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']):raise ValueError('fixed-037 topology not synchronized')
 affected={r for op in remaining for r in map(int,op['face_rows'])}
 changed=set(map(int,np.flatnonzero(np.any(out['source_O_faces']!=base['source_O_faces'],axis=1))))
 if changed!=affected:raise ValueError('changed face rows differ from retained operation rows')
 if affected & set(map(int,union.ravel())):raise ValueError('retained operation touches a union self-pair witness')
 boundary0,nonmanifold0=helper.topology(base['source_O_faces'])
 boundary1,nonmanifold1=helper.topology(out['source_O_faces'])
 if nonmanifold0 or nonmanifold1 or boundary0!=boundary1:raise ValueError('boundary/manifold invariants failed')
 for key in base:
  if key not in helper.ALLOWED_CHANGED_KEYS and key!='points_world_m' and not helper.identical(base[key],out[key]):
   raise ValueError(f'unexpected noncoordinate array mutation: {key}')
 if set(out)-set(base)-helper.ALLOWED_CHANGED_KEYS:raise ValueError('unexpected array key added')
 return out,{
  'schema':'numi.human.connected-composition-538.v1',
  'status':'COMPOSED_CONSTRUCTION_CANDIDATE_PENDING_ROOT_AREA_EDGE_AND_FULL_AUDITS',
  'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,
  '531_self_pair_count':int(len(p1)),'536_self_pair_count':int(len(p2)),
  'deduplicated_union_self_pair_count':int(len(union)),
  'union_self_pair_face_count':int(len(set(map(int,union.ravel())))),
  '531_selected_flip_count':len(ops),'removed_flip_count':len(removed),
  'retained_flip_count':len(remaining),'removed_operations':removed,
  'retained_operation_indices':[int(i) for i,_ in keep],
  'changed_face_rows':sorted(changed),
  'source_noncoordinate_topology_from_531_baseline_plus_retained_flips':True,
  'points_bitwise_copied_from_536_candidate':True,
  'fixed037_connectivity_synchronized':True,
  'boundary_edge_set_unchanged':True,'nonmanifold_edges_after':0,
  'no_retained_flip_touches_union_self_pair_faces':True,
  'area_and_edge_gates':'NOT_RECHECKED_FOR_COMPOSED_CANDIDATE',
  'exact_self_surface_and_185_bank_audits':'NOT_RUN_PENDING_ROOT_SLOT',
  'scope_limit':'Composition only. Full exact self and 185-bank audits, full area/edge gates, layer transfer, volume, and admission remain unverified.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);a=ap.parse_args()
 pins=require_inputs()
 helper=loadmod('replay_surviving_flips_534_for_538',HELPER)
 replay=loadmod('cross_seam_search_405_for_538',REPLAY)
 base=load_npz(BASE);full=load_npz(C531);coord=load_npz(COORD)
 report531=json.loads(R531.read_text());audit531=json.loads(A531.read_text())
 audit536=json.loads(A536.read_text());coordreport=json.loads(COORD_REPORT.read_text())
 if report531.get('candidate_output_sha256')!=pins['531_candidate']['sha256']:raise ValueError('531 report candidate pin mismatch')
 if report531.get('input_candidate_sha256')!=pins['baseline']['sha256']:raise ValueError('531 report baseline pin mismatch')
 if audit531.get('candidate_sha256')!=pins['531_candidate']['sha256'] or audit531.get('complete_surface_audit') is not True or int(audit531.get('exact_PLC_surface_pairs',-1))!=85:raise ValueError('531 complete exact audit does not match')
 if audit536.get('candidate_sha256')!=pins['536_coordinate_candidate']['sha256'] or audit536.get('complete_surface_audit') is not True or int(audit536.get('exact_PLC_surface_pairs',-1))!=4:raise ValueError('536 complete exact audit does not match')
 if coordreport.get('candidate_sha256')!=pins['536_coordinate_candidate']['sha256'] or coordreport.get('coordinate_branch_noncoordinate_arrays_bitwise_equal_to_base') is not True:raise ValueError('536 report does not certify coordinate branch')
 if sha(HELPER)!=PINS['534_replay_helper'][0] or sha(REPLAY)!=PINS['405_replay_owner'][0]:raise ValueError('replay source changed after pin')
 helper.verify_pins(BASE,C531,R531,A531,P531,REPLAY)
 helper.validate_operation_manifest(base,full,report531)
 pairs531=np.load(P531,allow_pickle=False);pairs536=np.load(P536,allow_pickle=False)
 out,meta=compose(base,full,report531,pairs531,pairs536,coord,helper,replay)
 od=Path(a.output_dir)
 if od.exists():raise FileExistsError(od)
 od.mkdir(parents=True)
 dst=od/'candidate.npz';np.savez_compressed(dst,**out)
 meta.update({'candidate_path':str(dst),'candidate_sha256':sha(dst),'input_pins':pins})
 rp=od/'report.json';rp.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report_path':str(rp),'report_sha256':sha(rp),**meta},indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()

