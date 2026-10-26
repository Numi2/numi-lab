"""Pair-specific SAT targets inside the existing connected construction fitter."""
from pathlib import Path
import importlib.util,json,os,sys
import numpy as np
from pair_sat_routing_551 import choose_pair_sat_plane
HERE=Path(__file__).resolve().parent;CAP=HERE.parent/'raw-lp-direction-capture-546/capture_raw_lp_directions546.py'
s=importlib.util.spec_from_file_location('capture551',CAP);cap=importlib.util.module_from_spec(s);s.loader.exec_module(cap)
if cap.sha(CAP)!='850893a0cc3271fee42e8342e8d73c786cc396eb2f1df4d267c28556c2ace073':raise ValueError('capture source changed')
AUDIT={};PLANES={}
def install_pair_sat551(objective):
 if getattr(objective,'_pair_sat551_installed',False):raise ValueError('SAT adapter installed twice')
 original_plane=objective.outward_plane_for_pair;builder=objective.build_fixed_bone_outward_plane_rows
 calls=[]
 def plane(skin,bone):
  chosen=choose_pair_sat_plane(skin,bone);old=original_plane(skin,bone)
  calls.append((chosen,float(np.max(np.maximum(0.,old['corner_lower_mm'])))))
  return chosen
 objective.outward_plane_for_pair=plane
 def build(points,faces,support,anchors,pairs,bones,total_weight=1.):
  pairs=list(pairs);calls.clear()
  r=builder(points,faces,support,anchors,pairs,bones,total_weight)
  if len(calls)!=len(pairs):raise ValueError('pair plane call inventory mismatch')
  for field in ['metadata','guard_metadata']:
   for record in r[field]:
    chosen=calls[int(record['pair_index'])][0]
    record.update(plane_selection='finite_pair_minimum_deficit_sat_551',sat_axis_source=chosen['axis_source'],sat_axis_index=chosen['axis_index'],sat_axis_sign=chosen['axis_sign'],claim_scope='one_finite_pair_projection_target_only_not_whole_bank_certificate')
  PLANES.clear();PLANES.update(
   pair_identity_face_body_bone_triangle=np.array([[p['face_row'],p['body'],p['bone_id'],p['triangle_id']] for p in pairs],np.int64),
   normal_world_unit=np.array([p[0]['normal_world_unit'] for p in calls]),
   plane_point_mm=np.array([p[0]['plane_point_mm'] for p in calls]),
   margin_mm=np.array([p[0]['float32_ulp_margin_mm'] for p in calls]),
   corner_lower_mm=np.array([p[0]['corner_lower_mm'] for p in calls]),
   original_outward_max_positive_lower_mm=np.array([p[1] for p in calls]),
   sat_max_positive_lower_mm=np.array([p[0]['max_positive_corner_deficit_mm'] for p in calls]),
   axis_source=np.array([p[0]['axis_source'] for p in calls]),
   axis_index=np.array([p[0]['axis_index'] for p in calls],np.int64),
   axis_sign=np.array([p[0]['axis_sign'] for p in calls],np.int64))
  AUDIT.update(pair_count=len(pairs),corner_row_count=len(r['lower']),nonpositive_corner_count=int(np.count_nonzero(r['lower']<=0)),mean_original_max_deficit_mm=float(np.mean(PLANES['original_outward_max_positive_lower_mm'])),mean_sat_max_deficit_mm=float(np.mean(PLANES['sat_max_positive_lower_mm'])),axis_source_counts={str(k):int(v) for k,v in zip(*np.unique(PLANES['axis_source'],return_counts=True))})
  return r
 objective.build_fixed_bone_outward_plane_rows=build;objective._pair_sat551_installed=True
original=cap._load_original_fit_source
needle='    result=install(owner,objective,crossing);basis=owner.build_immediate_parent_basis;build=owner.build_clearance_rows;depth=owner.build_output_node_direction_rows\n'
def patched(*args,**kwargs):
 source=original(*args,**kwargs)
 if source.count(needle)!=1:raise ValueError('frozen construction install hook changed')
 return 'from fit_pair_sat_planes551 import install_pair_sat551\n'+source.replace(needle,'    install_pair_sat551(objective)\n'+needle)
cap._load_original_fit_source=patched
def annotate():
 out=Path(sys.argv[sys.argv.index('--output')+1]);rp=out/'report.json'
 if not rp.exists():return
 owner=sys.modules['fit_pair_sat_planes551']
 decisions=out.parent/'pair-sat-planes551.npz'
 np.savez_compressed(decisions,**owner.PLANES)
 r=json.loads(rp.read_text());r['pair_sat_routing_551']={**owner.AUDIT,'scope':'offline construction target, exact185bank andselfchecks required','native_admission':False,'accepted_physical_steps':0,'launcher_sha256':cap.sha(Path(__file__)),'helper_sha256':cap.sha(HERE/'pair_sat_routing_551.py'),'decisions_path':str(decisions),'decisions_sha256':cap.sha(decisions)}
 prior=r.get('fixed_bone_outward_plane_objective_458',{})
 prior.update(name='finite-pair SAT minimum-deficit construction targets551',plane_identity='support of actual registered original bone triangle on chosen pair-specific SAT axis',each_met_pair_plane_separates_that_complete_triangle_pair=True,whole_bank_clearance_certificate=False)
 r['fixed_bone_outward_plane_objective_458']=prior
 rp.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
 raw=Path(os.environ[cap.CAPTURE_ENV])/'report.json'
 if raw.exists():
  rr=json.loads(raw.read_text());rr['pair_sat_routing_551']=r['pair_sat_routing_551'];raw.write_text(json.dumps(rr,indent=2,sort_keys=True,allow_nan=False)+'\n')
def main():
 try:cap.main()
 finally:annotate()
if __name__=='__main__':main()
