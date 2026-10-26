"""Opt-in exit-plane construction variant; frozen owners are patched only in memory."""
from pathlib import Path
import importlib.util,json,os,sys
HERE=Path(__file__).resolve().parent
CAP=HERE.parent/'raw-lp-direction-capture-546/capture_raw_lp_directions546.py'
s=importlib.util.spec_from_file_location('capture550',CAP);cap=importlib.util.module_from_spec(s);s.loader.exec_module(cap)
if cap.sha(CAP)!='850893a0cc3271fee42e8342e8d73c786cc396eb2f1df4d267c28556c2ace073':raise ValueError('capture source changed')
original=cap._load_original_fit_source
needle='    result=install(owner,objective,crossing);basis=owner.build_immediate_parent_basis;build=owner.build_clearance_rows;depth=owner.build_output_node_direction_rows\n'
def patched(*args,**kwargs):
 source=original(*args,**kwargs)
 if source.count(needle)!=1:raise ValueError('frozen construction install hook changed')
 return 'from coherent_exit_plane_adapter550 import install_coherent_exit550\n'+source.replace(needle,'    install_coherent_exit550(objective,owner)\n'+needle)
cap._load_original_fit_source=patched
def annotate():
 import coherent_exit_plane_adapter550 as h
 out=Path(sys.argv[sys.argv.index('--output')+1]);rp=out/'report.json'
 if not rp.exists():return
 r=json.loads(rp.read_text())
 r['coherent_exit_plane_550']={**h.AUDIT,'scope':'construction routing only; exact self/containment/all185 surface checks required','native_admission':False,'accepted_physical_steps':0,'launcher_sha256':cap.sha(Path(__file__))}
 prior=r.get('fixed_bone_outward_plane_objective_458',{})
 prior.update(name='opt-in coherent exit-facet planes per skin-face/typed-bank group',each_met_pair_plane_separates_that_complete_triangle_pair=False,plane_identity='see crossed_triangle_id and actual plane_triangle_id in row metadata')
 r['fixed_bone_outward_plane_objective_458']=prior
 rp.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
 raw=Path(os.environ[cap.CAPTURE_ENV])/'report.json'
 if raw.exists():
  rr=json.loads(raw.read_text());rr['coherent_exit_plane_550']=r['coherent_exit_plane_550'];raw.write_text(json.dumps(rr,indent=2,sort_keys=True,allow_nan=False)+'\n')
def main():
 try:cap.main()
 finally:annotate()
if __name__=='__main__':main()
