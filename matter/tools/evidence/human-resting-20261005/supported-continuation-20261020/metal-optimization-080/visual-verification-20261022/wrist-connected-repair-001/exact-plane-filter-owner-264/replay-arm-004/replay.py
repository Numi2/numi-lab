import sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001';O=E/'exact-plane-filter-owner-264/replay-arm-004'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False)
 f=load('fixedroi264','/tmp/fixed_rest_seam_roi_002.py')
 old=load('numilab_human.cavity_before264',E/'incremental-exact-audit-owner-253/src/numilab_human/cardiac_cavity_intersections.py')
 new=load('numilab_human.cavity_after264',E/'exact-plane-filter-owner-264/src/numilab_human/cardiac_cavity_intersections.py')
 p=E/'joint-functional-owner-audit-20261023/original128-local-vertex-fit-004-refreshed/candidate.nhskin';assert sha(p)=='0de586c667f5ba6155364d95efa3ab20250c31d41ef807f62d6d8d868cb2c305'
 base=f.RB.decode(f.BASE.read_bytes());F=np.asarray(base['inds'],dtype=np.int64);neutral=f.REST/'registered-skin-poses.json';A0,b0=f.RB.reference_transforms(base,f.m.pose(json.loads(neutral.read_text()),'current'),[0,0,0]);ref=f.RB.positions(base,A0,b0).astype('<f4');qmap,members,classes,valid,*_=f.source_map(F,ref)
 pose=f.m.POSES['shoulder_elevation']/'registered-skin-poses.json';data=f.RB.decode(p.read_bytes());A,B=f.RB.reference_transforms(data,f.m.pose(json.loads(pose.read_text()),'current'),[0,0,0]);X=f.RB.positions(data,A,B).astype('<f4');eff,metadata=f.effective_source_ids(qmap,members,valid,X);records,deg=f.exact_records(X,F,eff);assert deg==0
 times={};reports={}
 for name,module in [('before',old),('after',new)]:
  t=time.monotonic();reports[name]=module._audit_pair(records,records,same_surface=True);times[name]=time.monotonic()-t;print(name,times[name],reports[name]['count'],flush=True)
 assert reports['before']==reports['after'];assert reports['after']['count']==30
 report={'schema':'numi.human.exact-self-audit-parity-replay.v1','inputs':{'candidate':sha(p),'pose':sha(pose),'before_predicates':sha(old.__file__),'after_predicates':sha(new.__file__),'runner':sha(__file__)},'world_float32_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'face_count':len(F),'timing_seconds':times,'whole_report_exactly_equal':True,'report':reports['after'],'scope':'One matched static source-audit comparison, not simulation throughput or anatomy admission.'};(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='report'}),flush=True)
if __name__=='__main__':main()
