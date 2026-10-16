from pathlib import Path
import argparse,json,hashlib,datetime,subprocess,os,time,importlib.util,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--build',required=True);p.add_argument('--mode',choices=['observer-off','observer-on','coupon'],required=True);p.add_argument('--case',required=True);p.add_argument('--timeout',type=int,default=14400);a=p.parse_args()
s=q/'snapshots'/a.source;b=q/a.build;c=q/'solver'/a.case;d=q/'analysis'/(a.case+'-command-001')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text())
assert bm['success'] and bm['source_sha256']==f['source_sha256']
for n,h in f['source_sha256'].items():assert sha(s/n)==h,n
for n,h in bm['artifacts_sha256'].items():assert sha(n)==h,n
ps=subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True)
workers=[l for l in ps.splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')]
assert not workers,workers
capfile=q/'analysis/native539-source203-cauchy-all-fallback-command-001/capacity-helper.py'
spec=importlib.util.spec_from_file_location('capacity210',capfile);cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap);cap.B=q
assert cap.available()>=cap.MINIMUM
assert not c.exists();d.mkdir(exist_ok=False)
old=json.loads((q/'analysis/native539-source203-cauchy-all-fallback-command-001/command-spec.json').read_text())
argv=list(old['native_argv'])
def setarg(key,value):
 i=argv.index(key);argv[i+1]=str(value)
argv[0]=str(b/'numi-matter-cardboard-probe')
setarg('--metallib',b/'shaders/NumiMatter.metallib');setarg('--output',c/'output')
for key,filename in [('--liner-material','hajali2009_liner_hill_bilinear_fit.nmatter'),('--medium-material','hajali2009_medium_hill_bilinear_fit.nmatter'),('--glue-material','starch2007_finite_glue_elastic.nmatter')]:setarg(key,s/'matter/materials'/filename)
if a.mode!='coupon':
 for key,value in [('--steps',2),('--hold-steps',1),('--unload-steps',1),('--relax-steps',1),('--release-steps',1),('--post-release-relax-steps',1),('--bend-angle','.01')]:setarg(key,value)
if a.mode!='observer-off':argv+=['--coupon-checkpoint-every','8' if a.mode=='coupon' else '2']
env=dict(old['environment']);env['NM_FEM_STRICT_INNER_FORCING']='1'
specification={'schema':'numi.cardboard.source210-native-command.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'runtime_dir':str(b),'source_build_sha256':sha(b/'source-build.json'),'case':str(c),'mode':a.mode,'native_argv':argv,'environment':env,'timeout_s':a.timeout,'physical_residual_tolerance':5e-7,'restart':16,'changes_from539':['Strict active-RHS inner forcing requested; physical operator and outer tolerance unchanged','Periodic accepted snapshots requested without stopping the full schedule'],'qualification_boundary':'This run alone cannot establish mesh/time convergence, component calibration, physical assembly, or crush/impact qualification.','prediction':'No completed trajectory is assumed. Strict inner stopping may reduce the formerly under-solved active RHS; unchanged outer certificates determine accepted controls.','question':'Does this exact combined source complete its requested schedule under unchanged outer certificates, and does periodic observation preserve production histories?'}
sp=d/'command-spec.json';sp.write_text(json.dumps(specification,indent=2)+'\n')
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout',str(a.timeout),'--question',specification['question'],'--prediction',specification['prediction']]
for k,v in sorted(env.items()):cmd+=['--env',k+'='+v]
for path in [sp,s/'frozen-source.json',q/'analysis/source209-strict-inner-forcing-owner-review-001/report.json']:cmd+=['--input-file',str(path)]
cmd+=['--']+argv
(d/'launch.json').write_text(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':cmd,'workers':workers,'available_bytes':cap.available(),'source_files':len(f['source_sha256']),'cap_helper_sha256':sha(capfile),'script_sha256':sha(__file__)},indent=2)+'\n')
reserve=d/'emergency-reserve.bin';proc=None;errors=[];timed=False;space=False;minimum=cap.available();start=time.monotonic()
clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};clean['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';clean['PYTHONDONTWRITEBYTECODE']='1'
try:
 cap.write_reserve_file(reserve)
 with (d/'wrapper.log').open('xb') as out:
  proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=clean,start_new_session=True)
  print(json.dumps({'started':str(c),'pid':proc.pid,'mode':a.mode}),flush=True)
  while proc.poll() is None:
   minimum=min(minimum,cap.available());space=minimum<cap.ABORT_FLOOR;timed=time.monotonic()-start>a.timeout+120
   if space or timed:
    reserve.unlink(missing_ok=True);errors.extend(cap.terminate_owned_native(proc,argv[0])['cleanup_errors']);break
   time.sleep(2)
except BaseException as ex:
 errors.append(repr(ex))
 if proc is not None and proc.poll() is None:errors.extend(cap.terminate_owned_native(proc,argv[0])['cleanup_errors'])
finally:
 reserve.unlink(missing_ok=True)
 code=proc.poll() if proc else None
 unchanged=all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
 r={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'timed_out':timed,'space_abort':space,'errors':errors,'wall_seconds':time.monotonic()-start,'minimum_available_bytes':minimum,'source_and_artifacts_unchanged':unchanged,'source_build_sha256':sha(b/'source-build.json'),'wrapper_sha256':sha(d/'wrapper.log'),'all_six_qualification_axes_remain_separate':True}
 (d/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
sys.exit(code if type(code)is int and code!=0 else int(bool(errors) or not unchanged or timed or space or code is None))
