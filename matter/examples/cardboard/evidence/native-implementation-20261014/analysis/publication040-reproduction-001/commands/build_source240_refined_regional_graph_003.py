from pathlib import Path
import json,shutil,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clone(base,s):
 f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());s.mkdir()
 for n,h in {**cfg['files'],**f['source_sha256']}.items():
  assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
 return f,cfg
def freeze(s,f,cfg,changed):
 for n in changed:
  f['source_sha256'][n]=sha(s/n)
  if n in cfg['files']:cfg['files'][n]=sha(s/n)
 (s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
base=q/'snapshots/source-shipping-resolution-controls-239-002';s=q/'snapshots/source-shipping-resolution-controls-239-003';f,cfg=clone(base,s)
n='matter/tools/cardboard_probe.mm';p=s/n;t=p.read_text();old='require(arguments.ny >= 1u && arguments.ny <= 32u,\n            "--ny must lie in [1, 32]");';assert t.count(old)==1
t=t.replace(old,'require(arguments.ny >= 1u &&\n                arguments.ny <= (arguments.shippingBox ? 64u : 32u),\n            arguments.shippingBox ? "--ny must lie in [1, 64] for shipping blank bands"\n                                  : "--ny must lie in [1, 32]");');p.write_text(t)
f['source239_shipping_width_admission_repair']={'prior_failed_report':str(q/'analysis/source239-shipping-resolution-controls-002/report.json'),'shipping_max_ny':64,'existing_box_band_limit':64,'strip_max_ny_unchanged':32,'reason':'The existing strip ny validator incorrectly blocked the third shipping width refinement although shipping blank bands already admit64.'}
freeze(s,f,cfg,[n]);shipping=s
base=q/'snapshots/source-refined-regional-graph-240-002';s=q/'snapshots/source-refined-regional-graph-240-003';d=q/'analysis/source240-refined-regional-graph-003';b=q/'build-source240-refined-regional-graph-003';f,cfg=clone(base,s);d.mkdir()
n='matter/tools/cardboard_probe.mm';p=s/n;t=(shipping/n).read_text()
old='graph-Schwarz selector 5 is a bounded conditioning-only coupon experiment (<=4096 nodes, <=32768 tets, <=4 environments)';assert t.count(old)==1
t=t.replace(old,'graph-Schwarz selector 5 is a bounded conditioning-only pure coupon experiment (<=32768 nodes, <=131072 tets, <=4 environments; graph arenas <=96 MiB)')
t=t.replace('graph_schwarz_coupon_node_cap\\\": 4096','graph_schwarz_coupon_node_cap\\\": 32768');p.write_text(t)
f['source240_shipping_cli_repair_composed']={'source_snapshot':str(shipping),'frozen_sha256':sha(shipping/'frozen-source.json'),'all_other_source240_files_retained_exactly':True}
freeze(s,f,cfg,[n])
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
commands=[];n='matter/tools/cardboard_refined_graph_layout_check.cpp'
a=['clang++','-O2','-std=c++20','-I'+str(s/'matter/tools'),'-I'+str(s/'matter/src'),'-I'+str(s/'matter/include'),str(s/n),'-o',str(d/'layout-check')]
p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);(d/'compile.stdout').write_bytes(p.stdout);(d/'compile.stderr').write_bytes(p.stderr);commands.append({'argv':a,'exit_code':p.returncode})
if p.returncode==0:
 a=[str(d/'layout-check')];p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);(d/'layout.stdout.jsonl').write_bytes(p.stdout);(d/'layout.stderr').write_bytes(p.stderr);commands.append({'argv':a,'exit_code':p.returncode})
if all(x['exit_code']==0 for x in commands):
 a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2','--target','numi-matter-cardboard-probe','--target','numi-matter-regional-graph-schwarz-layout-check']
 with (d/'build.log').open('xb') as out:p=subprocess.run(a,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env)
 commands.append({'argv':a,'exit_code':p.returncode})
 if p.returncode==0:
  cmds=[[str(b/'matter/numi-matter-regional-graph-schwarz-layout-check')],[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/test_cardboard_shipping_resolution_controls.py'),'--probe',str(b/'matter/numi-matter-cardboard-probe')]]
  cmds += [[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
  for i,a in enumerate(cmds):
   p=subprocess.run(a,cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);(d/f'test-{i}.stdout').write_bytes(p.stdout);(d/f'test-{i}.stderr').write_bytes(p.stderr);commands.append({'argv':a,'exit_code':p.returncode})
   if p.returncode:break
r={'schema':'numi.cardboard.source240-refined-regional-graph.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'commands':commands,'all_pass':len(commands)==8 and all(x['exit_code']==0 for x in commands),'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'native_preconditioner_qualification_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(int(not r['all_pass']))
