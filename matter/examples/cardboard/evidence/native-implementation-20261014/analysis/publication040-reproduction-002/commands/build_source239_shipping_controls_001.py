from pathlib import Path
import json,hashlib,shutil,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-refined-coupon-arenas-238-001';s=q/'snapshots/source-shipping-resolution-controls-239-001';d=q/'analysis/source239-shipping-resolution-controls-001';b=q/'build-source239-shipping-resolution-controls-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base/'frozen-source.json')=='5907cc9e05247159cf8010ce2d5d0e06106a4188ea811ccb59e9c35de8346508'
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
n='matter/tools/cardboard_probe.mm';p=s/n;t=p.read_text()
def rep(a,b,count=1):
 global t
 assert t.count(a)==count,(a,t.count(a),count);t=t.replace(a,b)
rep('bool fgmresRestartExplicit = false;','bool fgmresRestartExplicit = false;\n    bool fgmresIterationsExplicit = false;\n    bool newtonIterationsExplicit = false;\n    bool nxExplicit = false;\n    bool nyExplicit = false;\n    bool thicknessSlicesExplicit = false;\n    bool timestepExplicit = false;')
rep('else if (option == "--newton-iterations") arguments.newtonIterations = parseUnsigned(value, option);','else if (option == "--newton-iterations") {\n            arguments.newtonIterations = parseUnsigned(value, option);\n            arguments.newtonIterationsExplicit = true;\n        }')
rep('else if (option == "--fgmres-iterations")\n            arguments.fgmresIterations = parseUnsigned(value, option);','else if (option == "--fgmres-iterations") {\n            arguments.fgmresIterations = parseUnsigned(value, option);\n            arguments.fgmresIterationsExplicit = true;\n        }')
rep('else if (option == "--nx") arguments.nxPerPitch = parseUnsigned(value, option);','else if (option == "--nx") { arguments.nxPerPitch = parseUnsigned(value, option); arguments.nxExplicit = true; }')
rep('else if (option == "--ny") arguments.ny = parseUnsigned(value, option);','else if (option == "--ny") { arguments.ny = parseUnsigned(value, option); arguments.nyExplicit = true; }')
rep('else if (option == "--thickness-slices")\n            arguments.thicknessSlices = parseUnsigned(value, option);','else if (option == "--thickness-slices") {\n            arguments.thicknessSlices = parseUnsigned(value, option);\n            arguments.thicknessSlicesExplicit = true;\n        }')
rep('else if (option == "--dt")\n            arguments.timestepSeconds = parseDouble(value, option);','else if (option == "--dt") {\n            arguments.timestepSeconds = parseDouble(value, option);\n            arguments.timestepExplicit = true;\n        }')
rep('defaults.subdivisionsPerBand = 16u;','defaults.subdivisionsPerBand = arguments.nyExplicit ? arguments.ny : 16u;')
rep('arguments.boxConfig = selected.design.blankConfig;','arguments.boxConfig = selected.design.blankConfig;\n        // The blank is extruded in authored bands, rather than the strip\n        // extrusion below. Keep the exported ny equal to that actual mesh.\n        arguments.ny = arguments.boxConfig.subdivisionsPerBand;')
rep('arguments.nxPerPitch = 8u;\n        arguments.thicknessSlices = 1u;\n        arguments.fgmresIterations = 1024u;','if (!arguments.nxExplicit) arguments.nxPerPitch = 8u;\n        if (!arguments.thicknessSlicesExplicit) arguments.thicknessSlices = 1u;\n        if (!arguments.fgmresIterationsExplicit) arguments.fgmresIterations = 1024u;')
rep('arguments.newtonIterations = 14u;\n        arguments.timestepSeconds = 25.0e-6;','if (!arguments.newtonIterationsExplicit) arguments.newtonIterations = 14u;\n        if (!arguments.timestepExplicit) arguments.timestepSeconds = 25.0e-6;')
old='std::cout << "{\\\"schema\\\":\\\"numi.cardboard.probe-argument-validation.v1\\\",\\\"arguments_valid\\\":true,\\\"file_inputs_validated\\\":false,\\\"mesh_generated\\\":false,\\\"native_runtime_created\\\":false}\\n";'
assert old in t
new='''std::cout << std::setprecision(17)
                    << "{\\"schema\\":\\"numi.cardboard.probe-argument-validation.v1\\",\\"arguments_valid\\":true,\\"file_inputs_validated\\":false,\\"mesh_generated\\":false,\\"native_runtime_created\\":false,\\"resolved\\":{"
                    << "\\"shipping_box\\":" << (arguments.shippingBox ? "true" : "false")
                    << ",\\"nx_per_pitch\\":" << arguments.nxPerPitch
                    << ",\\"ny\\":" << arguments.ny
                    << ",\\"thickness_slices\\":" << arguments.thicknessSlices
                    << ",\\"box_subdivisions_per_band\\":" << arguments.boxConfig.subdivisionsPerBand
                    << ",\\"dt_s\\":" << arguments.timestepSeconds
                    << ",\\"newton_iterations\\":" << arguments.newtonIterations
                    << ",\\"fgmres_iterations\\":" << arguments.fgmresIterations << "}}\\n";'''
t=t.replace(old,new);p.write_text(t)
tn='matter/tools/test_cardboard_shipping_resolution_controls.py';(s/tn).write_text("#!/usr/bin/env python3\n\"\"\"Exercise the real driver's resolved shipping controls without creating a mesh or Metal runtime.\"\"\"\nfrom pathlib import Path\nimport argparse,json,subprocess\np=argparse.ArgumentParser();p.add_argument(\"--probe\",type=Path,required=True);a=p.parse_args()\nchecks=0;commands=[]\ndef call(extra,ok=True):\n global checks\n cmd=[str(a.probe),\"--shipping-box\",\"--shipping-preflight-only\",\"--validate-only\"]+extra\n r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);commands.append({\"argv\":cmd,\"exit_code\":r.returncode})\n if not ok:\n  assert r.returncode!=0,(cmd,r.stdout,r.stderr);checks+=1;return\n assert r.returncode==0,(cmd,r.stdout,r.stderr)\n d=json.loads(r.stdout);assert d[\"arguments_valid\"] and not d[\"mesh_generated\"] and not d[\"native_runtime_created\"]\n checks+=1;return d[\"resolved\"]\nbase=call([])\nexpected={\"shipping_box\":True,\"nx_per_pitch\":8,\"ny\":16,\"thickness_slices\":1,\"box_subdivisions_per_band\":16,\"dt_s\":25e-6,\"newton_iterations\":14,\"fgmres_iterations\":1024}\nassert base==expected,(base,expected);checks+=1\nexplicit=[\"--nx\",\"8\",\"--ny\",\"16\",\"--thickness-slices\",\"1\",\"--dt\",\".000025\",\"--newton-iterations\",\"14\",\"--fgmres-iterations\",\"1024\"]\nassert call(explicit)==base;checks+=1\nfor nx in (16,32):\n r=call([\"--nx\",str(nx)]);assert r==dict(base,nx_per_pitch=nx);checks+=1\nfor ny in (32,64):\n r=call([\"--ny\",str(ny)]);assert r==dict(base,ny=ny,box_subdivisions_per_band=ny);checks+=1\nfor layers in (2,4):\n r=call([\"--thickness-slices\",str(layers)]);assert r==dict(base,thickness_slices=layers);checks+=1\nfor dt in (12.5e-6,6.25e-6):\n r=call([\"--dt\",str(dt)]);assert r==dict(base,dt_s=dt);checks+=1\nr=call([\"--newton-iterations\",\"64\",\"--fgmres-iterations\",\"512\"])\nassert r==dict(base,newton_iterations=64,fgmres_iterations=512);checks+=1\nfor option,value in ((\"--nx\",\"0\"),(\"--ny\",\"0\"),(\"--thickness-slices\",\"0\"),(\"--dt\",\"-1\"),(\"--newton-iterations\",\"129\"),(\"--fgmres-iterations\",\"2048\")):\n call([option,value],False)\nr=subprocess.run([str(a.probe),\"--preset\",\"literature2009\",\"--validate-only\",\"--nx\",\"32\",\"--ny\",\"8\",\"--thickness-slices\",\"4\",\"--dt\",\".00005\"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)\nassert r.returncode==0,r.stderr\nv=json.loads(r.stdout)[\"resolved\"]\nassert v[\"nx_per_pitch\"]==32 and v[\"ny\"]==8 and v[\"thickness_slices\"]==4 and v[\"dt_s\"]==.00005\nchecks+=1\nprint(json.dumps({\"schema\":\"numi.cardboard.shipping-resolution-cli-checks.v1\",\"checks\":checks,\"physical_steps\":0,\"shipping_mesh_or_convergence_qualified\":False,\"commands\":commands}))\n")
for n in ['matter/tools/cardboard_probe.mm',tn]:
 f['source_sha256'][n]=sha(s/n)
 if n in cfg['files']:cfg['files'][n]=sha(s/n)
f['source239_shipping_resolution_controls']={'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'changed':['matter/tools/cardboard_probe.mm',tn],'default_shipping_geometry_changed':False,'default_shipping_mesh_ny_metadata_corrected_from_4_to_16':True,'explicit_controls_preserved':['nx','ny as subdivisions per blank band','thickness-slices','dt','newton-iterations','fgmres-iterations'],'physical_operator_changed':False,'qualification':'Argument execution only until actual generated meshes and complete stage trajectories are independently compared.'}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2','--target','numi-matter-cardboard-probe']
(d/'command.json').write_text(json.dumps({'argv':a,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:rc=subprocess.run(a,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env).returncode
tests=[]
if rc==0:
 cmds=[[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/tn),'--probe',str(b/'matter/numi-matter-cardboard-probe')]]
 cmds += [[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/n)] for n in ['test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
 for i,a in enumerate(cmds):
  p=subprocess.run(a,cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);(d/f'test-{i}.stdout').write_bytes(p.stdout);(d/f'test-{i}.stderr').write_bytes(p.stderr);tests.append({'argv':a,'exit_code':p.returncode,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
  if p.returncode:break
report={'schema':'numi.cardboard.source239-shipping-resolution-controls.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'build_exit_code':rc,'tests':tests,'all_pass':rc==0 and len(tests)==3 and all(t['exit_code']==0 for t in tests),'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'actual_mesh_validation_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(int(not report['all_pass']))
