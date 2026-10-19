from pathlib import Path
import csv, hashlib, json, math, os, shutil, subprocess, sys, time

ROOT=Path('/Users/n/numi-human-separation-20261018/C')
CAND=ROOT/'aortic-wall-pressure-volume-258'
PREP=CAND/'prepared-run-plan-newton16-005.json'
OUT=CAND/'native-32step-paired-newton16-005'
INTEGRATION=ROOT/'integrated-general-tissue-258'
BUILD=CAND/'build-newton16-app-001'
BINARY=BUILD/'numi-human-native'
APP=BUILD/'source/apps/numilab_human_myosim_visual_probe.mm'
WALL_HEADER=BUILD/'source/apps/NumiHumanRestingAorticWall.hpp'
COUPLING_HEADER=BUILD/'source/apps/NumiHumanVesselWallCoupling.hpp'
COMPILE_RECEIPT=BUILD/'compile-receipt.json'
PRODUCTION_BUILD_RECEIPT=ROOT/'integrated-general-tissue-258/production-merged-007/build-production-001/build-receipt.json'
LINK_RECEIPT=BUILD/'link-receipt.json'
BUILD_INPUT=BUILD/'build-input.json'
DEPFILE=BUILD/'human-native.d'
BASE_INV=ROOT/'integrated-general-tissue-258/native-paired-006-inner-forcing/all_optins_clean_control/invocation.json'
MSL=ROOT/'integrated-general-tissue-258/production-merged-007/MatterSource.metal'
RUNTIME_SOURCE=ROOT/'sustained-blood-247/native-mpm-grid-boundary-001/native-kkt-grid-floor96-prep-001/input-runtime-reaction-rollback.mm'
LIB=ROOT/'integrated-general-tissue-258/production-merged-007/build-production-001/libmetalrobo.dylib'
WRAPPER=Path('/Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py')
PYTHON=Path('/Users/n/numi-human-prep-venv-20261005/bin/python3')
WALL_OBJECT='Human/BodyParts3D/stable-9/reduced-cavity-wall-v1'
EXPECTED={
 'runtime_source':'438911d8f1bdaa9f820c6d2bba1bbe87182d2bd14e485f5bbf5fcd2c1e01014c',
 'runtime_library':'ec651fdc3ebd835f22590b9937308fa88678e0e90ae3dcf36f131cf5efd3854e',
 'matter_source':'85fba053a66dac84e818b2a34ca406cc5d608eddd1ef2df1d6f6b0b55f4b2f34',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
require(not OUT.exists(),f'refusing existing output {OUT}')
for p in (BINARY,APP,WALL_HEADER,COUPLING_HEADER,COMPILE_RECEIPT,LINK_RECEIPT,BUILD_INPUT,DEPFILE,PRODUCTION_BUILD_RECEIPT,BASE_INV,MSL,RUNTIME_SOURCE,LIB,WRAPPER,PYTHON): require(p.is_file(),f'missing pinned input {p}')
compile_record=json.loads(COMPILE_RECEIPT.read_text())
link_record=json.loads(LINK_RECEIPT.read_text())
require(compile_record['candidate_app_sha256']==sha(APP),'Newton compile receipt app pin mismatch')
require(link_record['candidate_app_sha256']==sha(APP),'Newton link receipt app pin mismatch')
require(compile_record['product_path']==str(BUILD/'numilab_human_myosim_visual_probe.mm.o'),'compile output target mismatch')
require(Path(compile_record['argv'][compile_record['argv'].index('-c')+1])==APP,'compile source target mismatch')
require(link_record['product_path']==str(BINARY) and link_record['product_sha256']==sha(BINARY),'Newton binary does not match link receipt')
build_input=json.loads(BUILD_INPUT.read_text())
require(build_input['candidate_app_sha256']==sha(APP),'fenced build app pin mismatch')
expected_dependencies={d['path']:d['sha256'] for d in build_input['source_dependency_hashes']}
require(len(expected_dependencies)==158,'unexpected source dependency inventory size')
for path,pin in expected_dependencies.items(): require(Path(path).is_file() and sha(path)==pin,'source dependency hash changed: '+path)
actual_dependencies=set((DEPFILE.read_text().replace('\\\n',' ').split(': ',1)[1]).split())
require(actual_dependencies==set(expected_dependencies),'compiler dependency file differs from precompiled dependency inventory')
require(sha(RUNTIME_SOURCE)==EXPECTED['runtime_source'],'reaction-rollback runtime source hash changed')
prod_build=json.loads(PRODUCTION_BUILD_RECEIPT.read_text())
require(prod_build['metal_source_sha256']==EXPECTED['matter_source'],'production build receipt does not pin selected MSL')
require(prod_build['base_library_sha256']=='5a58b578134a5d7ce31dd6cc7a31713914ded0d64dca2fc571eec9363c8a4f8b','unexpected production base runtime library')
require(sha(LIB)==EXPECTED['runtime_library'],'matched runtime library hash changed')
require(sha(MSL)==EXPECTED['matter_source'],'matched Matter source hash changed')
base=json.loads(BASE_INV.read_text())
base_assets={}
for v in base['argv'][1:]:
 p=Path(v)
 if p.is_file(): base_assets[str(p.resolve())]=sha(p)
for k,v in base['environment'].items():
 if k.endswith(('_SOURCE','_MATERIAL')) and Path(v).is_file(): base_assets[str(Path(v).resolve())]=sha(v)
baseargv=list(base['argv']); baseargv[0]=str(BINARY)
prior_test_msl=base.get('environment',{}).get('NUMI_MATTER_TEST_SOURCE')
if prior_test_msl: base_assets.pop(str(Path(prior_test_msl).resolve()),None)
baseargv[baseargv.index('--muscle-step-count')+1]='32'
baseargv[baseargv.index('--muscle-step-seconds')+1]='0.00025000'
baseenv=dict(base['environment'])
alias=CAND/'lib-production-newton16-005'; alias.mkdir(parents=True,exist_ok=True)
aliaslib=alias/'libmetalrobo.dylib'
if not aliaslib.exists(): aliaslib.symlink_to(LIB)
require(aliaslib.resolve()==LIB.resolve() and sha(aliaslib)==EXPECTED['runtime_library'],'Newton run loader alias does not resolve to pinned runtime library')
baseenv.update({
 'DYLD_LIBRARY_PATH':str(alias), 'DYLD_PRINT_LIBRARIES':'1',
 'NUMI_MATTER_TEST_SOURCE':str(MSL), 'NUMI_MATTER_TEST_SOURCE_SHA256':sha(MSL),
 'NUMI_HUMAN_BLOOD_MPM_ENABLED':'0',
 'NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE':'1',
 'NUMI_HUMAN_RESTING_PASSIVE_FASCIA':'1',
 'NUMI_HUMAN_RESTING_AORTIC_WALL':'1',
 'NUMI_HUMAN_RESTING_SKIN_FAT_FEM':'0',
 'NUMI_HUMAN_RESTING_PROSTATE_FEM':'0',
 'NUMI_HUMAN_RESTING_ORGAN_BATCH_FEM':'0',
 'NUMI_HUMAN_RESTING_OPTIC_NERVE_FEM':'0',
 'NUMI_HUMAN_RESTING_INFERRED_ACL':'1',
 'NUMI_HUMAN_RESTING_FGMRES_BUDGET':'0',
 'NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS':'16',
 'NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT':'0',
 'NUMI_MATTER_GPU_TIMING':'0',
 'NUMI_HUMAN_SHARED_TISSUE_ROLLBACK_TEST':'0',
})
cases=[{'name':'wall_on','wall_enabled':True},{'name':'wall_off','wall_enabled':False}]
runplan={
 'schema':'numi.aortic_wall_pressure_volume_paired.v1',
 'scope':'Two matched 32-step, 0.25 ms native runs with passive shared-runtime Newton budget 16 and default FGMRES budget 64, differing only in registered aortic wall opt-in. MPM/optic nerve/skin-fat/prostate/organ-batch are off; passive fascia and its required bilateral ACL FEM owner stay on because the wall requires the shared tissue owner. Measures per-step CV aortic pressure and terminal accepted FEM state. Offline geometric cavity volume is reconstructed from the exact pinned 24x8 capped-wall tetrahedral topology; terminal pressure/volume association and ON/OFF pressure contrast only, not full time-resolved pressure-volume causality. The wall is a closed capped reservoir slice, not a patent lumen/network port. No visual verification.',
 'base_invocation_path':str(BASE_INV),'base_invocation_sha256':sha(BASE_INV),
 'app_source_path':str(APP),'app_sha256':sha(APP),'aortic_wall_header_path':str(WALL_HEADER),'aortic_wall_header_sha256':sha(WALL_HEADER),'vessel_wall_coupling_header_path':str(COUPLING_HEADER),'vessel_wall_coupling_header_sha256':sha(COUPLING_HEADER),'compile_receipt_path':str(COMPILE_RECEIPT),'compile_receipt_sha256':sha(COMPILE_RECEIPT),'link_receipt_path':str(LINK_RECEIPT),'link_receipt_sha256':sha(LINK_RECEIPT),'production_build_receipt_path':str(PRODUCTION_BUILD_RECEIPT),'production_build_receipt_sha256':sha(PRODUCTION_BUILD_RECEIPT),
 'binary_path':str(BINARY),'binary_sha256':sha(BINARY),'runtime_source_path':str(RUNTIME_SOURCE),'runtime_source_sha256':sha(RUNTIME_SOURCE),
 'runtime_library_path':str(LIB),'runtime_library_sha256':sha(LIB),'matter_source_path':str(MSL),'matter_source_sha256':sha(MSL),
 'matter_mpm_enabled':False,'step_count':32,'step_seconds':0.00025,'physical_horizon_seconds':0.008,
 'mesh_contract':{'object':WALL_OBJECT,'nodes':532,'tetrahedra':1584,'circumferential_segments':24,'axial_segments':8,'cavity_faces_expected':432},
 'cases':cases,'runtime_controls':{'NUMI_HUMAN_RESTING_PASSIVE_FASCIA':'1','NUMI_HUMAN_RESTING_INFERRED_ACL':'1','NUMI_HUMAN_RESTING_AORTIC_WALL':'per_case','NUMI_HUMAN_RESTING_SKIN_FAT_FEM':'0','NUMI_HUMAN_RESTING_PROSTATE_FEM':'0','NUMI_HUMAN_RESTING_ORGAN_BATCH_FEM':'0','NUMI_HUMAN_RESTING_OPTIC_NERVE_FEM':'0','NUMI_HUMAN_BLOOD_MPM_ENABLED':'0','NUMI_HUMAN_RESTING_FGMRES_BUDGET':'0','NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS':'16','NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT':'0','NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE':'1'},'source_asset_sha256':base_assets,'visual_verification':False,
 'runner_path':str(Path(__file__).resolve()),'runner_sha256':sha(Path(__file__)),
}
if '--execute' not in sys.argv:
 require(not PREP.exists(),f'refusing existing prepared plan {PREP}')
 PREP.write_text(json.dumps(runplan,indent=2)+'\n')
 print(json.dumps({'prepared_plan':str(PREP),'plan_sha256':sha(PREP),'runner_sha256':runplan['runner_sha256'],'cases':cases,'execution_started':False},indent=2))
 sys.exit(0)
require(PREP.is_file(),f'missing prepared plan {PREP}')
frozen=json.loads(PREP.read_text())
require(sha(Path(__file__))==frozen.get('runner_sha256'),'runner changed after preparation')
require(frozen==runplan,'prepared plan no longer matches current source pins or controls')
require(not OUT.exists(),f'refusing existing output {OUT}')
OUT.mkdir(parents=True)
(OUT/'run-plan.json').write_text(json.dumps(runplan,indent=2)+'\n')
launcher=[str(PYTHON),str(WRAPPER),'--']
attempts=[]
for case in cases:
 d=OUT/case['name']; d.mkdir(); outdir=d/'output'; outdir.mkdir()
 argv=list(baseargv); argv[4]=str(outdir)
 env=dict(baseenv)
 env['NUMI_HUMAN_RESTING_AORTIC_WALL']='1' if case['wall_enabled'] else '0'
 env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(d/'common-field-failure.json')
 assets=dict(base_assets)
 for path,pin in ((BINARY,sha(BINARY)),(APP,sha(APP)),(WALL_HEADER,sha(WALL_HEADER)),(COUPLING_HEADER,sha(COUPLING_HEADER)),(COMPILE_RECEIPT,sha(COMPILE_RECEIPT)),(LINK_RECEIPT,sha(LINK_RECEIPT)),(BUILD_INPUT,sha(BUILD_INPUT)),(DEPFILE,sha(DEPFILE)),(PRODUCTION_BUILD_RECEIPT,sha(PRODUCTION_BUILD_RECEIPT)),(RUNTIME_SOURCE,EXPECTED['runtime_source']),(LIB,EXPECTED['runtime_library']),(MSL,EXPECTED['matter_source'])):
  assets[str(path.resolve())]=pin
 inv={'schema':'numi.aortic_wall_pressure_volume_attempt.v1','case':case,'argv':argv,'launcher_argv':launcher+argv,'environment':env,'assets_sha256':assets,'plan_sha256':sha(OUT/'run-plan.json'),'requested_step_count':32,'step_seconds':0.00025,'visual_verification':False}
 (d/'invocation.json').write_text(json.dumps(inv,indent=2)+'\n')
 childenv=os.environ.copy();childenv.update(env); t=time.monotonic()
 with (d/'native.log').open('w') as f: proc=subprocess.run(launcher+argv,env=childenv,stdout=f,stderr=subprocess.STDOUT)
 inv.update({'process_exit':proc.returncode,'wall_seconds':time.monotonic()-t,'runtime_library_sha256_after':sha(LIB)})
 (d/'invocation.json').write_text(json.dumps(inv,indent=2)+'\n')
 attempts.append({'case':case['name'],'exit':proc.returncode,'wall_seconds':inv['wall_seconds'],'native_log':str(d/'native.log')})
 if proc.returncode: break

def csv_rows(path):
 with Path(path).open(newline='') as f: return list(csv.DictReader(f))
def parse_wall(logpath,expected_steps):
 lines=Path(logpath).read_text(errors='replace').splitlines()
 begin=next((x for x in lines if x.startswith('resting_FEM_endpoint_capture=begin ')),None)
 require(begin is not None,'missing terminal FEM capture')
 b=dict(t.split('=',1) for t in begin.split()[1:])
 require(int(b['steps'])==expected_steps,'FEM endpoint step count mismatch')
 summary_line=next((x for x in lines if x.startswith('resting_FEM_object_capture=summary object='+WALL_OBJECT+' ')),None)
 require(summary_line is not None,'registered wall object absent from endpoint capture')
 s=dict(t.split('=',1) for t in summary_line.split()[1:])
 require((int(s['node_count']),int(s['tet_count']))==(532,1584),'wall endpoint topology count mismatch')
 first_node=int(s['first_node'])
 nodes={}; tets={}
 for line in lines:
  if line.startswith('resting_FEM_node,object='+WALL_OBJECT+','):
   d=dict(t.split('=',1) for t in line.split(',')[1:]); local=int(d['local'])
   nodes[local]=((float(d['initial_x']),float(d['initial_y']),float(d['initial_z'])),(float(d['accepted_x']),float(d['accepted_y']),float(d['accepted_z'])))
  elif line.startswith('resting_FEM_tet,object='+WALL_OBJECT+','):
   d=dict(t.split('=',1) for t in line.split(',')[1:]); local=int(d['local'])
   tets[local]=tuple(int(d['n'+str(i)])-first_node for i in range(4))
 require(len(nodes)==532 and len(tets)==1584,'incomplete terminal wall mesh rows')
 n,axial=24,8
 wall_j=[]
 for line in lines:
  if line.startswith('resting_FEM_tet,object='+WALL_OBJECT+','):
   d=dict(t.split('=',1) for t in line.split(',')[1:]); wall_j.append(float(d['J']))
 require(len(wall_j)==1584,'incomplete terminal wall deformation-J rows')
 inner={48*z+2*i for z in range(axial+1) for i in range(n)}
 cap={tuple(sorted((432,2*i,2*((i+1)%n)))) for i in range(n)}
 cap |= {tuple(sorted((433,48*axial+2*i,48*axial+2*((i+1)%n)))) for i in range(n)}
 boundary={}
 for local in range(1584):
  a,b,c,d=tets[local]
  for face in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
   key=tuple(sorted(face)); count,oriented=boundary.get(key,(0,face)); boundary[key]=(count+1,oriented)
 require(all(count in (1,2) for count,_ in boundary.values()),'nonmanifold endpoint tetra boundary')
 selected=[]
 for key,(count,face) in boundary.items():
  if count==1 and (all(v in inner for v in key) or key in cap): selected.append((face[0],face[2],face[1]))
 require(len(selected)==432,f'exact cavity-face reconstruction mismatch: {len(selected)}')
 def volume(frame):
  origin=nodes[selected[0][0]][frame]; total=0.0
  for face in selected:
   a,b,c=[tuple(nodes[i][frame][k]-origin[k] for k in range(3)) for i in face]
   cross=(b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0])
   total+=(a[0]*cross[0]+a[1]*cross[1]+a[2]*cross[2])/6.0
  return abs(total)
 v0,v1=volume(0),volume(1)
 require(v0>0 and v1>0,'nonpositive cavity geometry')
 inner_disp=max(math.dist(nodes[i][0],nodes[i][1]) for i in inner)
 return {'first_node':first_node,'node_count':len(nodes),'tetrahedra':len(tets),'cavity_boundary_faces':len(selected),'initial_geometric_cavity_volume_m3':v0,'terminal_geometric_cavity_volume_m3':v1,'volume_change_ml':(v1-v0)*1e6,'volume_change_fraction':v1/v0-1,'max_inner_boundary_displacement_m':inner_disp,'max_wall_displacement_m':float(s['max_displacement_m']),'max_free_wall_displacement_m':float(s['max_free_displacement_m']),'terminal_min_tet_J':min(wall_j),'terminal_max_tet_J':max(wall_j),'terminal_all_tet_J_positive':all(j>0.0 for j in wall_j),'endpoint_log_sha256':sha(logpath)}
summary={'schema':'numi.aortic_wall_pressure_volume_summary.v1','run_plan_sha256':sha(OUT/'run-plan.json'),'attempts':attempts,'visual_verification':False}
if len(attempts)==2 and all(a['exit']==0 for a in attempts):
 on=OUT/'wall_on'; off=OUT/'wall_off'
 onlog=on/'native.log'; offlog=off/'native.log'
 oncsv=csv_rows(on/'output/resting-coupled.csv'); offcsv=csv_rows(off/'output/resting-coupled.csv')
 require(len(oncsv)==32 and len(offcsv)==32,'expected 32 pressure rows in each case')
 require(all(int(r['step'])==i+1 for i,r in enumerate(oncsv)) and all(int(r['step'])==i+1 for i,r in enumerate(offcsv)),'pressure CSV step sequence mismatch')
 require([r['time_s'] for r in oncsv]==[r['time_s'] for r in offcsv],'paired case time grids differ')
 geom=parse_wall(onlog,32)
 def continuity(rows):
  return {'blood_ml_first':float(rows[0]['blood_ml']),'blood_ml_terminal':float(rows[-1]['blood_ml']),'blood_ml_delta':float(rows[-1]['blood_ml'])-float(rows[0]['blood_ml']),'max_abs_blood_error_ml':max(abs(float(r['blood_error_ml'])) for r in rows),'max_abs_body_mass_residual_kg':max(abs(float(r['body_mass_residual_kg'])) for r in rows),'max_abs_oxygen_balance_error_stpd_ml':max(abs(float(r['oxygen_balance_error_stpd_ml'])) for r in rows),'max_abs_co2_balance_error_stpd_ml':max(abs(float(r['co2_balance_error_stpd_ml'])) for r in rows),'PaO2_mmhg_first':float(rows[0]['PaO2_mmhg']),'PaO2_mmhg_terminal':float(rows[-1]['PaO2_mmhg']),'PaCO2_mmhg_first':float(rows[0]['PaCO2_mmhg']),'PaCO2_mmhg_terminal':float(rows[-1]['PaCO2_mmhg'])}
 summary.update({'status':'both_native_cases_passed','wall_on_geometry':geom,'wall_on_pressure_mmhg_first_logged':float(oncsv[0]['aorta_mmhg']),'wall_on_pressure_mmhg_terminal':float(oncsv[-1]['aorta_mmhg']),'wall_off_pressure_mmhg_first_logged':float(offcsv[0]['aorta_mmhg']),'wall_off_pressure_mmhg_terminal':float(offcsv[-1]['aorta_mmhg']),'terminal_wall_on_minus_off_pressure_mmhg':float(oncsv[-1]['aorta_mmhg'])-float(offcsv[-1]['aorta_mmhg']),'wall_on_logged_pressure_delta_mmhg':float(oncsv[-1]['aorta_mmhg'])-float(oncsv[0]['aorta_mmhg']),'accepted_blood_gas_and_mass_closure':{'wall_on':continuity(oncsv),'wall_off':continuity(offcsv)},'pressure_series_comparison':{'wall_on_sha256':sha(on/'output/resting-coupled.csv'),'wall_off_sha256':sha(off/'output/resting-coupled.csv'),'rows':32,'terminal_timestamps_equal':oncsv[-1]['time_s']==offcsv[-1]['time_s']}})
 for name in ('wall_on','wall_off'):
  log=(OUT/name/'native.log').read_text(errors='replace')
  loaded=[]
  for line in log.splitlines():
   if 'libmetalrobo' in line and str(LIB) in line: loaded.append(line)
  require(bool(loaded),f'actual loaded runtime path not confirmed in {name} dyld log')
  require('resting_integrated_body=completed' in log,f'integrated scene completion absent in {name}')
  summary.setdefault('loaded_runtime',{})[name]={'path':str(LIB),'sha256':sha(LIB),'dyld_line':loaded[-1]}
else:
 summary['status']='incomplete_or_failed_native_cases'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
if summary['status']!='both_native_cases_passed': sys.exit(1)
