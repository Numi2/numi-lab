from pathlib import Path
import json,hashlib,subprocess,os,time,datetime
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=Path(__file__).resolve().parent
A=E.parent/'hand-actuation-qualification-001'
base=json.loads((A/'results/scoped-telemetry004-bilateral-wrist-flexion-reference-initial-80-step10ms/invocation.json').read_text())
for name,expected in base['input_hashes'].items():
 p=Path(name);p=p if p.is_absolute() else R/p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,(name,'hash changed')
manifest_path=E/'scoped-native-build-manifest-006.json'
manifest=json.loads(manifest_path.read_text())
assert hashlib.sha256(Path(manifest['binary_path']).read_bytes()).hexdigest()==manifest['binary_sha256']
results=[]
for label,activation,ids in [('biceps-parity006',.5,[230,231,293,294]),('elbow-flexors006',.5,[230,231,232,233,293,294,295,296])]:
 out=E/'results'/('bilateral-'+label);out.mkdir(parents=True,exist_ok=False)
 argv=base['argv'].copy();argv[0]=manifest['binary_path'];argv[4]=str(out)
 j=argv.index('--muscle-activation');argv[j+1]=str(activation)
 i=0;clean=[]
 while i<len(argv):
  if argv[i]=='--activated-source-muscle-index':i+=2
  else:clean.append(argv[i]);i+=1
 for n in ids:clean+=['--activated-source-muscle-index',str(n)]
 receipt=dict(base,argv=clean,cohort='bilateral_'+label,activation=activation,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),qualification_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Core selected-tendon arm test. Existing free joints, source muscle/tendon laws, bed contacts, accepted-state/replay. Not wholebody FEM/contact or sustained range qualification.')
 receipt['binary_path']=manifest['binary_path'];receipt['binary_sha256']=manifest['binary_sha256'];receipt['input_hashes']={k:v for k,v in base['input_hashes'].items() if k not in [base['binary_path'],base.get('build_source_manifest_path')]};receipt['input_hashes'][manifest['binary_path']]=manifest['binary_sha256'];receipt['input_hashes'][str(manifest_path)]=hashlib.sha256(manifest_path.read_bytes()).hexdigest();receipt['build_source_manifest_path']=str(manifest_path);receipt['build_source_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest();receipt['native_source_sha256']={manifest['source_path']:manifest['source_sha256'],manifest['shadow_header']['path']:manifest['shadow_header']['sha256']}
 (out/'invocation.json').write_text(json.dumps(receipt,indent=2)+'\n')
 env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_HUMAN_RESTING_','NUMI_HUMAN_SUPPORTED_'))}
 for k in ['NUMI_HUMAN_WRIST_CONTOUR_ENABLE','NUMI_HUMAN_WRIST_CONTOUR_SCALE']:env.pop(k,None)
 env['NUMI_HUMAN_SUPPORTED_INITIALIZATION']='0'
 t=time.monotonic()
 try:
  run=subprocess.run(clean,cwd=R,env=env,capture_output=True,text=True,timeout=180);rc=run.returncode;stdout=run.stdout;stderr=run.stderr
 except subprocess.TimeoutExpired as z:
  rc=124;stdout=z.stdout or b'';stderr=z.stderr or b''
  stdout=stdout.decode(errors='replace') if isinstance(stdout,bytes) else stdout;stderr=stderr.decode(errors='replace') if isinstance(stderr,bytes) else stderr
 elapsed=time.monotonic()-t
 (out/'stdout.log').write_text(stdout);(out/'stderr.log').write_text(stderr)
 result={'native_exit_code':rc,'wall_elapsed_seconds':elapsed,'simulated_seconds':.01,'status':'native_completed_pending_arm_assessment' if rc==0 else 'FAIL','scope':receipt['scope'],'invocation_sha256':hashlib.sha256((out/'invocation.json').read_bytes()).hexdigest()}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(label,json.dumps(result),flush=True);results.append(result)
 if rc:break
(E/'pair006-run-summary.json').write_text(json.dumps(results,indent=2)+'\n')
