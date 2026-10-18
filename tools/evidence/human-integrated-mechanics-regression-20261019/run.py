from pathlib import Path
import json,os,subprocess,time,hashlib
c=Path('/Users/n/numi-human-separation-20261018/C');p=c/'current-operator-integration-207';d=c/'integrated-mechanics-runtime-216';b=d/'build';b.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
link=json.load(open(c/'blood-transfer-208/build/build.json'))['link'].copy()
for old,new in [(c/'blood-transfer-208/build/runtime.mm.o',c/'fem-mpm-finalizer-213/build/runtime.mm.o'),(p/'full-build/CMakeFiles/metalrobo.dir/src/metal/MetalArticulatedOperator.mm.o',c/'body-final-gate-order-215/build/operator.o')]:
 assert link.count(str(old))==1;link[link.index(str(old))]=str(new)
lib=b/'libmetalrobo.dylib';link[link.index('-o')+1]=str(lib)
r=subprocess.run(link,capture_output=True,text=True);(b/'link.log').write_text(r.stdout+r.stderr);assert r.returncode==0
(b/'build.json').write_text(json.dumps({'link':link,'library_sha256':sha(lib),'objects':[{'path':x,'sha256':sha(x)} for x in link if x.endswith('.o')]},indent=2)+'\n')
parent=c/'blood-transfer-208/fullscene-prep/fullscene-212-008/attempts/cut-open-mpm-on-8-projection1-compact500mm-dormant-midpoint/invocation.json';old=json.load(open(parent));out=d/'native-run';assert not out.exists()
assetpins={path:h for path,h in old['asset_sha256'].items() if Path(path).suffix in {'.json','.metal','.nhrigid','.nhmyo','.nhbones','.nhcnt','.nheq','.nhtendon','.nhanatomy','.nhtissue','.nhskin'}}
for path,h in assetpins.items():
 assert sha(path)==h,'runtime input changed '+path
argv=old['argv'].copy();argv[4]=str(out);assert sha(argv[0])==old['binary_sha256']
env=old['environment'].copy();env['DYLD_LIBRARY_PATH']=str(b);env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(d/'common-field-failure.json')
run=['/Users/n/numi-human-prep-venv-20261005/bin/python3',old['exclusive_wrapper'],'--',*argv]
receipt={'schema':old['schema'],'scope':'headless full-scene regression after combined FEM finalizer and ordinary body final-gate ordering changes; legacy adapter path unchanged','parent_invocation':str(parent),'parent_invocation_sha256':sha(parent),'argv':argv,'launcher_argv':run,'environment':env,'binary_sha256':sha(argv[0]),'runtime_library_path':str(lib),'runtime_library_sha256':sha(lib),'asset_sha256':assetpins,'compiled_dependencies_note':'Parent includes sparse source headers no longer present at original paths; these are not runtime assets. Existing app binary and new linked runtime objects are pinned separately.','build_receipt':str(b/'build.json'),'build_receipt_sha256':sha(b/'build.json'),'requested_step_count':8,'requested_dt_s':0.00025,'requested_physical_horizon_s':0.002,'visual_verification':False}
(d/'invocation.json').write_text(json.dumps(receipt,indent=2)+'\n');e=os.environ.copy();e.update(env);t=time.monotonic()
with (d/'launcher.log').open('w') as f:r=subprocess.run(run,env=e,stdout=f,stderr=subprocess.STDOUT)
receipt.update(process_exit=r.returncode,wall_seconds=time.monotonic()-t,runtime_library_sha256_after=sha(lib));(d/'invocation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['process_exit','wall_seconds','runtime_library_sha256','binary_sha256']}));raise SystemExit(r.returncode)
