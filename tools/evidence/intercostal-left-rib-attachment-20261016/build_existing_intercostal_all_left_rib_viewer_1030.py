from pathlib import Path
import json,hashlib,subprocess,time,sys,signal
W=Path('/Users/n/numi-lab-anatomy-replay-capture-20261016')
R=Path('/Users/n/numi-human-retained-delivery-20261009')
O=R/'native-intercostal-all-left-rib-build-20261016-1030'
BASE=Path('/Users/n/numi-lab-source-allocation-build-1403-003/build-report.json')
PIN=Path(__file__).with_suffix('.pins.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ABI=Path('/Users/n/numi-lab-resting-qualification-publish-1395/matter/include')
PUB=Path('/Users/n/numi-lab-rib-slot-publication-20261016')
paths=[W/'tools/test_intercostal_rib_motion_attachment.mm',W/'apps/NumiHumanRestingSourceAllocation.hpp',*sorted(p for p in ABI.rglob('*') if p.is_file()),Path(__file__),BASE,W/'src/metal/HumanHostPhaseTrace.hpp',W/'matter/tools/human_resting_runtime.hpp',W/'apps/numilab_human_myosim_visual_probe.mm',*sorted((W/'apps').glob('NumiHuman*.hpp'))]
if sys.argv[1:]==['--freeze-reviewed-inputs']:
 assert not PIN.exists();PIN.write_text(json.dumps({str(p):sha(p) for p in paths},indent=2)+'\n');raise SystemExit
assert sys.argv[1:]==['--execute'] and not (O/'build-report.json').exists()
pins=json.loads(PIN.read_text());assert pins=={str(p):sha(p) for p in paths}
O.mkdir(exist_ok=True);(O/'bin').mkdir(exist_ok=True);start=time.monotonic();signal.alarm(600)
report=dict(retained_runtime_abi_headers=str(ABI),retained_runtime_abi_reason='The old frozen GPU runtime must consume its exact retained C++ WorldSource/CompiledWorld/RuntimeStateSnapshot layout. Full source-head runtime rebuild is a separate qualification. The first fresh build incorrectly mixed current Matter headers and retained binary and crashed in RestingRun initialization; that failure is preserved.',source_revision=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),input_sha256=pins,commands=[],native_gpu_confirmation=False)
base=json.loads(BASE.read_text())['commands']
for command in base:
 argv=[a.replace('/Users/n/numi-lab-resting-qualification-publish-1395',str(W)).replace('/Users/numi/numi-lab-source-allocation-build-1403-003',str(O)) for a in command['argv']]
 if command['phase']=='compile':argv.insert(1,'-I'+str(ABI))
 t=time.monotonic()
 with (O/(command['phase']+'.log')).open('wb') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 report['commands'].append(dict(phase=command['phase'],argv=argv,returncode=r.returncode,wall_seconds=time.monotonic()-t))
 (O/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
 assert r.returncode==0,command['phase']
assert pins=={str(p):sha(p) for p in paths}
report.update(binary_sha256=sha(O/'bin/numi-human-native'),elapsed_seconds=time.monotonic()-start)
(O/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='PASS_NATIVE_VIEWER_BUILD_ONLY',binary_sha256=report['binary_sha256'],wall_seconds=report['elapsed_seconds'])),flush=True)
