"""Correct the wrapper success-marker check from retained native output only."""
from pathlib import Path
import hashlib,json
root=Path.cwd();out=root/'.numi/runs/american-crow/ccd-generic-contact-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 with (out/name).open('x') as f:json.dump(value,f,indent=2)
receipt=json.loads((out/'native.receipt.json').read_text())
stdout=(out/'native.json').read_text()
assert receipt['exit_code']==0 and not receipt['timed_out'] and not receipt['interrupted']
assert receipt['stdout_sha256']==sha(out/'native.json') and receipt['stderr_sha256']==sha(out/'native.stderr')
assert stdout.startswith('metal_world_contact=ok ') and 'release_gate_40k=pass' in stdout
loaded=[x for x in (out/'native.stderr').read_text().splitlines() if x.startswith('dyld[') and 'libmetalrobo.dylib' in x]
assert loaded and all(x.endswith(str(root/'build/lib/libmetalrobo.dylib')) for x in loaded)
write('analysis-revision.json',{'native_execution_repeated':False,
 'reason':'Wrapper incorrectly expected metal_world_contact=passed; existing executable emits metal_world_contact=ok. Original exit0 and retained output pass the native assertions.',
 'original_script_sha256':sha(root/'build/crow-ccd-distance-build-01/generic_contact.py'),
 'reader_sha256':sha(Path(__file__)), 'receipt_sha256':sha(out/'native.receipt.json')})
write('summary.json',{'status':'passed','existing_native_assertions_passed':True,
 'stdout_sha256':sha(out/'native.json'),'native_execution_repeated':False,
 'reported_wall_steps_per_second':101721,'reported_gpu_steps_per_second':104040,
 'scope':'Existing generic contact benchmark and native engineering assertions. Not a matched performance improvement, crow training throughput, or biological qualification.'})
print('Retained generic contact output passed; wrapper marker corrected without execution')
