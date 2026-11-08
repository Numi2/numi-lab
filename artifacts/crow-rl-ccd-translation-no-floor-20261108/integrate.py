"""Actual rebuilt production shader parity and owning native regressions."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();B=R/'build/crow-ccd-translation-build-02';A=R/'artifacts/crow-rl-ccd-translation-no-floor-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
results=[]
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(R/'build/lib'),DYLD_PRINT_LIBRARIES='1',METALROBO_METALLIB=str(R/'build/shaders/MetalRobo.metallib'))
def run(n,cmd,timeout=180):
    t=time.monotonic()
    with (B/f'integrated-{n}.stdout').open('xb') as out,(B/f'integrated-{n}.stderr').open('xb') as err:
        code=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=timeout).returncode
    results.append({'name':n,'command':cmd,'exit_code':code,'seconds':time.monotonic()-t})
    write(B/'integration-execution.json',results);assert code==0,(n,code)
run('grid',[str(B/'cases_probe'),str(R/'build/shaders/MetalRobo.metallib'),str(B/'cases.bin'),str(B/'integrated-grid.bin')])
assert (B/'integrated-grid.bin').read_bytes()==(B/'candidate.bin').read_bytes()
for n in ('clamp','open','zero-friction','refinement'):
    run(n,[str(R/'build/bin/metalrobo_ccd_sweep_probe'),str(R/'build/shaders/MetalRobo.metallib'),str(R/'build/shaders/CCDSweepProbe.metallib'),str(B/f'input/{n}.ccdsweep'),str(B/f'integrated-{n}.bin')])
    assert (B/f'integrated-{n}.bin').read_bytes()==(B/f'{n}.bin').read_bytes()
run('contact',[str(R/'build/bin/metalrobo_metal_world_contact_probe')])
run('object-push',[str(R/'build/bin/metalrobo_crow_object_push_check'),str(B/'integrated-object-push')])
write(A/'integration.json',{'production_grid_all_8192_records_byte_exact':True,'all_four_production_captured_replays_byte_exact':True,
    'native_contact_regression_passed':True,'native_object_push_controls':json.loads((B/'integrated-object-push/summary.json').read_text()),
    'runtime':{p:sha(R/p) for p in ('src/metal/Collision.metal','build/lib/libmetalrobo.dylib','build/shaders/MetalRobo.metallib','build/shaders/CCDSweepProbe.metallib','build/bin/metalrobo_metal_world_contact_probe','build/bin/metalrobo_crow_object_push_check')},
    'commands':results,'scope':'Production implementation matches qualified primitive candidate on tested conditions; full beak hold/lift/release remains to be tested.'})
print(json.dumps({'status':'passed','production_grid_records_exact':8192,'captured_replays_exact':4,'native_contact':'passed','native_object_push':'passed'}))
