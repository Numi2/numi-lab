from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
R=Path.cwd();B=R/'build/crow-ccd-event-status-build-01';A=R/'artifacts/crow-rl-ccd-event-status-20261108'
P=R/'build/crow-beak-hold-build-02';T=R/'.numi/retained-runtime/beak-hold-ccd-fix-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text())
for p,h in plan['inputs'].items():assert sha(R/p)==h,p
assert not subprocess.check_output(['git','status','--porcelain'])
receipts=[]
def run(name,command,expected=0,env=None):
    start=time.monotonic()
    with (B/f'{name}.stdout').open('xb') as out,(B/f'{name}.stderr').open('xb') as err:
        code=subprocess.run(command,stdout=out,stderr=err,env=env,timeout=180).returncode
    receipts.append({'name':name,'command':command,'exit_code':code,'seconds':time.monotonic()-start})
    write(B/'execution.json',receipts);assert code==expected,(name,code)
probe=R/'build/bin/metalrobo_ccd_event_status_probe';current=R/'build/shaders/MetalRobo.metallib'
shutil.copy2(probe,B/'event-status-probe')
shutil.copy2(current,B/'candidate.metallib')
for name,shader,expect in [('baseline',T/'MetalRobo.metallib','baseline'),('candidate',current,'preserve'),('repeat',current,'preserve')]:
    run(name,[str(probe),str(shader),str(B/f'{name}.bin'),expect])
baseline=(B/'baseline.bin').read_bytes();candidate=(B/'candidate.bin').read_bytes()
record=64+64+48+288;lanes=15
assert len(candidate)==len(baseline)==4*lanes*record
healthy_exact=all(baseline[i*lanes*record:i*lanes*record+record]==candidate[i*lanes*record:i*lanes*record+record] for i in range(4))
assert healthy_exact and candidate==(B/'repeat.bin').read_bytes()
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',METALROBO_METALLIB=str(current))
for name,arm,expected in [('clamp',0,1),('open',1,0)]:
    run(name,[str(P/'probe-with-runtime'),str(B/name),str(arm),'1','32'],expected,env)
    files=['trace.tsv']+(['failure-q.f32','failure-v.f32','failure-scene.bin','failure-world-status.bin','failure.json'] if name=='clamp' else ['final-q.f32','final-v.f32','final-scene.bin','scene.bin','summary.json'])
    for n in files:assert (B/name/n).read_bytes()==(P/name/n).read_bytes(),(name,n)
run('contact-regression',[str(T/'contact-probe')],0,env)
import numpy as np
old=np.fromfile(P/'clamp/failure-contact-status.bin',dtype='<u4').reshape(4,72)
new=np.fromfile(B/'clamp/failure-contact-status.bin',dtype='<u4').reshape(4,72)
decoded=[]
for lane in range(4):
    v=new[lane];f=v.view('<f4')
    decoded.append({'lane':lane,'code':int(v[0]),'substep':int(v[3]),'advance_count':int(v[36]),'clustered_impacts':int(v[37]),
      'zero_time_replays':int(v[38]),'first_event_key':[int(v[44]),int(v[45])],'event_times':f[60:64].tolist(),
      'old_advance_count':int(old[lane,36]),'old_zero_time_replays':int(old[lane,38])})
write(A/'results.json',{'healthy_initializer_bytes_exact':healthy_exact,'candidate_repeat_exact':True,
  'failed_lanes_complete_initializer_buffers_unchanged':True,'clamp_trace_accepted_rollback_world_status_and_failure_record_exact':True,
  'open_trace_final_physical_and_scene_trajectory_bytes_exact':True,'native_contact_regression_passed':True,
  'first_failure_records':decoded,'commands':receipts,'candidate_shader_sha256':sha(current),
  'qualification':'Diagnostic preservation only. Clamp remains a rejected physical trial.'})
print(json.dumps({'first_failure_records':decoded,'healthy_exact':healthy_exact}),flush=True)
