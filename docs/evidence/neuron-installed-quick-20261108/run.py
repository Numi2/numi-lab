import datetime, hashlib, json, os, pathlib, signal, subprocess
base=pathlib.Path('/Users/numi/numi-neuron-unified-smoke-20261108')
pkg=pathlib.Path('/Users/numi/numi-lab-neuron-unified-relocated-20261108')
source=pathlib.Path('/Users/numi/numi-lab-neuron-unified-20261108')
revision='1b14aa38ad2e6f1bf66b614583b196e1d7e5b922'
paths=[pkg/'bin/numi',pkg/'libexec/numi/neurons',pkg/'bin/metalrobo_neuron_culture_probe',pkg/'lib/libmetalrobo.dylib',pkg/'lib/metalrobo/NumiNeuron.metallib',pkg/'share/numi/REVISION']
def hashes(): return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==revision
assert not subprocess.check_output(['git','status','--short'],cwd=source,text=True).strip()
assert (pkg/'share/numi/REVISION').read_text().strip()==revision
assert not (base/'result.json').exists()
env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NUMANX_'))}
env.update(NUMI_LAB_ROOT=str(pkg), NUMI_WORKSPACE_ROOT=str(base/'workspace'))
cmd=[str(pkg/'bin/numi'),'neurons','protocol','--preset','potter-switch-v1','--network-seed','2056','--sensory-mapping','0','--ablation','none','--quick']
record={'revision':revision,'command':cmd,'timeout_seconds':90,'started_utc':now(),'hash_before':hashes(),'scope':'synthetic quick protocol; no seven-hour checkpoint, manifest, physical or biological qualification'}
child=subprocess.Popen(cmd,cwd=base/'workspace',env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
record['pid']=child.pid
try:
    out,err=child.communicate(timeout=90)
except subprocess.TimeoutExpired:
    record['timeout']=True
    os.killpg(child.pid,signal.SIGTERM)
    try: out,err=child.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL)
        out,err=child.communicate()
(base/'stdout.txt').write_text(out)
(base/'stderr.txt').write_text(err)
record.update(ended_utc=now(),returncode=child.returncode,hash_after=hashes())
record['identities_unchanged']=record['hash_before']==record['hash_after']
try:
    x=json.loads(out)
    record['outcome']={k:x.get(k) for k in ('schema','network_seed','sensory_mapping','windows','accepted_state_fingerprint','simulation_only')}
    record['valid_outcome']=x.get('network_seed')==2056 and x.get('sensory_mapping')==0 and x.get('simulation_only') is True and bool(x.get('accepted_state_fingerprint')) and int(x.get('windows',0))>0
except (ValueError,TypeError): record['valid_outcome']=False
record['success']=child.returncode==0 and not record.get('timeout') and record['identities_unchanged'] and record['valid_outcome']
(base/'result.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:record[k] for k in ('success','returncode','started_utc','ended_utc','identities_unchanged','valid_outcome')},indent=2),flush=True)
raise SystemExit(0 if record['success'] else 1)
