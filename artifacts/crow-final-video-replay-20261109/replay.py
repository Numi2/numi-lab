from pathlib import Path
import gzip,hashlib,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-final-video-replay-20261109';B=R/'build/crow-final-video-replay-20261109';orig=R/'artifacts/crow-unused-layout-confirmation-round4-20261109';plan=json.loads((orig/'plan.json').read_text())
for path,digest in plan['bound_sha256'].items():
 h=hashlib.sha256((R/path).read_bytes()).hexdigest();assert h==digest,(path,h,digest)
for course in ['reserved-final-i','reserved-final-j']:
 old=json.loads((orig/course/'results.json').read_text());cmd=old['argv'].copy()
 cmd[cmd.index('--state-trace')+1]=str(B/(course+'.tsv'))
 cmd+=['--crow-replay-pack',str(B/(course+'-replay.json'))]
 env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','DYLD_LIBRARY_PATH':str(R/'build/crow-conditional-native-observer-20261109')}
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=900)
 (A/(course+'.stdout')).write_text(p.stdout)
 with (A/(course+'.stderr.gz')).open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
 if p.returncode:raise RuntimeError(p.stderr[-5000:])
 d=json.loads(p.stdout);assert d['first_episode_navigation']==old['first_episode']
 with gzip.open(orig/course/'lane1.tsv.gz','rt') as f:oldlines=f.read().splitlines()
 newlines=(B/(course+'.tsv')).read_text().splitlines()
 assert oldlines==newlines,(course,len(oldlines),len(newlines),next((i for i,(a,b) in enumerate(zip(oldlines,newlines)) if a!=b),None))
 rp=B/(course+'-replay.json');v=json.loads(rp.read_text());frames=v['payload']['frames']
 assert len(frames)==3000 and all(z['step']==i+1 for i,z in enumerate(frames))
 report={'course':course,'seed':old['seed'],'lane':1,'step_count':len(frames),'first_episode_equal':True,'accepted_state_trace_byte_exact':True,'first_episode_supported':old['lanes'][1]['supported_finish'],'replay_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'payload_sha256':v['payload_sha256'],'scope':'native deterministic rerun matched the retained first-episode q and scene trace and outcome exactly; replay includes composed body states for visualization'}
 (A/(course+'-verification.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
