from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-flight-source-diagnostics-20261108';B=R/'build/crow-flight-source-diagnostics-01'
runs=json.load(open(A/'runs.json')); results=[]
for source_idx in [1,3]:
 base=runs[source_idx];cmd=base['argv'][:];name=base['name']
 cmd[cmd.index('--metallib')+1]=str(B/'obstacle-free-v2.metallib')
 target=B/('obstacle-free-v2-'+name+'.tsv');cmd[cmd.index('--state-trace')+1]=str(target)
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(B/'diagnostic-lib')}
 t=time.monotonic();p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=300)
 (A/('obstacle-free-v2-'+name+'.stdout')).write_text(p.stdout)
 (A/('obstacle-free-v2-'+name+'.stderr')).write_text(p.stderr)
 if p.returncode:raise RuntimeError(p.stderr[-4000:])
 data=json.loads(p.stdout)
 events=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CONTACT_TRACE ')]
 a=(B/(name+'.tsv')).read_text().splitlines()[1:];b=target.read_text().splitlines()[1:]
 identical_prefix=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
 result={'name':name,'argv':cmd,'seconds':time.monotonic()-t,'result':data['first_episode_navigation'],'failed_environment_steps':data['failed_environment_steps'],'fatal_contacts':events,'exact_trace_prefix_steps':identical_prefix,
 'shader_sha256':hashlib.sha256((B/'obstacle-free-v2.metallib').read_bytes()).hexdigest(),
 'scope':'diagnostic course-obstacle collision suppression; targets and visible scene retained; ground physics unchanged'}
 results.append(result);(A/'obstacle-free-v2-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,'same prefix',identical_prefix,'failures',data['first_episode_navigation']['physical_failure_count'],
       'safe completion',data['first_episode_navigation']['safe_completion_count'],
       'waypoints',data['first_episode_navigation']['maximum_waypoints_by_environment'],flush=True)
