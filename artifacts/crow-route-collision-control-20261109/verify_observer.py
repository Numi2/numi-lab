from pathlib import Path
import hashlib,json,os,subprocess
R=Path.cwd(); A=R/'artifacts/crow-route-collision-control-20261109'; B=R/'build/crow-route-collision-control-20261109'
ref=json.loads((A/'route-results.json').read_text());cmd=ref['argv'][:]
cmd[cmd.index('--steps')+1]='1200';cmd[cmd.index('--state-trace')+1]=str(B/'uninstrumented-lane1.tsv')
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'};env.pop('CROW_CONTACT_TRACE',None)
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=480)
(A/'observer-off.stdout').write_text(p.stdout);(A/'observer-off.stderr').write_text(p.stderr)
if p.returncode:raise RuntimeError(p.stderr[-4000:])
d=json.loads(p.stdout);a=(B/'tracking-lane0.tsv').read_text().splitlines();b=(B/'uninstrumented-lane1.tsv').read_text().splitlines()
fields=['steps_by_environment','physical_failure_by_environment','maximum_waypoints_by_environment','first_completion_step_by_environment','termination_reason_by_environment']
matches={x:ref['first_episode'][x]==d['first_episode_navigation'][x] for x in fields}
result={'argv':cmd,'trace_samples_compared':min(len(a),len(b))-1,'selected_lane_state_trace_exact':a[:1201]==b[:1201],'first_episode_arrays_exact':matches,'failed_environment_steps':d['failed_environment_steps'],'limitation':'contact candidates are final published substep, not necessarily first fatal microstep'}
(A/'observer-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
