from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-flight-source-diagnostics-20261108';B=R/'build/crow-flight-source-diagnostics-01'
runs=json.load(open(A/'runs.json'));results=[]
for idx in (1,3):
 base=runs[idx];cmd=base['argv'][:];name=base['name']
 cmd[cmd.index('--metallib')+1]=str(B/'filtered-action.metallib')
 trace=B/('all-contacts-'+name+'.tsv');cmd[cmd.index('--state-trace')+1]=str(trace)
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CROW_CONTACT_TRACE':'1','DYLD_LIBRARY_PATH':str(B/'diagnostic-lib')}
 t=time.monotonic();p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=300)
 (A/('all-contacts-'+name+'.stdout')).write_text(p.stdout);(A/('all-contacts-'+name+'.stderr')).write_text(p.stderr)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 first=json.loads(p.stdout)['first_episode_navigation']
 expected=json.load(open(A/(name+'.stdout')))['first_episode_navigation']
 q0=(B/(name+'.tsv')).read_bytes();q1=trace.read_bytes()
 events=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CONTACT_TRACE ')]
 first_events=[]
 for lane in range(8):
  hits=[e for e in events if e['lane']==lane and e['step']<=first['steps_by_environment'][lane]+1]
  first_events.append(hits[0] if hits else None)
 entry={'name':name,'argv':cmd,'seconds':time.monotonic()-t,'same_first_episode_result':first==expected,
 'same_selected_lane_trajectory':q0==q1,'first_events_by_lane':first_events,
 'first_episode_reasons':first['termination_reason_by_environment'],
 'note':'Published contact pairs are from the final substep. Filtered actions were published by a diagnostic shader; selected-lane physical trace matches uninstrumented execution.'}
 results.append(entry);(A/'all-contacts-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,'same result',entry['same_first_episode_result'],'same trace',entry['same_selected_lane_trajectory'],'first records',sum(x is not None for x in first_events),flush=True)
 if not entry['same_first_episode_result'] or not entry['same_selected_lane_trajectory']:raise RuntimeError('Instrumentation changed matched behavior')
