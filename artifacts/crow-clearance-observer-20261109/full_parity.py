from pathlib import Path
import gzip,json,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-clearance-observer-20261109';B=R/'build/crow-clearance-observer-20261109'
prior=R/'artifacts/crow-stage-selected-eval-20261109/training'
d=json.loads((prior/'results.json').read_text());cmd=d['argv'].copy()
cmd[cmd.index('--state-trace')+1]=str(B/'full-parity.tsv')
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600)
if p.returncode:raise RuntimeError(p.stderr[-5000:])
(A/'full-parity.stdout').write_text(p.stdout)
with (A/'full-parity.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
def samples(lines):
 return {(z['lane'],z['step']):z for l in lines if l.startswith('CROW_FLIGHT_SAMPLE ') for z in [json.loads(l.split(' ',1)[1])]}
with gzip.open(prior/'rollout.stderr.gz','rt') as f:old=samples(f)
new=samples(p.stderr.splitlines())
ends=d['first_episode']['steps_by_environment']
keys=[k for k in old if k[1]<=ends[k[0]]]
bad=[k for k in keys if k not in new or old[k]['q']!=new[k]['q'] or old[k]['v']!=new[k]['v']]
records=[json.loads(l.split(' ',1)[1]) for l in p.stderr.splitlines() if l.startswith('CROW_CLEARANCE_SAMPLE ')]
final=[]
for lane,end in enumerate(ends):
 match=[z for z in records if z['lane']==lane and z['end_step'] in (end,end+1)]
 assert match,(lane,end,match)
 z=next((x for x in match if x['end_step']==end),match[0]);z['first_episode_end_step']=end
 final.append(z)
assert keys and not bad and len(final)==16,(len(keys),bad[:10],len(final))
report={'matched_first_episode_samples':len(keys),'qv_mismatches':len(bad),'lane_minima':final,'sampling':'accepted control-step endpoints; four obstacle boxes; perch excluded','minimum_is_not_continuous':True,'termination_alignment':'contact transition can be observed one control submission after first-episode boundary; that accepted contact witness is included when no boundary snapshot exists'}
(A/'full-parity-results.json').write_text(json.dumps(report,indent=2)+'\n')
print('matched',len(keys),'mismatches',len(bad),'clearance rows',len(final),flush=True)
