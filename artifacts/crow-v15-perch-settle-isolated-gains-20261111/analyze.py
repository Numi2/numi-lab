from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-perch-settle-isolated-gains-20261111');rows=[]
for D in sorted(A.glob('gain-*')):
 run=json.loads((D/'run-result.json').read_text());last={};fall={};release={};finished={};maxsupport=collections.Counter();caps=collections.Counter();failure=[]
 with gzip.open(D/'trace.log.gz','rt') as f:
  for l in f:
   a=l.split()
   if a and a[0]=='step':
    e,t=map(int,a[1:3]);last[e]=t;z=float(a[a.index('q')+3])
    if z<.1 and e not in fall:fall[e]=t
   elif len(a)>13 and a[0]=='event':
    e,t=map(int,a[1:3]);maxsupport[e]=max(maxsupport[e],int(a[4]))
    if int(a[3]) and e not in release:release[e]=t
    if int(a[11]) and e not in finished:finished[e]=t
 with gzip.open(D/'substeps.log.gz','rt') as f:
  for l in f:
   a=l.split()
   if a and a[0]=='CROW_TOE_VELOCITY':
    t,k,e=map(int,a[1:4]);v=list(map(float,a[4:]));
    if t<=last.get(e,-1)+1 and max(map(abs,v[-18:-2]))>=11.99:caps[e]+=1
   elif a and (a[0]=='failure' or a[0]=='CROW_SUBSTEP_ERROR'):failure.append(l.strip())
 rows.append({'gain_rad_per_m':run['gain'],'exit_code':run['exit_code'],'last_accepted_zero_based_step':last,'first_fall_zero_based_step':fall,'first_release_zero_based_step':release,'first_return_zero_based_step':finished,'max_supported_streak':dict(maxsupport),'toe_cap_substeps_before_end':dict(caps),'failure':failure})
out={'rows':rows,'limitations':'Two deliberately difficult constructed starts per gain, not full sequence qualification. Accepted prefixes only; failed attempt logs may include rolled-back substeps. Baseline differs from earlier failed starts in velocity damping.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
