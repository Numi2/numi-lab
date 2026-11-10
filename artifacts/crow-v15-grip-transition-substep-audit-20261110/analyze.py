from pathlib import Path
import gzip,json
D=Path('artifacts/crow-v15-grip-transition-substep-audit-20261110');ref={};cur={};names={};groups={};seen=set();status=[]
with gzip.open('artifacts/crow-v15-force-normalized-grip-screen-20261110/results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if v and v[0]=='trace' and int(v[1])==1 and int(v[2])==1 and int(v[3])<120:ref[int(v[3])]=v[4:]
with gzip.open(D/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='bodyid':names[int(v[1])]=v[2]
  elif v[0]=='trace':cur[int(v[3])]=v[4:]
assert len(ref)==len(cur)==120
mismatch=[t for t in range(120) if ref[t]!=cur[t]]
with gzip.open(D/'substeps.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='CROW_SUBSTEP_STATUS':seen.add((int(v[1])-1,int(v[2])));status.append(v)
  elif v[0]=='CROW_SUBSTEP_ALL':
   step,sub=int(v[1])-1,int(v[2]);a,b=int(v[4]),int(v[5]);imp,sep,nx,nz=map(float,v[6:10]);an,bn=names.get(a,''),names.get(b,'');toe=an if '_toe' in an and bn=='locomotion_ground' else bn if '_toe' in bn and an=='locomotion_ground' else None
   if toe is None or imp<=0:continue
   oriented=nx if toe==an else -nx;opposing=(oriented>.5 if 'rear_toe' in toe else oriented<-.5)
   for mode in (['all','opposing'] if opposing else ['all']):
    z=groups.setdefault((step,toe,mode),[0.,0.]);z[0]+=imp
    if sub==15:z[1]+=imp
rows=[{'step':t,'toe':name,'mode':mode,'interval_impulse':v[0],'last_substep_impulse':v[1],'interval_force_above_threshold':v[0]>.0002,'last_force_above_threshold':v[1]>.0000125} for (t,name,mode),v in sorted(groups.items())]
miss=[x for x in rows if x['interval_force_above_threshold'] and not x['last_force_above_threshold']]
out={'trace_mismatch_steps':mismatch,'observed_substeps':len(seen),'expected_substeps':1920,'body_names':names,'contacts':rows,'interval_only_counts':{mode:sum(x['mode']==mode for x in miss) for mode in ('all','opposing')}};(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('baseline mismatch',len(mismatch),'substeps observed',len(seen),'of 1920','missed interval threshold',out['interval_only_counts'])
