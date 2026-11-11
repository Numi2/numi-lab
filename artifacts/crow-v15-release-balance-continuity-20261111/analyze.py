from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-release-balance-continuity-20261111');B=Path('artifacts/crow-v15-front-toe-release-20261111')
def read(p):
 traces={};contacts={};toes={};outcome='missing'
 with gzip.open(p/'results.txt.gz','rt') as f:
  for l in f:
   v=l.split()
   if not v:continue
   k=v[0]
   if k=='trace':traces[int(v[3])]=v[4:]
   elif k=='case_result':outcome=' '.join(v[3:])
   elif k=='interval_contact':toes.setdefault(int(v[3]),[]).append(float(v[5]))
   elif k=='allcontact':
    t=int(v[3]);an,bn=v[4:6];nz=float(v[7]);imp=float(v[8]);body=bn if an=='locomotion_ground' else an;up=max((1 if body==bn else -1)*nz*imp,0);d=contacts.setdefault(t,{'left':0.,'right':0.,'other':0.});side=next((s for s in ('left','right') if '_'+s+'_' in body),None)
    if side and ('_foot' in body or '_toe' in body):d[side]+=up
    else:d['other']+=imp
 return traces,contacts,toes,outcome
a,c,p,outcome=read(A);b,_,_,_=read(B);first_release=next((t for t in sorted(b) if int(b[t][0])==4),None);assert first_release is not None;diff=[t for t in range(first_release) if a.get(t)!=b.get(t)];records=[]
for t in sorted(a):
 v=a[t];phase=int(v[0]);qx,qy=float(v[9]),float(v[10]);vel=[float(v[8]),float(v[11]),float(v[12])];tilt=math.acos(max(-1.,min(1.,1-2*(qx*qx+qy*qy))));speed=math.sqrt(sum(z*z for z in vel));d=c.get(t,{'left':0.,'right':0.,'other':0.});supported=min(d['left'],d['right'])>.0002/16 and d['left']+d['right']>=.5*.45004*9.81*(.02/16) and d['other']<=1e-7 and tilt<=.35 and speed<=.12;toe=p.get(t,[]);opened=len(toe)==8 and max(abs(float(z)) for z in v[13:21])<.05 and max(toe)<=1e-7;records.append((t,phase,supported,opened,tilt,speed))
tail=records[-50:];r={'reference_samples_before_release':first_release,'reference_differences':len(diff),'first_reference_differences':diff[:10],'samples':len(records),'outcome':outcome,'release_sample':next((z[0] for z in records if z[1]==4),None),'first_open_supported_sample':next((z[0] for z in records if z[1]==4 and z[2] and z[3]),None),'terminal_50_open_and_supported':len(tail)==50 and all(z[1]==4 and z[2] and z[3] for z in tail),'terminal_50_supported':len(tail)==50 and all(z[2] for z in tail),'terminal_50_open':len(tail)==50 and all(z[3] for z in tail)};r['strict_pass']=len(records)==1400 and not diff and r['release_sample'] is not None and r['terminal_50_open_and_supported'];(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
