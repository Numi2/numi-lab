from pathlib import Path
import gzip,json,math,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-wing-stroke-kinematics-audit-20261113';A.mkdir(exist_ok=True);p=R/'artifacts/crow-v15-passive-aero-flap-amplitude-20261113/push-wing-20261126-rebound-amplitude/run.stderr.gz';d={}
with gzip.open(p,'rt') as f:
 for line in f:
  z=line.split()
  if not z or z[0] not in ('CROW_WING_EFFORT','CROW_AERO_WRENCH','CROW_KIN_STATE'):continue
  t,k,e=map(int,z[1:4]);
  if e<2 or not (376<=t<=400):continue
  key=e,t,k;x=d.setdefault(key,{})
  if z[0]=='CROW_WING_EFFORT':x['w']=list(map(float,z[4:]))
  elif z[0]=='CROW_KIN_STATE':x['q']=list(map(float,z[6:47]));x['v']=list(map(float,z[47:]));assert len(x['v'])==40
  else:
   b=int(z[4]);x['f'+str(b)]=list(map(float,z[5:]));
assert len(d)==6*25*16 and all(all(k in x for k in ('w','q','v','f5','f6')) for x in d.values())
def solve(matrix, vec):
 a=[row[:] + [b] for row,b in zip(matrix,vec)]
 for i in range(3):
  j=max(range(i,3),key=lambda j:abs(a[j][i]));a[i],a[j]=a[j],a[i];v=a[i][i];assert abs(v)>1e-10;a[i]=[x/v for x in a[i]]
  for j in range(3):
   if j!=i:
    v=a[j][i];a[j]=[x-v*y for x,y in zip(a[j],a[i])]
 return [row[3] for row in a]
def fit(data):
 mat=[[0.]*3 for _ in range(3)];vec=[0.]*3
 for t,y in data:
  v=[1.,math.cos(2*math.pi*6.4*t),math.sin(2*math.pi*6.4*t)]
  for i in range(3):
   vec[i]+=v[i]*y
   for j in range(3):mat[i][j]+=v[i]*v[j]
 c=solve(mat,vec);return {'mean':c[0],'amplitude':math.hypot(c[1],c[2]),'phase_rad':math.atan2(-c[2],c[1])}
rows=[]
for e in range(2,8):
 rowsides=[]
 for side in range(2):
  off=7*side;body=5+side;z=[(t,k,x) for (a,t,k),x in d.items() if a==e];assert len(z)==400
  tr=fit([((t-376)*.02+k*.00125,x['w'][off+1]) for t,k,x in z]);q=fit([((t-376)*.02+k*.00125,x['w'][off+2]) for t,k,x in z]);phase=(q['phase_rad']-tr['phase_rad']+math.pi)%(2*math.pi)-math.pi
  groups=[]
  for sign in (1,-1):
   arr=[x for t,k,x in z if sign*x['v'][8+side]>0];groups.append({'sign_flap_velocity':sign,'n':len(arr),'mean_wing_Fz_N':sum(x['f'+str(body)][2] for x in arr)/len(arr) if arr else None})
  pr=[x['q'][11+side] for t,k,x in z];rowsides.append({'side':side,'target_fit':tr,'actual_fit':q,'actual_minus_target_phase_rad':phase,'actual_over_target_amplitude':q['amplitude']/tr['amplitude'],'pronation_min_rad':min(pr),'pronation_max_rad':max(pr),'force_by_flap_velocity_sign':groups})
 rows.append({'lane':e,'half_amplitude_rad':.25 if e<4 else .45 if e<6 else .65,'sides':rowsides})
res={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'window_ticks':[376,400],'phase_fit_hz':6.4,'rows':rows,'scope':'Posthoc sinusoidal fit and phase-stratified modeled forces in one used seed and correlated lanes; mean forces include moving body and contacts. Flap velocity sign is coordinate convention, not independently labelled biological upstroke/downstroke. Does not prove biological realism or causal effect.'};(A/'analysis.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
