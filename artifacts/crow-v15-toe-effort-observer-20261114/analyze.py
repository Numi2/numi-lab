from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-toe-effort-observer-20261114');P=Path('artifacts/crow-v15-pitch-normal-support-20261114/trace.log.gz');kinds=('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage')
def load(p):
 d={}
 with gzip.open(p,'rt') as f:
  for line in f:
   z=line.split()
   if z and z[0] in kinds:d.setdefault((int(z[1]),int(z[2]),z[0]),[]).append(z[3:])
 return d
cur=load(A/'trace.log.gz');old=load(P);prior=[t for t in range(450) if any(cur.get((0,t,k))!=old.get((0,t,k)) for k in kinds)];paired=[t for t in range(450) if any(cur.get((0,t,k))!=cur.get((1,t,k)) for k in kinds)]
eff={};vel={};bad=[];errors=[]
with gzip.open(A/'substeps.log.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_SUBSTEP_ERROR','failure'):errors.append(line.strip())
  if z[0] in ('CROW_TOE_V_SUBSTEP','CROW_TOE_DRIVE_EFFORT'):
   t,sub,e=map(int,z[1:4]);a=list(map(float,z[4:]));key=(e,t,sub);d=eff if z[0]=='CROW_TOE_DRIVE_EFFORT' else vel
   if key in d or len(a)!=16 or not all(map(math.isfinite,a)):bad.append([z[0],e,t,sub])
   d[key]=a
missing=[key for key in ((e,t,sub) for e in range(2) for t in range(1,451) for sub in range(32)) if key not in eff or key not in vel]
rows=[]
for e in range(2):
 for lo,hi in ((270,290),(290,300),(300,305),(305,310),(310,330)):
  for group,inds in (('left_front',range(0,6)),('left_rear',range(6,8)),('right_front',range(8,14)),('right_rear',range(14,16))):
   a=[eff[e,t,s][i] for t in range(lo,hi) for s in range(32) for i in inds if (e,t,s) in eff];v=[vel[e,t,s][i] for t in range(lo,hi) for s in range(32) for i in inds if (e,t,s) in vel]
   rows.append({'lane':e,'trace_steps_inclusive_exclusive':[lo,hi],'group':group,'n':len(a),'mean_effort_Nm':sum(a)/len(a) if a else None,'max_abs_effort_Nm':max(map(abs,a),default=None),'fraction_abs_at_least_0_0199_Nm':sum(abs(x)>=.0199 for x in a)/len(a) if a else None,'max_abs_velocity_rad_s':max(map(abs,v),default=None)})
out={'prior_lane0_first_differing_zero_based':next(iter(prior),None),'between_lane_first_differing_zero_based':next(iter(paired),None),'effort_records':len(eff),'velocity_records':len(vel),'missing_records':len(missing),'bad':bad,'errors':errors,'rows':rows,'limitations':'Working effort is the implicit-drive refresh buffer, sampled at end of each encoded substep; pre-contact working effort is not contact impulse and trace epoch alignment needs separate proof. 0.0199 Nm threshold refers to nominal 0.02 Nm model DOF cap, not actual speed-dependent motor envelope.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
