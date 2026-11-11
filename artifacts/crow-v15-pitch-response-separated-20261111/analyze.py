from pathlib import Path
import collections,gzip,json,math,sys
root=Path("artifacts/crow-v15-pitch-response-separated-20261111"); arm=sys.argv[1]; assert arm in ("zero","positive","negative","strong_positive"); A=root/arm;states={};ends=collections.Counter();coverage=collections.Counter();speeds=collections.Counter();caps=collections.Counter();totals=collections.defaultdict(lambda:[0.]*37);overflow=[];errors=[];peak=collections.defaultdict(float)
with gzip.open(A/"trace.log.gz","rt") as f:
 for line in f:
  z=line.split()
  if z and z[0]=="step":
   e,t=int(z[1]),int(z[2])+1;qi=z.index("q");vi=z.index("v");states[e,t]=[list(map(float,z[qi+1:vi])),list(map(float,z[vi+1:]))];ends[e]=max(ends[e],t)
with gzip.open(A/"substeps.log.gz","rt") as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ("failure","CROW_SUBSTEP_ERROR"):errors.append(line.strip())
  if z[0] not in ("CROW_TWO_TOE_SUBSTEP","CROW_TOE_VELOCITY"):continue
  t,k,e=map(int,z[1:4]);key=e,t,k
  if t>ends[e]:continue  # rolled-back attempt is not accepted state
  if z[0]=="CROW_TWO_TOE_SUBSTEP":
   coverage[key]+=1
   if int(z[4])>int(z[5]) or len(z[6:])!=37:overflow.append(key)
   for i,v in enumerate(map(float,z[6:])):totals[e,t][i]+=v
  else:
   speeds[key]+=1;v=list(map(float,z[4:]));maximum=max(abs(w) for w in v[-18:-2]);peak[e]=max(peak[e],maximum)
   if maximum>=11.99:caps[e]+=1
rows=[]
for e in range(8):
 n=ends[e]; vals=[];support=[];nonfoot=0;fall=None
 for t in range(1,n+1):
  if (e,t) not in states:continue
  q,v=states[e,t];pitch=math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3])))); vals.append((t,pitch,v[4],q[2]));s=totals[e,t];left=sum(s[21:27])>=.0002 and sum(s[27:29])<=-.0002;right=sum(s[29:35])>=.0002 and sum(s[35:37])<=-.0002;nonfoot+=abs(s[4])>1e-7
  support.append(min(s[0],s[1])>.0002 and s[0]+s[1]>=.5*.45006*9.81*.02 and left and right and abs(s[4])<=1e-7)
  if fall is None and q[2]<.1:fall=t
 missing=sum(coverage[e,t,k]!=1 for t in range(1,n+1) for k in range(32));missing_v=sum(speeds[e,t,k]!=1 for t in range(1,n+1) for k in range(32));complete=n==550 and len(vals)==550 and not missing and not missing_v and not overflow and not errors
 last=vals[-100:];window=complete and len(last)==100 and all(abs(z[1])<=.25 for z in last) and all(support[-100:]);passed=window and not nonfoot and not caps[e] and fall is None
 rows.append({"lane":e,"gain":{"zero":0,"positive":.15,"negative":-.15,"strong_positive":.30}[arm],"initial_pitch_rad":.04 if e%2==0 else -.04,"accepted_intervals":n,"fall_below_root_height_0_1m":fall,"max_abs_pitch_rad":max((abs(z[1]) for z in vals),default=None),"max_abs_pitch_rate_rad_s":max((abs(z[2]) for z in vals),default=None),"final_pitch_rad":vals[-1][1] if vals else None,"final_pitch_rate_rad_s":vals[-1][2] if vals else None,"final_100_supported_and_pitch_bound":window,"all_run_nonfoot_contact_intervals":nonfoot,"toe_speed_cap_substeps":caps[e],"peak_substep_toe_speed_rad_s":peak[e],"missing_contact_substeps":missing,"missing_velocity_substeps":missing_v,"pass":passed})
parity=[]
for e in range(2,8):
 base=e%2; times=range(1,min(10,ends[e],ends[base])+1);bad=next((t for t in times if states[e,t]!=states[base,t]),None);parity.append({"lane":e,"reference_lane":base,"matched_steps":len(list(times)) if bad is None else bad-1,"first_mismatch":bad})
out={"rows":rows,"paired_prefix":parity,"observer_errors":errors,"overflows":overflow,"limits":"Pitch/rate sampled at control endpoints, velocity caps at physics-substep endpoints. This stationary bench is not an autonomous landing or a biological validation."}
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
for r in rows:print(json.dumps(r))
print("paired prefix",parity,"errors",errors[-3:],"overflow",len(overflow))
