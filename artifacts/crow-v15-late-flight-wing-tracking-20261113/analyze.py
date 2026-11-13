from pathlib import Path
import gzip,json,math,collections,hashlib
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');S=R/'crow-v15-late-flight-aero-observer-20261113';A=R/'crow-v15-late-flight-wing-tracking-20261113';A.mkdir(exist_ok=True);p=S/'push-wing-20261126-rebound-lateaero/run.stderr.gz';d=collections.defaultdict(list);force=collections.defaultdict(list)
with gzip.open(p,'rt') as f:
 for l in f:
  z=l.split()
  if not z or z[0] not in ('CROW_WING_EFFORT','CROW_AERO_FORCE'):continue
  t,k,e=map(int,z[1:4]);
  if e<2 or t<376 or t>400:continue
  if z[0]=='CROW_AERO_FORCE':force[e].append(float(z[7]));continue
  v=list(map(float,z[4:]));assert len(v)==14
  for j in (0,7):
   effort,target,q0,q1,v0,vpre,v1=v[j:j+7];assert all(map(math.isfinite,v[j:j+7]));d[e].append((effort,target,q0,q1,v0,vpre,v1))
rows=[]
for e,x in sorted(d.items()):
 assert len(x)==25*16*2 and len(force[e])==25*16;bound=2.5 if e<4 else 3 if e<6 else 4;diff=[v[1]-v[2] for v in x];work=[v[0]*(v[3]-v[2]) for v in x];rows.append({'lane':e,'bound':bound,'fraction_at_effort_limit':sum(abs(v[0])>=bound-1e-5 for v in x)/len(x),'target_error_rms_rad':math.sqrt(sum(y*y for y in diff)/len(diff)),'target_error_max_rad':max(map(abs,diff)),'max_abs_wing_speed_rad_s':max(abs(v[4]) for v in x),'effort_times_accepted_displacement_positive':sum(max(0,y) for y in work),'effort_times_accepted_displacement_negative_magnitude':sum(max(0,-y) for y in work),'mean_modeled_vertical_aero_force_n':sum(force[e])/len(force[e])})
(A/'analysis.json').write_text(json.dumps({'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Used seed, correlated lanes, both flapping DOFs ticks 376–400. Effort times displacement is an observer-based work proxy, not independently validated muscle work or biological power; target error, saturation and force describe this simulated controller and model only.','rows':rows},indent=2)+'\n')
for x in rows:print(x)
