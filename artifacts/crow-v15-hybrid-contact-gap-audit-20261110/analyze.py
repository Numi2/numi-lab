from pathlib import Path
import gzip,json,numpy as np
A=Path('artifacts/crow-v15-hybrid-contact-gap-audit-20261110');P=Path('artifacts/crow-v15-matched-ccd-grip-probe-20261110');old={};new={};poses={};contacts={};ground=None
with gzip.open(P/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if v and v[0]=='trace' and v[1:3]==['1','1']:old[int(v[3])]=v[4:]
with gzip.open(A/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='trace':new[int(v[3])]=v[4:]
  elif v[0]=='pose':poses[(int(v[3]),v[4])]=np.array(list(map(float,v[5:])))
  elif v[0]=='groundpose':ground=np.array(list(map(float,v[3:])))
  elif v[0]=='solvergap':contacts.setdefault((int(v[3]),v[4]),[]).append({'gap_m':float(v[5]),'impulse_ns':float(v[6]),'normal_x':float(v[7]),'normal_z':float(v[8])})
def rot(q):
 x,y,z,w=q;return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def gap(toe,ground):
 a=toe[:3];A=rot(toe[3:]);B=rot(ground[3:]);b=ground[:3]+B@np.array([0,0,-.055]);d=b-a;g=[]
 for u in [*A.T,*B.T]+[np.cross(A[:,i],B[:,j]) for i in range(3) for j in range(3)]:
  n=np.linalg.norm(u)
  if n>1e-8:
   u=u/n;g.append(abs(d@u)-sum(abs(A.T@u)*[.018,.0025,.0092])-sum(abs(B.T@u)*[.025,.72,.055]))
 return float(max(g))
diff=[k for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)];rows=[]
for (t,name),p in sorted(poses.items()):
 c=contacts.get((t,name),[]);rows.append({'step':t,'toe':name,'phase':int(new[t][0]),'accepted_sat_gap_m':gap(p,ground),'solver_contact_count':len(c),'solver_min_gap_m':min((x['gap_m'] for x in c),default=None),'solver_normal_impulse_ns':sum(x['impulse_ns'] for x in c)})
worst=min(rows,key=lambda r:r['accepted_sat_gap_m']);grip=[r for r in rows if r['phase'] in (2,3)];penetrating=[r for r in grip if r['accepted_sat_gap_m']<-.0005];disagree=[r for r in penetrating if r['solver_min_gap_m'] is not None and r['solver_min_gap_m']>=0];unreported=[r for r in penetrating if r['solver_contact_count']==0];coincident=[r for r in rows if r['solver_min_gap_m'] is not None];phase_stats={}
for phase in (2,3,4):
 z=[r for r in rows if r['phase']==phase and r['accepted_sat_gap_m']<-.0005];phase_stats[str(phase)]={'penetrating_samples':len(z),'zero_normal_impulse_samples':sum(r['solver_normal_impulse_ns']<=1e-10 for r in z),'worst_gap_m':min((r['accepted_sat_gap_m'] for r in z),default=None)}
out={'phase_statistics':phase_stats,'paired_point_samples':len(coincident),'median_absolute_gap_difference_m':float(np.median([abs(r['solver_min_gap_m']-r['accepted_sat_gap_m']) for r in coincident])) if coincident else None,'trace_exact':not diff,'trace_samples':len(new),'trace_differences':diff[:20],'worst_accepted_pose':worst,'grip_phase_toe_samples':len(grip),'grip_samples_over_0_5mm':len(penetrating),'penetrating_samples_with_nonnegative_reported_contact_gap':len(disagree),'penetrating_samples_without_reported_toe_contacts':len(unreported),'examples_disagree':disagree[:10],'examples_unreported':unreported[:10],'limitations':'SAT from accepted q pose; solver point separations are contact records from the last collision solve and need not share the same point, axis or time. Absence of a retained record is not proof of absent substep contact. One stationary-perch start, no biological claim.'}
(A/'paired.json').write_text(json.dumps(rows,separators=(',',':'))+'\n');(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
