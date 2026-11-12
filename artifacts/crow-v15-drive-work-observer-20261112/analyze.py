from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-drive-work-observer-20261112');dt=.02/16;arms={};joints=list(range(13,19))+[38,39];toes=list(range(22,38));limits={int(z[0]):float(z[-1]) for z in (line.split() for line in (A/'dof-map.txt').read_text().splitlines())}
for name in ('extension','noextension'):
 assert json.loads((A/(name+'-parity.json')).read_text())['endpoint_mismatches']==0
 samples={};loads={}
 with gzip.open(A/('candidate-20261127-rebound-'+name)/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_DRIVE_WORK '):
    z=line.split();t,k,e,j,flags,qi=map(int,z[1:7]);v=list(map(float,z[7:]));assert (e,t,k,j) not in samples and all(map(math.isfinite,v));samples[e,t,k,j]=(flags,qi,v)
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 340<=t<=365:loads[e,t,k]=sum(float(x) for x in z[8:10])>1e-7
 assert len(samples)==8*26*16*40 and len(loads)==8*26*16
 continuity=max(abs(samples[e,t,k,j][2][1]-samples[e,t,k-1,j][2][2]) for e in range(8) for t in range(340,366) for k in range(1,16) for j in range(40));assert continuity<1e-6,continuity
 rows=[]
 for e in range(8):
  totals=collections.defaultdict(float);peak=0.;cap_counts=collections.Counter()
  for t in range(348,366):
   for k in range(16):
    for j in joints+toes:
     flags,qi,(raw,v0,v1,K,D,cK,cD,cW)=samples[e,t,k,j];assert flags & 32 and qi!=4294967295
     # Under implicit drives, the solver adds h*D+h^2*K to generalized inertia.
     # This translates to an effective effort raw-(D+h*K)*delta_v. Contact
     # and clipping, finite-time velocity approximation and actuator/passive
     # decomposition still prevent interpreting this as true muscle work.
     effective=raw-(cD*D+dt*cK*K)*(v1-v0);peak=max(peak,abs(effective));cap_counts[j]+=int(limits[j]>0 and abs(v1)>=limits[j]-1e-4);group='leg' if j in joints else 'toes';weighted=effective*(v0+v1)*.5*dt;endpoint=effective*v1*dt
     totals[group+'_net_midpoint_J']+=weighted;totals[group+'_positive_midpoint_J']+=max(weighted,0);totals[group+'_negative_midpoint_J']+=min(weighted,0);totals[group+'_net_endpoint_J']+=endpoint
     if loads[e,t,k]:totals[group+'_loaded_midpoint_J']+=weighted
  rows.append({'lane':e,'window_steps':[348,365],'loaded_substeps':sum(loads[e,t,k] for t in range(348,366) for k in range(16)),'peak_abs_inferred_effort_Nm':peak,'velocity_limit_counts':dict(cap_counts),'work_estimate_validated':False,**totals})
 arms[name]={'velocity_continuity_max_abs':continuity,'rows':rows}
out={'timestep_s':dt,'arms':arms,'scope':'Matched development rollouts and exact logged state/contact parity. Estimated numerical generalized drive work, not isolated biological muscle work. Formula accounts for the implicit-drive armature using pre/post-contact velocities; midpoint and endpoint quadratures are reported separately. Knee velocities hit the hard 22 rad/s limit repeatedly in all extension lanes; the resulting projection impulse is not included in the inferred drive effort. Consequently the reported work numbers are invalid as accepted actuator work. Contact impulses, quadrature, clipping, passive and non-drive sources are further limitations. Retain only as diagnostic data; do not use these numbers for a leg-powered claim.'}
(A/'work-estimates.json').write_text(json.dumps(out,indent=2)+'\n')
for e in range(8):
 print(e,*[f"{n}: {arms[n]['rows'][e]['leg_net_midpoint_J']:.5f} J (positive {arms[n]['rows'][e]['leg_positive_midpoint_J']:.5f}, endpoint {arms[n]['rows'][e]['leg_net_endpoint_J']:.5f}, loaded {arms[n]['rows'][e].get('leg_loaded_midpoint_J',0):.5f})" for n in arms])
