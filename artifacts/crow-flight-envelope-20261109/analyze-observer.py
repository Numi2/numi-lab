from pathlib import Path
import gzip,json,math,statistics
A=Path('artifacts/crow-flight-envelope-20261109')
out={}
for name in ['yaw-negative','yaw-zero','yaw-positive']:
 v=json.loads((A/('observer-'+name+'-verification.json')).read_text())
 ss=[json.loads(t.split(' ',1)[1]) for t in gzip.open(A/('observer-'+name+'.stderr.gz'),'rt') if t.startswith('CROW_FLIGHT_SAMPLE ')]
 per=[]
 for lane in range(32):
  laneS=[x for x in ss if x['lane']==lane and 1001<=x['step']<=1100]
  if len(laneS)!=100: raise ValueError((name,lane,len(laneS)))
  err=[];mom=[[],[],[]]
  for x in laneS:
   q=x['q'];a=x['applied_filtered_action'];X,Y,Z,W=q[3:7]
   pred=.02*(a[13]*(1-2*(X*X+Y*Y))+a[14]*2*(Y*Z+W*X))
   err.append(abs(a[15]-pred))
   for j in range(3):mom[j].append(a[15+j])
  per.append({'lane':lane,'actuator_projection_within_0_003_fraction':sum(e<=.003 for e in err)/len(err),'maximum_projection_error_Nm':max(err),'mean_root_preaero_Nm':statistics.mean(mom[0]),'mean_wing_moment_Nm':statistics.mean(mom[1]),'mean_root_aero_plus_tail_Nm':statistics.mean(mom[2])})
 fields=['mean_root_preaero_Nm','mean_wing_moment_Nm','mean_root_aero_plus_tail_Nm']
 out[name]={'on_off_trace_identical':v['observer_on_off_trace_identical'],'first_episodes_identical':v['first_episode_identical'],'failed_steps':{k:v['runs'][k]['failed_environment_steps'] for k in ['on','off']},'samples_per_lane':100,'calibration_passed_lanes':sum(r['actuator_projection_within_0_003_fraction']>=.95 for r in per),'median_lane_means_Nm':{k:statistics.median(r[k] for r in per) for k in fields},'maximum_projection_error_Nm':max(r['maximum_projection_error_Nm'] for r in per),'lanes':per}
(A/'observer-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for k,v in out.items():print(k,'calibrated',v['calibration_passed_lanes'],'medians',v['median_lane_means_Nm'],'max projection error',v['maximum_projection_error_Nm'])
