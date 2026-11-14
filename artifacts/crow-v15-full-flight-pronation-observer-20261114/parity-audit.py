from pathlib import Path
import collections,gzip,hashlib,json,sys,math
p=Path('artifacts/crow-v15-full-flight-pronation-observer-20261114');base=Path('artifacts/crow-v15-actor-pronation-slew-20261114/turn-20261126-rebound-slew/run.stderr.gz');new=p/sys.argv[1]/'run.stderr.gz'
def read(path,full):
 h=hashlib.sha256();ph=hashlib.sha256();n=pn=0;coverage=collections.Counter();worst={};errors=[];full_max_effort=0.;full_max_speed=0.;nonfinite=0
 with gzip.open(path,'rb') as f:
  for line in f:
   if line.startswith(b'CROW_SUPPORT_SAMPLE '):h.update(line);n+=1
   elif line.startswith(b'CROW_PRONATION_EFFORT '):
    z=line.split();t,sub,e=map(int,z[1:4]);v=list(map(float,z[4:]));
    if len(v)!=14:errors.append([e,t,sub,'bad field count']);continue
    if 600<=t<=1200:ph.update(line);pn+=1
    if full and 390<=t<=1800:
     coverage[e,t,sub]+=1
     if not all(math.isfinite(x) for x in v):nonfinite+=1
     full_max_effort=max(full_max_effort,abs(v[0]),abs(v[7]));full_max_speed=max(full_max_speed,*(abs(v[j]) for j in (4,5,6,11,12,13)))
     for side,j in enumerate((0,7)):
      r={'lane':e,'side':side,'interval':t,'substep':sub,'effort_Nm':v[j],'target_rad':v[j+1],'position_rad':v[j+2],'tracking_error_rad':abs(v[j+1]-v[j+2])};k=(e,side)
      if k not in worst or r['tracking_error_rad']>worst[k]['tracking_error_rad']:worst[k]=r
   elif b'OBSERVER_ERROR' in line or line.startswith(b'CROW_SUBSTEP_ERROR'):errors.append(line.decode(errors='replace').strip())
 return {'state_sha256':h.hexdigest(),'state_count':n,'overlap_sha256':ph.hexdigest(),'overlap_count':pn,'coverage':coverage,'worst':worst,'errors':errors,'full_max_effort':full_max_effort,'full_max_speed':full_max_speed,'nonfinite':nonfinite}
a=read(base,False);b=read(new,True);expected=8*1411*16;missing=sum(b['coverage'][e,t,s]!=1 for e in range(8) for t in range(390,1801) for s in range(16));extra=sum(v-1 for v in b['coverage'].values() if v>1);w=sorted(b['worst'].values(),key=lambda x:(x['lane'],x['side']));maxerr=max((x['tracking_error_rad'] for x in w),default=None);maxeff=max((abs(x['effort_Nm']) for x in w),default=None)
out={'base_state_count':a['state_count'],'observer_state_count':b['state_count'],'state_hash_equal':a['state_sha256']==b['state_sha256'],'base_state_sha256':a['state_sha256'],'observer_state_sha256':b['state_sha256'],'overlap_counts':[a['overlap_count'],b['overlap_count']],'overlap_hash_equal':a['overlap_sha256']==b['overlap_sha256'],'base_overlap_sha256':a['overlap_sha256'],'observer_overlap_sha256':b['overlap_sha256'],'expected_full_substeps':expected,'observed_unique_full_substeps':len(b['coverage']),'missing_or_nonunique_full_substeps':missing,'duplicate_excess':extra,'observer_errors':a['errors']+b['errors'],'worst_full_window_tracking_by_lane_side':w,'max_tracking_error_rad':maxerr,'max_full_window_effort_Nm':b['full_max_effort'],'max_full_window_speed_rad_s':b['full_max_speed'],'nonfinite_full_samples':b['nonfinite'],'instrumentation_parity_pass':a['state_count']==b['state_count']==8*1801 and a['state_sha256']==b['state_sha256'] and a['overlap_count']==b['overlap_count']==8*601*16 and a['overlap_sha256']==b['overlap_sha256'] and len(b['coverage'])==expected and missing==0 and extra==0 and not a['errors'] and not b['errors'],'tracking_below_008_all_observed':maxerr is not None and maxerr<.08,'effort_and_speed_in_bounds':b['nonfinite']==0 and b['full_max_effort']<=.1501 and b['full_max_speed']<=25.,'scope':'observer parity and sampled full-flight pronation position tracking; not controller qualification or navigation'}
(p/sys.argv[1]/'observer-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
