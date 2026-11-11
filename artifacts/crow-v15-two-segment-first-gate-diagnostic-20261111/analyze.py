from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-two-segment-first-gate-diagnostic-20261111');P=Path('artifacts/crow-v15-two-segment-contact-flight-route-dev-20261111')
p=json.load(gzip.open(P/'replay.json.gz','rt'))['payload'];assert p['environment']==1
with gzip.open(P/'trace.tsv.gz','rt') as f:
 header=next(f);assert 'nq=39' in header and 'scene_stride=13' in header;values=list(map(float,next(f).split()[1:]));left=values[52:55];right=values[65:68];target=[(x+y)/2 for x,y in zip(left,right)]
rows=[]
for f in p['frames']:
 pos=f['q'][:3];d=[pos[i]-target[i] for i in range(3)];rows.append({'step':f['step'],'root_m':pos,'offset_m':d,'distance_m':math.sqrt(sum(x*x for x in d))})
closest=min(rows,key=lambda x:x['distance_m']);cross=min(rows,key=lambda x:abs(x['offset_m'][0]));out={'environment':1,'gate_left_m':left,'gate_right_m':right,'first_waypoint_m':target,'configured_reach_radius_m':.42,'closest_approach':closest,'gate_plane_nearest_sample':cross,'scope':'retained development layout, lane 1 only; scene source ordering is ground then gate-left and gate-right; waypoint is gate midpoint'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
