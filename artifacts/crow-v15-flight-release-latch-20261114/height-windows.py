from pathlib import Path
import gzip,json,statistics
A=Path('artifacts/crow-v15-flight-release-latch-20261114');sources={'seed26_turn':A/'turn-reference.json.gz','seed27_straight':A/'seed27-straight-reference.json.gz'};p=A/'seed27-turn-reference.json.gz'
if p.exists(): sources['seed27_turn']=p
out={}
for label,path in sources.items():
 s={(x['lane'],x['step']):x for x in json.load(gzip.open(path,'rt'))};rows=[]
 for e in (6,7):
  windows=[]
  for lo,hi,goal in ((1000,1150,.55 if e==6 else .80),(1550,1800,.80 if e==6 else .55)):
   z=[s[e,t]['q'][2] for t in range(lo,hi+1)];windows.append({'steps':[lo,hi],'target_m':goal,'mean_m':statistics.mean(z),'mae_m':statistics.mean(abs(v-goal) for v in z),'min_m':min(z),'max_m':max(z)})
  rows.append({'lane':e,'windows':windows,'mean_height_change_m':windows[1]['mean_m']-windows[0]['mean_m']})
 out[label]=rows
result={'scope':'retrospective reuse of previously reported height windows; scripted used-seed development, not qualified commanded climb or descent','results':out};(A/'height-windows.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
