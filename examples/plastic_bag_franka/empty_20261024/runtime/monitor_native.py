from pathlib import Path
import json,subprocess
O=Path('/Users/home/Documents/Codex/2026-10-21/i-wa/outputs/plastic_bag_empty_franka_20261024');p=Path('/private/tmp/numi_empty_bag_contact_watch_20261024/raw.json');s=p.read_text();i=s.index('[',s.index('"frames"'))+1;d=json.JSONDecoder();f=None
while i<len(s):
 try:f,j=d.raw_decode(s,i)
 except json.JSONDecodeError:break
 i=j+1
first=json.loads((O/'first_drop/trajectory.json').read_text())['frames'][-1]
if f:print(json.dumps({'step':f['step'],'time_s':f['time_s'],'actual_corner_lift_m':f['front'][-1][1]-first['front'][-1][1],'grip_contact_nodes':f.get('grip_contact_nodes'),'minimum_vertex_table_gap_m':min(v[1] for v in f['front']+f['back'])-.108-.0000375}))
print((O/'contact_watch_latest.json').read_text())
print(subprocess.check_output(['ssh','macmini','tail -1 /Users/n/numi-plastic-bag-empty-franka-20261024/combined/solver_accuracy.csv'],text=True).strip())
